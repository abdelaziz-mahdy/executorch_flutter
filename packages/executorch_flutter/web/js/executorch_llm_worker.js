/**
 * ExecuTorch LLM worker.
 *
 * Runs the web build of the ExecuTorch LLM runner (wasm/executorch_llm.js +
 * .wasm) off the page's main thread, so the UI keeps painting while tokens are
 * generated. ExecuTorchLLM in package:executorch_flutter starts one worker per
 * loaded model and talks to it with these messages:
 *
 *   {id, type: 'load', modelUrl, tokenizerUrl}  -> {id, type: 'ok'} | error
 *   {id, type: 'generate', prompt, config}      -> {id, type: 'piece', piece}*,
 *                                                  then {id, type: 'done'} | error
 *   {type: 'stop'}                              -> ends the current generation
 *   {id, type: 'reset'}                         -> {id, type: 'ok'}
 *
 * Errors come back as {id, type: 'error', message}.
 *
 * Model and tokenizer URLs are read from the origin private file system
 * (OPFS), directory 'executorch-llm', file name encodeURIComponent(url), when
 * they are there, and fetched otherwise. Apps that want the files to survive a
 * reload download them there first. (OPFS rather than Cache Storage: browsers
 * cap a single cache entry, some at a few hundred MB, below an LLM's size.)
 *
 * The worker yields to its message queue between tokens, so 'stop' takes
 * effect after the token being computed.
 */
'use strict';

const STORE_DIR = 'executorch-llm';
const TOKENIZER_PATH = '/tokenizer.json';

// The wasm/ directory sits next to js/ in a Flutter web app; a page can point
// elsewhere with ?wasm=<url ending in />.
const wasmBase =
  new URL(self.location.href).searchParams.get('wasm') ||
  new URL('../wasm/', self.location.href).href;
importScripts(wasmBase + 'executorch_llm.js');

let et = null; // Emscripten module
let runner = null; // LLMRunner from executorch_llm_web.cpp
let generation = null; // {id, stopped}

// The runtime's recent stderr. ExecuTorch and XNNPACK explain failures there,
// so the lines that look like errors are attached to load errors.
const stderrTail = [];
function onStderr(line) {
  stderrTail.push(line);
  if (stderrTail.length > 200) stderrTail.shift();
  console.debug(line);
}
function withRuntimeErrors(message) {
  const lines = stderrTail.filter((l) => /^E |error|failed|unsupported/i.test(l));
  return lines.length ? `${message}\n${lines.slice(-8).join('\n')}` : message;
}

function post(message) {
  self.postMessage(message);
}

// The stored copy of [url] in the origin private file system, or null.
async function openStored(url) {
  if (!self.navigator || !navigator.storage || !navigator.storage.getDirectory) {
    return null;
  }
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(STORE_DIR);
    const handle = await dir.getFileHandle(encodeURIComponent(url));
    return await handle.getFile();
  } catch (_) {
    return null; // not stored
  }
}

async function openResponse(url) {
  const stored = await openStored(url);
  if (stored) {
    return new Response(stored, {
      headers: { 'Content-Length': String(stored.size) },
    });
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not download ${url} (HTTP ${response.status})`);
  }
  return response;
}

// Copies a response body into a _malloc'd buffer in the Wasm heap. With a
// known, uncompressed length it streams chunk by chunk, so the weights are
// never held twice; otherwise it reads the whole body first.
async function readIntoHeap(response) {
  const length = Number(response.headers.get('Content-Length'));
  const encoded = response.headers.get('Content-Encoding');
  if (!length || (encoded && encoded !== 'identity') || !response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    const ptr = et._malloc(bytes.length);
    if (!ptr) throw new Error(`Out of memory for a ${bytes.length}-byte model`);
    et.HEAPU8.set(bytes, ptr);
    return { ptr, size: bytes.length };
  }

  const ptr = et._malloc(length);
  if (!ptr) throw new Error(`Out of memory for a ${length}-byte model`);
  let offset = 0;
  try {
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.length > length) {
        throw new Error('The model is larger than its Content-Length');
      }
      // HEAPU8 is replaced when the heap grows, so read it on every chunk.
      et.HEAPU8.set(value, ptr + offset);
      offset += value.length;
    }
    if (offset !== length) {
      throw new Error(`The model download stopped at ${offset} of ${length} bytes`);
    }
  } catch (err) {
    et._free(ptr);
    throw err;
  }
  return { ptr, size: length };
}

async function load({ id, modelUrl, tokenizerUrl }) {
  if (runner) throw new Error('A model is already loaded in this worker');
  // Without locateFile Emscripten looks for the .wasm next to this script.
  et = await createExecuTorchLLMModule({
    locateFile: (file) => wasmBase + file,
    printErr: onStderr,
  });

  const tokenizer = await openResponse(tokenizerUrl);
  et.FS.writeFile(TOKENIZER_PATH, new Uint8Array(await tokenizer.arrayBuffer()));

  const model = await readIntoHeap(await openResponse(modelUrl));
  const instance = new et.LLMRunner();
  // On success the runner owns the model buffer; on failure deleting it frees
  // the buffer too.
  const failure = instance.load(model.ptr, model.size, TOKENIZER_PATH);
  if (failure) {
    instance.delete();
    throw new Error(withRuntimeErrors(failure));
  }
  et.FS.unlink(TOKENIZER_PATH);
  runner = instance;
  post({ id, type: 'ok' });
}

function generate({ id, prompt, config }) {
  if (!runner) throw new Error('No model is loaded');
  if (generation) throw new Error('A generation is already in progress');

  const first = runner.begin(
    prompt,
    config.maxNewTokens,
    config.seqLen,
    config.temperature,
    config.echo,
    config.ignoreEos,
    config.numBos,
    config.numEos,
  );
  if (first.error) throw new Error(first.error);
  for (const piece of first.pieces) post({ id, type: 'piece', piece });
  if (first.done) {
    post({ id, type: 'done' });
    return;
  }

  generation = { id, stopped: false };
  const channel = new MessageChannel();
  const finish = (message) => {
    channel.port1.close();
    generation = null;
    post(message);
  };
  // One token per task, so 'stop' and 'reset' messages run in between.
  channel.port1.onmessage = () => {
    if (generation.stopped) {
      finish({ id, type: 'done' });
      return;
    }
    let next;
    try {
      next = runner.step();
    } catch (err) {
      finish({ id, type: 'error', message: String(err && err.message || err) });
      return;
    }
    if (next.error) {
      finish({ id, type: 'error', message: next.error });
      return;
    }
    if (next.piece) post({ id, type: 'piece', piece: next.piece });
    if (next.done) {
      finish({ id, type: 'done' });
      return;
    }
    channel.port2.postMessage(null);
  };
  channel.port2.postMessage(null);
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    switch (message.type) {
      case 'load':
        await load(message);
        break;
      case 'generate':
        generate(message);
        break;
      case 'stop':
        if (generation) generation.stopped = true;
        break;
      case 'reset':
        if (generation) generation.stopped = true;
        if (runner) runner.reset();
        post({ id: message.id, type: 'ok' });
        break;
      default:
        throw new Error(`Unknown message type ${message.type}`);
    }
  } catch (err) {
    post({
      id: message.id,
      type: 'error',
      message: String((err && err.message) || err),
    });
  }
};
