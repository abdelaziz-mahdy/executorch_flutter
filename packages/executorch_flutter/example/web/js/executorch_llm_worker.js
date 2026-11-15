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
 *     (prompt: the UTF-8 bytes, as a Uint8Array)
 *                                                  then {id, type: 'done'} | error
 *   {type: 'stop'}                              -> ends the current generation
 *   {id, type: 'reset'}                         -> {id, type: 'ok'}, after the
 *                                                  current generation ends
 *
 * Errors come back as {id, type: 'error', message, kind}, where kind is
 * 'validation', 'memory', 'state', 'unsupported' or 'inference', the same
 * classes the native runner reports.
 *
 * Where files come from, for each of modelUrl and tokenizerUrl:
 * 1. The origin private file system (OPFS), directory 'executorch-llm', file
 *    name encodeURIComponent(url), when stored there and not empty. Apps that
 *    want the files to survive a reload download them there first. (OPFS
 *    rather than Cache Storage: browsers cap a single cache entry, some at a
 *    few hundred MB, below an LLM's size.)
 * 2. The URL itself.
 * 3. If the URL fails (404, or 403 from object stores that hide missing
 *    keys), a split copy described by url + '.parts.json':
 *    {"size": <bytes>, "parts": ["<name>", ...]}, part names relative to the
 *    manifest, concatenated in order. Static hosts that cap file size (GitHub
 *    Pages deploys refuse files over 100 MB) serve large models this way.
 *
 * The worker yields to its message queue between tokens, so 'stop' takes
 * effect after the token being computed.
 */
'use strict';

const STORE_DIR = 'executorch-llm';
// The tokenizer goes into the Wasm file system under its own file name:
// upstream picks some tokenizer kinds by name (Mistral's tekken.json). A URL
// fragment names the file when it looks like one (has an extension): blob:
// URLs, ?id= downloads, or any URL whose own name doesn't say tekken.json.
function tokenizerPath(url) {
  let name = '';
  try {
    const parsed = new URL(url);
    const last = decodeURIComponent(parsed.pathname.split('/').pop() || '');
    const fragment = parsed.hash.length > 1 ? decodeURIComponent(parsed.hash.slice(1)) : '';
    // A fragment that looks like a file name names the file; one that
    // doesn't (#v2) leaves the path's name in place.
    name = fragment.includes('.') ? fragment : last;
  } catch (_) {
    name = '';
  }
  name = name.replace(/[\/\\]/g, '_');
  if (name === '.' || name === '..') name = '';
  // In a folder of its own: names like `tmp` or `dev` are folders at the root.
  return '/tokenizer/' + (name || 'tokenizer.json');
}

// The wasm/ directory sits next to js/ in a Flutter web app; a page can point
// elsewhere with ?wasm=<url ending in />.
const wasmBase =
  new URL(self.location.href).searchParams.get('wasm') ||
  new URL('../wasm/', self.location.href).href;
importScripts(wasmBase + 'executorch_llm.js');

let et = null; // Emscripten module
let runner = null; // LLMRunner from executorch_llm_web.cpp
let generation = null; // {id, stopped, resets: [id]}

// The runtime's recent stderr. ExecuTorch and XNNPACK explain failures there,
// so the lines that look like errors are attached to load errors.
const stderrTail = [];
let stderrTotal = 0;
function onStderr(line) {
  stderrTotal += 1;
  stderrTail.push(line);
  if (stderrTail.length > 200) stderrTail.shift();
  console.debug(line);
}
// Lines logged from now on, for withRuntimeErrors.
function stderrMark() {
  return stderrTotal;
}

// [message] plus the error lines logged since [mark]: ExecuTorch's
// ("E 00:00:01.234 executorch:..."), XNNPACK's, and the tokenizer library's.
// Only lines from the failing step: an earlier step that succeeded can log
// errors too (each tokenizer format tried before the right one).
function withRuntimeErrors(message, mark) {
  const fresh = stderrTail.slice(Math.max(0, stderrTail.length - (stderrTotal - mark)));
  const lines = fresh
    .filter((l) => /^E \d\d:\d\d:\d\d|^E tokenizers:|^Error in XNNPACK/.test(l))
    .map((l) => (l.length > 300 ? l.slice(0, 300) + '…' : l));
  return lines.length ? `${message}\n${lines.slice(-8).join('\n')}` : message;
}

// Emscripten's file-system errors carry an errno and no message.
function errorText(err) {
  if (err && err.message) return String(err.message);
  if (err && typeof err.errno === 'number') return `File system error (errno ${err.errno})`;
  return String(err);
}

class WorkerError extends Error {
  constructor(message, kind) {
    super(message);
    this.kind = kind;
  }
}

function post(message) {
  self.postMessage(message);
}

// Revalidates any HTTP-cached copy: after a redeploy that changes a file at the
// same URL, a cached part or tokenizer would otherwise pair with new ones.
function fetchFresh(url) {
  return fetch(url, { cache: 'no-cache' });
}

// The stored copy of [url] in the origin private file system, or null. An
// empty file is a download that never finished, so it doesn't count.
async function openStored(url) {
  if (!self.navigator || !navigator.storage || !navigator.storage.getDirectory) {
    return null;
  }
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(STORE_DIR);
    const handle = await dir.getFileHandle(encodeURIComponent(url));
    const file = await handle.getFile();
    return file.size > 0 ? file : null;
  } catch (_) {
    return null; // not stored
  }
}

// A stream that concatenates the parts listed in [manifest], fetched in order.
function partsStream(manifestUrl, manifest) {
  const urls = manifest.parts.map((name) => new URL(name, manifestUrl).href);
  let index = 0;
  let reader = null;
  return new ReadableStream({
    async pull(controller) {
      for (;;) {
        if (!reader) {
          if (index >= urls.length) {
            controller.close();
            return;
          }
          const part = await fetchFresh(urls[index]);
          if (!part.ok || !part.body) {
            throw new Error(`Could not download ${urls[index]} (HTTP ${part.status})`);
          }
          index += 1;
          reader = part.body.getReader();
        }
        const { done, value } = await reader.read();
        if (!done) {
          controller.enqueue(value);
          return;
        }
        reader = null;
      }
    },
    cancel(reason) {
      // Stop the part being downloaded; later parts are never fetched.
      if (reader) return reader.cancel(reason);
      return undefined;
    },
  });
}

// Where a split copy of [url] is described: its path plus `.parts.json`, same
// query, no fragment.
function partsManifestUrl(url) {
  const manifest = new URL(url);
  manifest.hash = '';
  manifest.pathname += '.parts.json';
  return manifest.href;
}

// Opens [url] as a stream of its bytes. `length` is the expected size, and
// `exact` says whether the body must be exactly that long (a stored file or a
// split manifest) or the size is only a hint (Content-Length).
async function openSource(url) {
  const stored = await openStored(url);
  if (stored) return { body: stored.stream(), length: stored.size, exact: true };

  const response = await fetchFresh(url);
  if (response.ok && response.body) {
    // Content-Length is the length on the wire: with a compressed transfer it
    // is smaller than the body, and a cross-origin response hides the
    // Content-Encoding that would say so. It is only a size hint.
    const length = Number(response.headers.get('Content-Length')) || 0;
    return { body: response.body, length, exact: false };
  }
  if (response.body) response.body.cancel().catch(() => {});
  const manifestUrl = partsManifestUrl(url);
  const manifest = await fetchFresh(manifestUrl);
  if (manifest.ok) {
    const parts = await manifest.json();
    return { body: partsStream(manifestUrl, parts), length: parts.size, exact: true };
  }
  throw new Error(`Could not download ${url} (HTTP ${response.status})`);
}

// The largest buffer wasm32 can address; _malloc and _realloc take 32-bit
// sizes, so anything larger would wrap.
const MAX_HEAP_BYTES = 0xFFFFFFFF;

// Copies a stream into a _malloc'd buffer in the Wasm heap, chunk by chunk, so
// the weights are never held twice in full. The buffer starts at the expected
// length and grows if the body turns out longer.
async function readIntoHeap({ body, length, exact }) {
  if (length > MAX_HEAP_BYTES) {
    throw new WorkerError(`A ${length}-byte model does not fit in WebAssembly memory`, 'memory');
  }
  let capacity = length > 0 ? length : 64 * 1024 * 1024;
  let ptr = et._malloc(capacity);
  if (!ptr) throw new WorkerError(`Out of memory for a ${capacity}-byte model`, 'memory');
  let offset = 0;
  let reader = null;
  try {
    reader = body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (exact && offset + value.length > length) {
        throw new Error(`The model is longer than its expected ${length} bytes`);
      }
      if (offset + value.length > capacity) {
        // Double while that's cheap; near the 4 GB limit grow in 256 MB steps,
        // since realloc may need the old and new blocks at once.
        const step = capacity * 2 <= MAX_HEAP_BYTES / 2 ? capacity : 256 * 1024 * 1024;
        const grown = Math.min(Math.max(capacity + step, offset + value.length), MAX_HEAP_BYTES);
        if (grown < offset + value.length) {
          throw new WorkerError('The model does not fit in WebAssembly memory', 'memory');
        }
        const moved = et._realloc(ptr, grown);
        if (!moved) throw new WorkerError(`Out of memory for a ${grown}-byte model`, 'memory');
        ptr = moved;
        capacity = grown;
      }
      // HEAPU8 is replaced when the heap grows, so read it on every chunk.
      et.HEAPU8.set(value, ptr + offset);
      offset += value.length;
    }
    if (exact && offset !== length) {
      throw new Error(`The model stopped at ${offset} of ${length} bytes`);
    }
  } catch (err) {
    et._free(ptr);
    if (reader) reader.cancel().catch(() => {}); // don't keep downloading
    throw err;
  }
  if (offset === 0) {
    et._free(ptr);
    throw new Error('The model file is empty');
  }
  if (offset < capacity) {
    // Grown past the real size (no or compressed Content-Length): give the
    // rest back, since the runner keeps this buffer for the model's lifetime.
    const trimmed = et._realloc(ptr, offset);
    if (trimmed) ptr = trimmed;
  }
  return { ptr, size: offset };
}

async function readAll({ body, length, exact }) {
  const bytes = new Uint8Array(await new Response(body).arrayBuffer());
  if (exact && bytes.length !== length) {
    throw new Error(`The tokenizer is ${bytes.length} bytes, expected ${length}`);
  }
  return bytes;
}

async function load({ id, modelUrl, tokenizerUrl }) {
  if (runner) throw new WorkerError('A model is already loaded in this worker', 'state');
  // Without locateFile Emscripten looks for the .wasm next to this script.
  et = await createExecuTorchLLMModule({
    locateFile: (file) => wasmBase + file,
    printErr: onStderr,
  });

  const instance = new et.LLMRunner();
  try {
    // The tokenizer first, as natively: a bad one fails before the model
    // (hundreds of MB) is downloaded.
    const tokenizerFile = tokenizerPath(tokenizerUrl);
    et.FS.mkdirTree('/tokenizer');
    et.FS.writeFile(tokenizerFile, await readAll(await openSource(tokenizerUrl)));
    let loaded;
    let mark = stderrMark();
    try {
      loaded = instance.loadTokenizer(tokenizerFile);
    } finally {
      et.FS.unlink(tokenizerFile);
    }
    if (loaded.error) throw new WorkerError(withRuntimeErrors(loaded.error, mark), loaded.kind);

    const model = await readIntoHeap(await openSource(modelUrl));
    // From here the runner owns the model buffer, and delete() frees it.
    mark = stderrMark();
    loaded = instance.load(model.ptr, model.size);
    if (loaded.error) throw new WorkerError(withRuntimeErrors(loaded.error, mark), loaded.kind);
  } catch (err) {
    instance.delete();
    throw err;
  }
  runner = instance;
  post({ id, type: 'ok' });
}

function generate({ id, prompt, config }) {
  if (!runner) throw new WorkerError('No model is loaded', 'state');
  if (generation) {
    throw new WorkerError('A generation is already in progress on this runner', 'inference');
  }

  const started = runner.start(
    prompt,
    config.maxNewTokens,
    config.seqLen,
    config.temperature,
    config.ignoreEos,
    config.numBos,
    config.numEos,
  );
  if (started.error) throw new WorkerError(started.error, started.kind);

  // Pieces arrive as bytes; a piece can end partway through a character, so
  // one streaming decoder per generation joins them. Invalid bytes become
  // U+FFFD, and a BOM the model generates is kept, as on native.
  const decoder = new TextDecoder('utf-8', { ignoreBOM: true });
  // Like native's decoder, each byte order mark at the very start of the
  // output (echo included) is a piece of its own.
  let atStart = true;
  const emit = (text) => {
    if (atStart) {
      while (text.startsWith('\uFEFF')) {
        post({ id, type: 'piece', piece: '\uFEFF' });
        text = text.slice(1);
      }
      if (text) atStart = false;
    }
    if (text) post({ id, type: 'piece', piece: text });
  };
  // Echo where upstream does: after the checks, before prefill (so it also
  // shows when prefill fails). Decoded like a piece, so it matches native.
  if (config.echo) emit(new TextDecoder('utf-8', { ignoreBOM: true }).decode(prompt));

  const first = runner.prefill();
  if (first.error) throw new WorkerError(first.error, first.kind);
  emit(decoder.decode(first.bytes, { stream: true }));
  if (first.done) {
    emit(decoder.decode());
    post({ id, type: 'done' });
    return;
  }

  generation = { id, stopped: false, resets: [] };
  const channel = new MessageChannel();
  const finish = (message) => {
    channel.port1.close();
    emit(decoder.decode()); // bytes still held back, before the outcome
    const { resets } = generation;
    generation = null;
    post(message);
    // Like the native runner, a reset requested mid-generation waits for the
    // generation to end and then applies.
    for (const resetId of resets) {
      runner.reset();
      post({ id: resetId, type: 'ok' });
    }
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
      finish({ id, type: 'error', kind: 'inference', message: errorText(err) });
      return;
    }
    if (next.error) {
      finish({ id, type: 'error', kind: next.kind, message: next.error });
      return;
    }
    emit(decoder.decode(next.bytes, { stream: true }));
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
        if (generation) {
          generation.resets.push(message.id);
        } else {
          if (runner) runner.reset();
          post({ id: message.id, type: 'ok' });
        }
        break;
      default:
        throw new WorkerError(`Unknown message type ${message.type}`, 'state');
    }
  } catch (err) {
    post({
      id: message.id,
      type: 'error',
      kind: (err && err.kind) || 'inference',
      message: errorText(err),
    });
  }
};
