// Web side of the hosted-LLM downloads: files stream into the origin private
// file system (OPFS), in the `executorch-llm` directory that ExecuTorchLLM's
// worker reads, so a model downloads once and survives reloads. (Not Cache
// Storage: browsers cap a single cache entry, some at a few hundred MB.)
//
// GitHub release assets cannot be fetched from a browser (no CORS header), so
// the URLs go through the same MODEL_BASE_URL override the vision models use;
// the web deploy copies the files into the published site, splitting the ones
// over GitHub's 100 MB push limit into parts listed in `<file>.parts.json`.
library;

import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'package:crypto/crypto.dart';
import 'package:web/web.dart' as web;

import '../services/model_download_service.dart' show resolveModelUrl;
import 'llm_catalog.dart';

/// Whether this build can run LLMs at all.
const bool llmSupported = true;

/// The web runner has XNNPACK (CPU) only.
bool backendRunsHere(String backend) => backend == 'xnnpack';

/// The OPFS directory ExecuTorchLLM's web worker reads models from; each file
/// is named by its URL, percent-encoded.
const _storeDir = 'executorch-llm';

String _urlOf(LlmFile file) =>
    Uri.base.resolve(resolveModelUrl(file.url)).toString();

String _nameOf(String url) => Uri.encodeComponent(url);

/// Next to each file: the SHA-256 it was verified against, so a file left
/// from another catalog version (same URL, same size) isn't reused.
String _hashNameOf(String url) => '${_nameOf(url)}.sha256';

bool get _canWriteFiles {
  final handle = globalContext['FileSystemFileHandle'];
  if (handle == null) return false;
  final prototype = (handle as JSObject)['prototype'];
  return prototype != null && (prototype as JSObject).has('createWritable');
}

/// The OPFS store, or null where the browser has none this app can write to
/// (private windows, insecure origins, browsers without createWritable). Then
/// the worker downloads the files itself when the model loads, with no
/// progress and no copy kept across reloads.
Future<web.FileSystemDirectoryHandle?> _store() async {
  if (!_canWriteFiles) return null;
  try {
    final root = await web.window.navigator.storage.getDirectory().toDart;
    return await root
        .getDirectoryHandle(
          _storeDir,
          web.FileSystemGetDirectoryOptions(create: true),
        )
        .toDart;
  } catch (_) {
    return null;
  }
}

Future<web.File?> _storedFile(
  web.FileSystemDirectoryHandle store,
  String name,
) async {
  try {
    final handle = await store.getFileHandle(name).toDart;
    return await handle.getFile().toDart;
  } catch (_) {
    return null;
  }
}

/// Whether [file] is stored complete and was verified against its current
/// catalog hash.
Future<bool> _isStored(
  web.FileSystemDirectoryHandle store,
  LlmFile file,
) async {
  final url = _urlOf(file);
  final data = await _storedFile(store, _nameOf(url));
  if (data == null || data.size != file.size) return false;
  final hash = await _storedFile(store, _hashNameOf(url));
  if (hash == null) return false;
  return (await hash.text().toDart).toDart == file.sha256;
}

/// URLs of [entry]'s files, keyed by file name, or null if any file is not
/// stored yet.
Future<Map<String, String>?> cachedFiles(LlmEntry entry) async {
  final store = await _store();
  if (store == null) return null;
  final urls = <String, String>{};
  for (final file in entry.files) {
    if (!await _isStored(store, file)) return null;
    urls[file.name] = _urlOf(file);
  }
  return urls;
}

/// Downloads every missing file of [entry] into OPFS and returns their URLs.
/// [onProgress] receives the bytes received so far across all files and the
/// total. A file whose SHA-256 does not match the catalog is discarded and the
/// download fails.
Future<Map<String, String>> downloadLlm(
  LlmEntry entry, {
  void Function(int received, int total)? onProgress,
}) async {
  final urls = {for (final file in entry.files) file.name: _urlOf(file)};
  final total = entry.totalSize;
  final store = await _store();
  if (store == null) {
    // Nothing to store into: the worker fetches the files at load time.
    onProgress?.call(total, total);
    return urls;
  }

  var done = 0;
  for (final file in entry.files) {
    final url = urls[file.name]!;
    if (await _isStored(store, file)) {
      done += file.size;
      onProgress?.call(done, total);
      continue;
    }
    final start = done;
    try {
      await _download(store, file, url, (received) {
        done = start + received;
        onProgress?.call(done, total);
      });
    } on _StorageFull {
      // Not enough room to keep a copy (a private window's small quota, or a
      // full disk): the worker fetches the files at load time instead.
      onProgress?.call(total, total);
      return urls;
    }
  }
  return urls;
}

/// The browser refused to store more data.
class _StorageFull implements Exception {}

bool _isQuotaError(Object error) {
  // Errors from browser APIs arrive as the JS exception object.
  final jsError = error as JSAny?;
  if (!jsError.isA<JSObject>()) return false;
  final object = jsError as JSObject;
  return object.has('name') && object['name'].dartify() == 'QuotaExceededError';
}

/// Opens the bodies that, in order, make up [url]: the file itself, or its
/// parts when the host serves it split. Each part is fetched only when the
/// previous one has been read.
Future<List<Future<web.ReadableStream> Function()>> _openSource(
  String url,
) async {
  final response = await web.window.fetch(url.toJS).toDart;
  final body = response.body;
  if (response.ok && body != null) return [() async => body];
  // A split copy. Missing files answer 404, or 403 on object stores that
  // hide missing keys, so look for the manifest whatever the failure.
  final manifestUrl = '$url.parts.json';
  final manifest = await web.window.fetch(manifestUrl.toJS).toDart;
  if (!manifest.ok) {
    throw StateError('Downloading $url failed (HTTP ${response.status})');
  }
  final json =
      jsonDecode((await manifest.text().toDart).toDart) as Map<String, dynamic>;
  final base = Uri.parse(manifestUrl);
  return [
    for (final name in (json['parts'] as List).cast<String>())
      () => _openPart(base.resolve(name).toString()),
  ];
}

Future<web.ReadableStream> _openPart(String url) async {
  final response = await web.window.fetch(url.toJS).toDart;
  final body = response.body;
  if (!response.ok || body == null) {
    throw StateError('Downloading $url failed (HTTP ${response.status})');
  }
  return body;
}

Future<void> _download(
  web.FileSystemDirectoryHandle store,
  LlmFile file,
  String url,
  void Function(int received) onReceived,
) async {
  // Drop the old hash first: from here until the new one is written, the
  // file doesn't count as stored.
  await _remove(store, _hashNameOf(url));
  final bodies = await _openSource(url);

  final handle = await store
      .getFileHandle(_nameOf(url), web.FileSystemGetFileOptions(create: true))
      .toDart;
  // Writes go to a temporary file that only replaces the stored one on
  // close(), so an interrupted or rejected download leaves nothing behind.
  final writable = await handle.createWritable().toDart;
  final digestSink = _DigestSink();
  final hasher = sha256.startChunkedConversion(digestSink);
  var received = 0;
  try {
    for (final open in bodies) {
      final body = await open();
      final reader = body.getReader() as web.ReadableStreamDefaultReader;
      for (;;) {
        final chunk = await reader.read().toDart;
        if (chunk.done) break;
        final data = chunk.value! as JSUint8Array;
        try {
          await writable.write(data).toDart;
        } catch (e) {
          if (_isQuotaError(e)) throw _StorageFull();
          rethrow;
        }
        final bytes = data.toDart;
        hasher.add(bytes);
        received += bytes.length;
        onReceived(received);
      }
    }
    hasher.close();
    final actual = digestSink.value.toString();
    if (actual != file.sha256) {
      throw StateError(
        '${file.name} is corrupt: expected SHA-256 ${file.sha256}, '
        'got $actual',
      );
    }
  } catch (_) {
    await writable.abort().toDart;
    // getFileHandle(create: true) left an empty file; don't keep it.
    await _remove(store, _nameOf(url));
    rethrow;
  }
  try {
    await writable.close().toDart;
    await _writeText(store, _hashNameOf(url), file.sha256);
  } catch (e) {
    // The commit or the hash note can hit the quota too.
    await _remove(store, _nameOf(url));
    if (_isQuotaError(e)) throw _StorageFull();
    rethrow;
  }
}

Future<void> _writeText(
  web.FileSystemDirectoryHandle store,
  String name,
  String text,
) async {
  final handle = await store
      .getFileHandle(name, web.FileSystemGetFileOptions(create: true))
      .toDart;
  final writable = await handle.createWritable().toDart;
  await writable.write(text.toJS).toDart;
  await writable.close().toDart;
}

Future<void> _remove(web.FileSystemDirectoryHandle store, String name) async {
  try {
    await store.removeEntry(name).toDart;
  } catch (_) {
    // Not stored.
  }
}

/// Removes [entry]'s files from OPFS.
Future<void> deleteLlm(LlmEntry entry) async {
  final store = await _store();
  if (store == null) return;
  for (final file in entry.files) {
    final url = _urlOf(file);
    await _remove(store, _hashNameOf(url));
    await _remove(store, _nameOf(url));
  }
}

class _DigestSink implements Sink<Digest> {
  late Digest value;

  @override
  void add(Digest data) => value = data;

  @override
  void close() {}
}
