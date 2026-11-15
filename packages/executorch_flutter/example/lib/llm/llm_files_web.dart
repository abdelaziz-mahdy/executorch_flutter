// Web side of the hosted-LLM downloads: files stream into the origin private
// file system (OPFS), in the `executorch-llm` directory that ExecuTorchLLM's
// worker reads, so a model downloads once and survives reloads. (Not Cache
// Storage: browsers cap a single cache entry, some at a few hundred MB.)
//
// GitHub release assets cannot be fetched from a browser (no CORS header), so
// the URLs go through the same MODEL_BASE_URL override the vision models use;
// the web deploy copies the files into the published site.
library;

import 'dart:js_interop';

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

Future<web.FileSystemDirectoryHandle> _store() async {
  final root = await web.window.navigator.storage.getDirectory().toDart;
  return root
      .getDirectoryHandle(
        _storeDir,
        web.FileSystemGetDirectoryOptions(create: true),
      )
      .toDart;
}

/// Size of the stored copy of [url], or null if there is none.
Future<int?> _storedSize(
  web.FileSystemDirectoryHandle store,
  String url,
) async {
  try {
    final handle = await store.getFileHandle(_nameOf(url)).toDart;
    final file = await handle.getFile().toDart;
    return file.size;
  } catch (_) {
    return null;
  }
}

/// URLs of [entry]'s files, keyed by file name, or null if any file is not
/// stored yet.
Future<Map<String, String>?> cachedFiles(LlmEntry entry) async {
  final store = await _store();
  final urls = <String, String>{};
  for (final file in entry.files) {
    final url = _urlOf(file);
    if (await _storedSize(store, url) != file.size) return null;
    urls[file.name] = url;
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
  final store = await _store();
  final total = entry.totalSize;
  var done = 0;
  final urls = <String, String>{};
  for (final file in entry.files) {
    final url = _urlOf(file);
    if (await _storedSize(store, url) == file.size) {
      done += file.size;
      onProgress?.call(done, total);
      urls[file.name] = url;
      continue;
    }

    final response = await web.window.fetch(url.toJS).toDart;
    final body = response.body;
    if (!response.ok || body == null) {
      throw StateError(
        'Downloading ${file.name} failed (HTTP ${response.status})',
      );
    }

    final handle = await store
        .getFileHandle(_nameOf(url), web.FileSystemGetFileOptions(create: true))
        .toDart;
    // Writes go to a temporary file that only replaces the stored one on
    // close(), so an interrupted or rejected download leaves nothing behind.
    final writable = await handle.createWritable().toDart;
    final digestSink = _DigestSink();
    final hasher = sha256.startChunkedConversion(digestSink);
    try {
      final reader = body.getReader() as web.ReadableStreamDefaultReader;
      for (;;) {
        final chunk = await reader.read().toDart;
        if (chunk.done) break;
        final data = chunk.value! as JSUint8Array;
        await writable.write(data).toDart;
        final bytes = data.toDart;
        hasher.add(bytes);
        done += bytes.length;
        onProgress?.call(done, total);
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
      await store.removeEntry(_nameOf(url)).toDart;
      rethrow;
    }
    await writable.close().toDart;
    urls[file.name] = url;
  }
  return urls;
}

/// Removes [entry]'s files from OPFS.
Future<void> deleteLlm(LlmEntry entry) async {
  final store = await _store();
  for (final file in entry.files) {
    try {
      await store.removeEntry(_nameOf(_urlOf(file))).toDart;
    } catch (_) {
      // Not stored.
    }
  }
}

class _DigestSink implements Sink<Digest> {
  late Digest value;

  @override
  void add(Digest data) => value = data;

  @override
  void close() {}
}
