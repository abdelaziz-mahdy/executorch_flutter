// Native side of the hosted-LLM downloads: files stream straight to disk.
//
// The vision download service buffers a model in memory, which is fine for a
// few megabytes but not for an LLM, so these files are written in chunks and
// hashed as they arrive.
library;

import 'dart:ffi' show Abi;
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:executorch_flutter/executorch_flutter.dart'
    show executorchVersion;
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import 'llm_catalog.dart';

/// Whether this build can run LLMs at all.
const bool llmSupported = true;

/// Whether [backend] can run on this device. MLX ships for arm64 macOS only.
bool backendRunsHere(String backend) => switch (backend) {
  'xnnpack' => true,
  'mlx' => Abi.current() == Abi.macosArm64,
  _ => false,
};

Future<Directory> _dirFor(LlmEntry entry) async {
  final cache = await getApplicationCacheDirectory();
  return Directory('${cache.path}/llms/$executorchVersion/${entry.id}');
}

/// Local paths of [entry]'s files, keyed by file name, or null if any file is
/// missing. A file only reaches its final name after its hash checked out.
Future<Map<String, String>?> cachedFiles(LlmEntry entry) async {
  final dir = await _dirFor(entry);
  final paths = <String, String>{};
  for (final file in entry.files) {
    final f = File('${dir.path}/${file.name}');
    if (!f.existsSync() || f.lengthSync() != file.size) return null;
    paths[file.name] = f.path;
  }
  return paths;
}

/// Downloads every missing file of [entry] and returns the local paths.
///
/// [onProgress] receives the bytes received so far across all files and the
/// total. A file whose SHA-256 does not match the catalog is deleted and the
/// download fails.
Future<Map<String, String>> downloadLlm(
  LlmEntry entry, {
  void Function(int received, int total)? onProgress,
}) async {
  final dir = await _dirFor(entry);
  await dir.create(recursive: true);
  final total = entry.totalSize;
  var done = 0;
  final paths = <String, String>{};
  final client = http.Client();
  try {
    for (final file in entry.files) {
      final target = File('${dir.path}/${file.name}');
      if (target.existsSync() && target.lengthSync() == file.size) {
        done += file.size;
        onProgress?.call(done, total);
        paths[file.name] = target.path;
        continue;
      }
      final part = File('${target.path}.part');
      final response = await client.send(
        http.Request('GET', Uri.parse(file.url)),
      );
      if (response.statusCode != 200) {
        throw HttpException(
          'Downloading ${file.name} failed (HTTP ${response.statusCode})',
        );
      }
      final digestSink = _DigestSink();
      final hasher = sha256.startChunkedConversion(digestSink);
      final out = part.openWrite();
      try {
        await for (final chunk in response.stream) {
          out.add(chunk);
          hasher.add(chunk);
          done += chunk.length;
          onProgress?.call(done, total);
        }
      } finally {
        await out.close();
      }
      hasher.close();
      final actual = digestSink.value.toString();
      if (actual != file.sha256) {
        await part.delete();
        throw StateError(
          '${file.name} is corrupt: expected SHA-256 ${file.sha256}, '
          'got $actual',
        );
      }
      await part.rename(target.path);
      paths[file.name] = target.path;
    }
  } finally {
    client.close();
  }
  return paths;
}

/// Deletes [entry]'s downloaded files.
Future<void> deleteLlm(LlmEntry entry) async {
  final dir = await _dirFor(entry);
  if (dir.existsSync()) await dir.delete(recursive: true);
}

class _DigestSink implements Sink<Digest> {
  late Digest value;

  @override
  void add(Digest data) => value = data;

  @override
  void close() {}
}

/// [url] resolved against the app's location (only absolute URLs reach this on
/// native).
Uri appUri(String url) => Uri.base.resolve(url);
