// Fallback for a platform with neither dart:io nor JS interop: no LLM support.
// Native builds use llm_files_io.dart and web builds llm_files_web.dart.
library;

import 'llm_catalog.dart';

/// Whether this build can run LLMs at all.
const bool llmSupported = false;

bool backendRunsHere(String backend) => false;

Future<Map<String, String>?> cachedFiles(LlmEntry entry) async => null;

Future<Map<String, String>> downloadLlm(
  LlmEntry entry, {
  void Function(int received, int total)? onProgress,
}) => throw UnsupportedError('LLMs are not supported on this platform');

Future<void> deleteLlm(LlmEntry entry) async {}

/// [url] as given.
Uri appUri(String url) => Uri.parse(url);
