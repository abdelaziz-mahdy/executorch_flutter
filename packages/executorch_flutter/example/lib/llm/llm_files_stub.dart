// Web side of the hosted-LLM downloads. There is no WebAssembly build of the
// LLM runner yet, so the chat screen shows a notice instead of these.
library;

import 'llm_catalog.dart';

/// Whether this build can run LLMs at all.
const bool llmSupported = false;

bool backendRunsHere(String backend) => false;

Future<Map<String, String>?> cachedFiles(LlmEntry entry) async => null;

Future<Map<String, String>> downloadLlm(
  LlmEntry entry, {
  void Function(int received, int total)? onProgress,
}) => throw UnsupportedError('LLMs are not supported on web yet');

Future<void> deleteLlm(LlmEntry entry) async {}
