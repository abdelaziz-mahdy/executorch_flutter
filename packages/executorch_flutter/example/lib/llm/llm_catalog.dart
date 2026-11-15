// The hosted LLMs the example can download, read from the models repo.
//
// `<version>/llms.json` sits next to the vision models' `index.json` and lists
// LLM exports uploaded to the same GitHub release. It is a separate file
// because LLMs are exported by hand, not by the models repo's CI, which
// rewrites `index.json` on every run.
library;

import 'dart:convert';

import 'package:executorch_flutter/executorch_flutter.dart'
    show executorchVersion;
import 'package:http/http.dart' as http;

import 'chat_format.dart';
import 'llm_files_stub.dart'
    if (dart.library.io) 'llm_files_io.dart'
    if (dart.library.js_interop) 'llm_files_web.dart'
    show appUri;

/// One downloadable file of a hosted LLM.
class LlmFile {
  const LlmFile({
    required this.name,
    required this.url,
    required this.size,
    required this.sha256,
  });

  factory LlmFile.fromJson(Map<String, dynamic> json) => LlmFile(
    name: json['name'] as String,
    url: json['url'] as String,
    size: json['size'] as int,
    sha256: json['sha256'] as String,
  );

  final String name;
  final String url;
  final int size;
  final String sha256;
}

/// A hosted LLM: the model, its tokenizer, and how to talk to it.
class LlmEntry {
  const LlmEntry({
    required this.id,
    required this.displayName,
    required this.backend,
    required this.license,
    required this.sourceUrl,
    required this.format,
    required this.model,
    required this.tokenizer,
    this.metallib,
  });

  factory LlmEntry.fromJson(Map<String, dynamic> json) {
    final files = json['files'] as Map<String, dynamic>;
    final metallib = files['metallib'] as Map<String, dynamic>?;
    return LlmEntry(
      id: json['id'] as String,
      displayName: json['displayName'] as String,
      backend: json['backend'] as String,
      license: json['license'] as String,
      sourceUrl: json['sourceUrl'] as String,
      format: ChatFormat.fromJson(json['chatFormat'] as Map<String, dynamic>),
      model: LlmFile.fromJson(files['model'] as Map<String, dynamic>),
      tokenizer: LlmFile.fromJson(files['tokenizer'] as Map<String, dynamic>),
      metallib: metallib == null ? null : LlmFile.fromJson(metallib),
    );
  }

  final String id;
  final String displayName;

  /// `xnnpack` (CPU, every native platform) or `mlx` (Apple-Silicon GPU).
  final String backend;
  final String license;
  final String sourceUrl;
  final ChatFormat format;
  final LlmFile model;
  final LlmFile tokenizer;

  /// MLX's Metal kernels; set for `mlx` models only.
  final LlmFile? metallib;

  List<LlmFile> get files => [model, tokenizer, ?metallib];

  int get totalSize => files.fold(0, (sum, f) => sum + f.size);

  String get backendLabel => switch (backend) {
    'xnnpack' => 'CPU (XNNPACK)',
    'mlx' => 'Apple GPU (MLX)',
    _ => backend,
  };
}

/// Catalog location override, injected with `--dart-define=LLM_CATALOG_URL=`.
/// Lets a new catalog be tried before it is merged into the models repo, and
/// lets the web deploy point the app at the catalog it published with the
/// files. A relative URL resolves against the page's base URL.
const String llmCatalogUrlOverride = String.fromEnvironment('LLM_CATALOG_URL');

/// Fetches the LLM catalog for the ExecuTorch version the app was built with.
///
/// LLM exports only run on the runtime they were exported for, so unlike the
/// vision playground there is no version picker.
Future<List<LlmEntry>> fetchLlmCatalog({http.Client? client}) async {
  final base = appUri(
    llmCatalogUrlOverride.isNotEmpty
        ? llmCatalogUrlOverride
        : 'https://raw.githubusercontent.com/abdelaziz-mahdy/'
              'executorch_flutter_models/main/$executorchVersion/llms.json',
  );
  // Past any HTTP cache: the catalog's hashes must match the files now served.
  final url = base.replace(
    queryParameters: {
      ...base.queryParameters,
      't': '${DateTime.now().millisecondsSinceEpoch}',
    },
  );
  final c = client ?? http.Client();
  try {
    final response = await c.get(url);
    if (response.statusCode == 404) return const [];
    if (response.statusCode != 200) {
      throw Exception(
        'Could not load the LLM list (HTTP ${response.statusCode})',
      );
    }
    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return (json['llms'] as List<dynamic>)
        .cast<Map<String, dynamic>>()
        .map(LlmEntry.fromJson)
        .toList();
  } finally {
    if (client == null) c.close();
  }
}
