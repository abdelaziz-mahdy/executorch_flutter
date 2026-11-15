// ExecuTorch Flutter Example - LLM streaming chat screen.
//
// Self-contained demo of the on-device LLM API. Pick a hosted model (downloaded
// once into the app's cache) or point the screen at your own exported .pte and
// tokenizer.json, then chat with token-by-token streaming. Kept separate from
// the image-model playground (which loads byte buffers) because LLM weights are
// large and loaded from file paths.
library;

import 'dart:async';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';

import '../llm/chat_format.dart';
import '../llm/llm_catalog.dart';
import '../llm/llm_files_stub.dart'
    if (dart.library.io) '../llm/llm_files_io.dart'
    if (dart.library.js_interop) '../llm/llm_files_web.dart';

/// A single chat turn.
class _ChatMessage {
  _ChatMessage({required this.fromUser, this.text = ''});

  final bool fromUser;
  String text;
}

/// Download state of one hosted model.
class _DownloadState {
  _DownloadState({this.cached = false});

  bool cached;
  bool downloading = false;
  double progress = 0;
}

/// Streaming chat screen backed by [ExecuTorchLLM].
class LlmChatScreen extends StatefulWidget {
  const LlmChatScreen({super.key});

  @override
  State<LlmChatScreen> createState() => _LlmChatScreenState();
}

class _LlmChatScreenState extends State<LlmChatScreen> {
  final _modelPathCtrl = TextEditingController();
  final _tokenizerPathCtrl = TextEditingController();
  final _metallibPathCtrl = TextEditingController();
  final _promptCtrl = TextEditingController();
  final _scrollCtrl = ScrollController();
  final _messages = <_ChatMessage>[];

  ExecuTorchLLM? _llm;
  bool _loading = false;
  bool _generating = false;
  StreamSubscription<String>? _sub;
  String? _error;

  /// Hosted models that run on this device, or null while loading.
  List<LlmEntry>? _catalog;
  String? _catalogError;
  final _downloads = <String, _DownloadState>{};

  /// The loaded model's name and chat format.
  String? _modelName;

  /// Catalog id of the loaded hosted model; null for the user's own files.
  String? _loadedId;
  ChatFormat _format = ChatFormat.chatmlQwen3;

  /// Chat format picked for "use your own model".
  ChatFormat _ownFormat = ChatFormat.chatmlQwen3;

  // Generation settings (user-adjustable via the tune dialog). Temperature
  // defaults to 0 (greedy / deterministic) — the most reliable output; raise it
  // for more varied responses. Sampling is temperature-only (no top-p/top-k),
  // matching the native runner's GenerationConfig.
  double _temperature = 0;
  int _maxNewTokens = 512;

  bool get _loaded => _llm != null;

  /// One download or load at a time, and none while generating.
  bool get _busy =>
      _loading || _generating || _downloads.values.any((d) => d.downloading);

  @override
  void initState() {
    super.initState();
    if (llmSupported) _loadCatalog();
  }

  Future<void> _loadCatalog() async {
    setState(() {
      _catalog = null;
      _catalogError = null;
    });
    try {
      final all = await fetchLlmCatalog();
      final here = all.where((e) => backendRunsHere(e.backend)).toList();
      for (final entry in here) {
        _downloads[entry.id] = _DownloadState(
          cached: await cachedFiles(entry) != null,
        );
      }
      if (mounted) setState(() => _catalog = here);
    } catch (e) {
      if (mounted) {
        setState(() {
          _catalog = const [];
          _catalogError = '$e';
        });
      }
    }
  }

  /// Downloads [entry] if needed, then loads it.
  Future<void> _useHosted(LlmEntry entry) async {
    final state = _downloads[entry.id]!;
    setState(() {
      state.downloading = true;
      state.progress = 0;
      _error = null;
    });
    Map<String, String> paths;
    try {
      paths = await downloadLlm(
        entry,
        onProgress: (received, total) {
          if (mounted) setState(() => state.progress = received / total);
        },
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          state.downloading = false;
          _error = '$e';
        });
      }
      return;
    }
    if (!mounted) return;
    setState(() {
      state.downloading = false;
      state.cached = true;
    });
    await _load(
      name: entry.displayName,
      id: entry.id,
      format: entry.format,
      modelPath: paths[entry.model.name]!,
      tokenizerPath: paths[entry.tokenizer.name]!,
      metallibPath: entry.metallib == null ? null : paths[entry.metallib!.name],
    );
  }

  Future<void> _deleteHosted(LlmEntry entry) async {
    await deleteLlm(entry);
    if (mounted) setState(() => _downloads[entry.id]!.cached = false);
  }

  Future<void> _loadOwn() async {
    final modelPath = _modelPathCtrl.text.trim();
    final tokenizerPath = _tokenizerPathCtrl.text.trim();
    if (modelPath.isEmpty || tokenizerPath.isEmpty) {
      setState(() => _error = 'Enter both a model path and a tokenizer path.');
      return;
    }
    final metallibPath = _metallibPathCtrl.text.trim();
    await _load(
      name: modelPath.split(RegExp(r'[/\\]')).last,
      format: _ownFormat,
      modelPath: modelPath,
      tokenizerPath: tokenizerPath,
      metallibPath: metallibPath.isEmpty ? null : metallibPath,
    );
  }

  Future<void> _load({
    required String name,
    String? id,
    required ChatFormat format,
    required String modelPath,
    required String tokenizerPath,
    String? metallibPath,
  }) async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      await _llm?.dispose();
      _llm = null;
      _llm = await ExecuTorchLLM.load(
        modelPath: modelPath,
        tokenizerPath: tokenizerPath,
        // Required for MLX (Apple-GPU) models; harmless/ignored otherwise.
        mlxMetallibPath: metallibPath,
      );
      _modelName = name;
      _loadedId = id;
      _format = format;
      _messages.clear();
    } catch (e) {
      _error = '$e';
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _send() async {
    final llm = _llm;
    final text = _promptCtrl.text.trim();
    if (llm == null || text.isEmpty || _generating) return;
    _promptCtrl.clear();

    // Add the user turn, then build the templated prompt from full history.
    setState(() => _messages.add(_ChatMessage(fromUser: true, text: text)));
    final prompt = _format.build([
      for (final m in _messages) ChatTurn(fromUser: m.fromUser, text: m.text),
    ]);

    final assistant = _ChatMessage(fromUser: false);
    setState(() {
      _messages.add(assistant);
      _generating = true;
      _error = null;
    });
    _scrollToBottom();

    // Fresh KV cache each turn — the prompt already carries the full history.
    llm.reset();
    final done = Completer<void>();
    _sub = llm
        .generate(
          prompt,
          config: GenConfig(
            maxNewTokens: _maxNewTokens,
            temperature: _temperature,
          ),
        )
        .listen(
          (piece) {
            // Exports that declare the end-of-turn marker as a stop token stop
            // by themselves; others keep emitting it. Handle it here so the
            // package stays model-agnostic: hide the marker and stop.
            final cleaned = _format.clean(piece);
            if (cleaned.text.isNotEmpty) {
              setState(() => assistant.text += cleaned.text);
              _scrollToBottom();
            }
            if (cleaned.ended) _stop();
          },
          onError: (Object e) {
            setState(() => _error = '$e');
            if (!done.isCompleted) done.complete();
          },
          onDone: () {
            if (!done.isCompleted) done.complete();
          },
          cancelOnError: true,
        );
    await done.future;
    if (mounted) {
      setState(() {
        assistant.text = assistant.text.trim();
        _generating = false;
      });
    }
  }

  Future<void> _pickFile(
    TextEditingController controller,
    String label,
    List<String> extensions,
  ) async {
    final file = await openFile(
      acceptedTypeGroups: [XTypeGroup(label: label, extensions: extensions)],
    );
    if (file != null) {
      setState(() => controller.text = file.path);
    }
  }

  /// Open the generation-settings dialog (temperature, max new tokens).
  ///
  /// These map straight onto the native runner's GenerationConfig. Sampling is
  /// temperature-only — there is intentionally no top-p / top-k (the runner does
  /// not support them). Changes apply to the next message.
  Future<void> _openSettings() async {
    var temperature = _temperature;
    var maxNewTokens = _maxNewTokens;
    final applied = await showDialog<bool>(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setLocal) => AlertDialog(
          title: const Text('Generation settings'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Temperature: ${temperature.toStringAsFixed(2)}'
                '${temperature == 0 ? '  (greedy)' : ''}',
              ),
              Slider(
                value: temperature,
                max: 1.5,
                divisions: 30,
                label: temperature.toStringAsFixed(2),
                onChanged: (v) => setLocal(() => temperature = v),
              ),
              const Text(
                'Higher = more random. 0 is greedy (deterministic).',
                style: TextStyle(fontSize: 12),
              ),
              const SizedBox(height: 16),
              Text('Max new tokens: $maxNewTokens'),
              Slider(
                value: maxNewTokens.toDouble(),
                min: 32,
                max: 2048,
                divisions: 63,
                label: '$maxNewTokens',
                onChanged: (v) => setLocal(() => maxNewTokens = v.round()),
              ),
              const Text(
                'Upper bound on generated tokens per message.',
                style: TextStyle(fontSize: 12),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Apply'),
            ),
          ],
        ),
      ),
    );
    if (applied ?? false) {
      setState(() {
        _temperature = temperature;
        _maxNewTokens = maxNewTokens;
      });
    }
  }

  void _stop() => _llm?.stop();

  void _reset() {
    _llm?.reset();
    setState(_messages.clear);
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 150),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    _llm?.dispose();
    _modelPathCtrl.dispose();
    _tokenizerPathCtrl.dispose();
    _metallibPathCtrl.dispose();
    _promptCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('LLM Chat'),
        actions: [
          if (llmSupported)
            IconButton(
              icon: const Icon(Icons.tune),
              tooltip: 'Generation settings',
              onPressed: _generating ? null : _openSettings,
            ),
          if (_loaded)
            IconButton(
              icon: const Icon(Icons.refresh),
              tooltip: 'New conversation',
              onPressed: _generating ? null : _reset,
            ),
        ],
      ),
      body: llmSupported
          ? LayoutBuilder(
              builder: (context, constraints) => Column(
                children: [
                  // The panel scrolls and takes at most 60% of the height, so
                  // its fields stay reachable on a phone and the chat stays
                  // visible.
                  ConstrainedBox(
                    constraints: BoxConstraints(
                      maxHeight: constraints.maxHeight * 0.6,
                    ),
                    child: SingleChildScrollView(child: _buildLoader(context)),
                  ),
                  if (_error != null) _buildError(context),
                  const Divider(height: 1),
                  Expanded(child: _buildMessages(context)),
                  _buildComposer(context),
                ],
              ),
            )
          : const _UnsupportedNotice(),
    );
  }

  Widget _buildLoader(BuildContext context) {
    return ExpansionTile(
      // Re-keyed on load so the panel collapses once a model is ready.
      key: ValueKey(_modelName),
      initiallyExpanded: !_loaded,
      leading: Icon(_loaded ? Icons.check_circle : Icons.download),
      title: Text(_loaded ? 'Chatting with $_modelName' : 'Choose a model'),
      childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      children: [..._buildHostedList(context), _buildOwnModel(context)],
    );
  }

  List<Widget> _buildHostedList(BuildContext context) {
    final theme = Theme.of(context);
    final catalog = _catalog;
    if (catalog == null) {
      return const [
        Padding(
          padding: EdgeInsets.all(12),
          child: Center(child: CircularProgressIndicator()),
        ),
      ];
    }
    if (_catalogError != null) {
      return [
        ListTile(
          leading: const Icon(Icons.cloud_off),
          title: const Text('Could not load the hosted models'),
          subtitle: Text(_catalogError!),
          trailing: TextButton(
            onPressed: _loadCatalog,
            child: const Text('Retry'),
          ),
        ),
      ];
    }
    if (catalog.isEmpty) {
      return [
        ListTile(
          leading: const Icon(Icons.info_outline),
          title: Text(
            'No hosted models for ExecuTorch $executorchVersion on this device',
          ),
        ),
      ];
    }
    return [
      for (final entry in catalog)
        _HostedModelTile(
          entry: entry,
          state: _downloads[entry.id]!,
          busy: _busy,
          inUse: _loaded && _loadedId == entry.id,
          onUse: () => _useHosted(entry),
          onDelete: () => _deleteHosted(entry),
        ),
      Padding(
        padding: const EdgeInsets.only(top: 4, bottom: 8),
        child: Text(
          'Files download once and stay in the app cache.',
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ),
    ];
  }

  Widget _buildOwnModel(BuildContext context) {
    return ExpansionTile(
      title: const Text('Use your own model'),
      tilePadding: EdgeInsets.zero,
      childrenPadding: const EdgeInsets.only(bottom: 8),
      children: [
        const _SetupHelp(),
        const SizedBox(height: 12),
        DropdownButtonFormField<ChatFormat>(
          initialValue: _ownFormat,
          decoration: const InputDecoration(labelText: 'Chat format'),
          items: [
            for (final f in ChatFormat.presets)
              DropdownMenuItem(value: f, child: Text(f.label)),
          ],
          onChanged: (f) => setState(() => _ownFormat = f ?? _ownFormat),
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _modelPathCtrl,
          decoration: InputDecoration(
            labelText: 'Model path (.pte)',
            hintText: '/path/to/model.pte',
            suffixIcon: IconButton(
              icon: const Icon(Icons.folder_open),
              tooltip: 'Browse for .pte',
              onPressed: () =>
                  _pickFile(_modelPathCtrl, 'ExecuTorch model', ['pte']),
            ),
          ),
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _tokenizerPathCtrl,
          decoration: InputDecoration(
            labelText: 'Tokenizer path (tokenizer.json)',
            hintText: '/path/to/tokenizer.json',
            suffixIcon: IconButton(
              icon: const Icon(Icons.folder_open),
              tooltip: 'Browse for tokenizer',
              onPressed: () =>
                  _pickFile(_tokenizerPathCtrl, 'Tokenizer', ['json']),
            ),
          ),
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _metallibPathCtrl,
          decoration: InputDecoration(
            labelText: 'MLX metallib path (mlx.metallib) — MLX models only',
            hintText: '/path/to/mlx.metallib (leave empty for XNNPACK/CoreML)',
            suffixIcon: IconButton(
              icon: const Icon(Icons.folder_open),
              tooltip: 'Browse for mlx.metallib',
              onPressed: () =>
                  _pickFile(_metallibPathCtrl, 'MLX metallib', ['metallib']),
            ),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: FilledButton.icon(
            onPressed: _busy ? null : _loadOwn,
            icon: _loading
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.download_done),
            label: Text(_loading ? 'Loading…' : 'Load model'),
          ),
        ),
      ],
    );
  }

  Widget _buildError(BuildContext context) {
    return Container(
      width: double.infinity,
      color: Theme.of(context).colorScheme.errorContainer,
      padding: const EdgeInsets.all(12),
      child: Text(
        _error!,
        style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer),
      ),
    );
  }

  Widget _buildMessages(BuildContext context) {
    if (_messages.isEmpty) {
      return Center(
        child: Text(
          _loading
              ? 'Loading the model…'
              : _loaded
              ? 'Say something…'
              : 'Choose a model to start chatting.',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
      );
    }
    return ListView.builder(
      controller: _scrollCtrl,
      padding: const EdgeInsets.all(12),
      itemCount: _messages.length,
      itemBuilder: (context, i) => _bubble(context, _messages[i]),
    );
  }

  Widget _bubble(BuildContext context, _ChatMessage m) {
    final scheme = Theme.of(context).colorScheme;
    final align = m.fromUser ? Alignment.centerRight : Alignment.centerLeft;
    final color = m.fromUser
        ? scheme.primaryContainer
        : scheme.surfaceContainerHighest;
    final text = m.text.isEmpty && !m.fromUser ? '…' : m.text;
    return Align(
      alignment: align,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        constraints: const BoxConstraints(maxWidth: 520),
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(16),
        ),
        // Models answer in Markdown (lists, **bold**, code). It is re-parsed
        // as tokens stream in; half-finished syntax shows as plain text
        // until its closing marker arrives.
        child: m.fromUser
            ? Text(text)
            : MarkdownBody(
                data: text,
                selectable: !_generating,
                styleSheet: _markdownStyle(Theme.of(context)),
              ),
      ),
    );
  }

  Widget _buildComposer(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: _promptCtrl,
                enabled: _loaded && !_generating,
                minLines: 1,
                maxLines: 4,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => _send(),
                decoration: InputDecoration(
                  hintText: _loaded
                      ? 'Message $_modelName…'
                      : 'Load a model first',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                  filled: true,
                ),
              ),
            ),
            const SizedBox(width: 8),
            _generating
                ? IconButton.filledTonal(
                    onPressed: _stop,
                    icon: const Icon(Icons.stop),
                    tooltip: 'Stop',
                  )
                : IconButton.filled(
                    onPressed: _loaded ? _send : null,
                    icon: const Icon(Icons.send),
                    tooltip: 'Send',
                  ),
          ],
        ),
      ),
    );
  }
}

/// Colors for the Markdown in assistant bubbles, from the app's theme.
///
/// flutter_markdown_plus derives its defaults from package:flutter/material's
/// Theme, which an app on package:material_ui does not provide, so it would
/// fall back to light-theme colors. Its style sheet merges field by field, so
/// setting the colors here is enough.
MarkdownStyleSheet _markdownStyle(ThemeData theme) {
  final scheme = theme.colorScheme;
  final text = TextStyle(color: scheme.onSurface);
  final code = BoxDecoration(
    color: scheme.surfaceContainerLow,
    borderRadius: BorderRadius.circular(8),
  );
  return MarkdownStyleSheet(
    p: text,
    a: TextStyle(color: scheme.primary),
    h1: text,
    h2: text,
    h3: text,
    h4: text,
    h5: text,
    h6: text,
    listBullet: text,
    blockquote: text,
    tableHead: text,
    tableBody: text,
    checkbox: TextStyle(color: scheme.primary),
    code: TextStyle(
      color: scheme.onSurface,
      backgroundColor: scheme.surfaceContainerLow,
    ),
    codeblockDecoration: code,
    blockquoteDecoration: BoxDecoration(
      color: scheme.surfaceContainerLow,
      border: Border(left: BorderSide(color: scheme.primary, width: 3)),
    ),
    tableBorder: TableBorder.all(color: scheme.outlineVariant),
    horizontalRuleDecoration: BoxDecoration(
      border: Border(top: BorderSide(color: scheme.outlineVariant)),
    ),
  );
}

/// One hosted model: what it is, and a button to download and chat.
class _HostedModelTile extends StatelessWidget {
  const _HostedModelTile({
    required this.entry,
    required this.state,
    required this.busy,
    required this.inUse,
    required this.onUse,
    required this.onDelete,
  });

  final LlmEntry entry;
  final _DownloadState state;
  final bool busy;

  /// Loaded right now, so its files must not be deleted.
  final bool inUse;
  final VoidCallback onUse;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final sizeMB = (entry.totalSize / (1024 * 1024)).round();
    final details =
        '${entry.backendLabel} · $sizeMB MB · ${entry.license}'
        '${state.cached ? ' · downloaded' : ''}';
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: const Icon(Icons.smart_toy_outlined),
      title: Text(entry.displayName),
      subtitle: state.downloading
          ? Padding(
              padding: const EdgeInsets.only(top: 6),
              child: LinearProgressIndicator(value: state.progress),
            )
          : Text(details),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (state.cached && !state.downloading)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              tooltip: 'Delete downloaded files',
              onPressed: busy || inUse ? null : onDelete,
            ),
          FilledButton(
            onPressed: busy ? null : onUse,
            child: Text(
              state.downloading
                  ? '${(state.progress * 100).round()}%'
                  : state.cached
                  ? 'Chat'
                  : 'Download & chat',
            ),
          ),
        ],
      ),
    );
  }
}

/// Shown where the LLM runner isn't available.
class _UnsupportedNotice extends StatelessWidget {
  const _UnsupportedNotice();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.desktop_windows_outlined,
              size: 48,
              color: theme.colorScheme.primary,
            ),
            const SizedBox(height: 12),
            Text(
              "LLM chat isn't supported on this platform",
              style: theme.textTheme.titleMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'Run the example on Android, iOS, macOS, Windows, Linux or the '
              'web to download a model and chat with it on-device.',
              style: theme.textTheme.bodyMedium,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

/// Points at the guide and export scripts for bringing your own model.
///
/// Links are copyable rather than tappable to avoid pulling url_launcher into
/// the example just for this.
class _SetupHelp extends StatelessWidget {
  const _SetupHelp();

  static const _docsUrl =
      'https://github.com/abdelaziz-mahdy/executorch_flutter/blob/main/packages/executorch_flutter/docs/LLM.md';
  static const _exportUrl =
      'https://github.com/abdelaziz-mahdy/executorch_flutter_models/tree/main/python';

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Export any supported causal LM with the scripts below, then pick '
            'the .pte, its tokenizer.json, and the chat format it was trained '
            'with.',
            style: theme.textTheme.bodySmall?.copyWith(
              color: scheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 10),
          const _LinkRow(label: 'Setup guide', url: _docsUrl),
          const _LinkRow(label: 'Export scripts', url: _exportUrl),
        ],
      ),
    );
  }
}

/// A labelled URL with a copy button.
class _LinkRow extends StatelessWidget {
  const _LinkRow({required this.label, required this.url});

  final String label;
  final String url;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 150,
            child: Text(label, style: Theme.of(context).textTheme.bodySmall),
          ),
          Expanded(
            child: SelectableText(
              url,
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: scheme.primary),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.copy, size: 16),
            tooltip: 'Copy link',
            visualDensity: VisualDensity.compact,
            onPressed: () async {
              await Clipboard.setData(ClipboardData(text: url));
              if (context.mounted) {
                ScaffoldMessenger.of(
                  context,
                ).showSnackBar(SnackBar(content: Text('Copied $label link')));
              }
            },
          ),
        ],
      ),
    );
  }
}
