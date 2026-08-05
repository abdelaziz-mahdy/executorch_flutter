// ExecuTorch Flutter Example - LLM (Gemma 4) streaming chat screen.
//
// Self-contained demo of the on-device LLM API: point it at a local model .pte
// and tokenizer.json, then chat with token-by-token streaming. Kept separate from
// the image-model playground (which loads byte buffers) because LLM weights are
// large and loaded from file paths.
library;

import 'dart:async';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// A single chat turn.
class _ChatMessage {
  _ChatMessage({required this.fromUser, this.text = ''});

  final bool fromUser;
  String text;
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

  // Generation settings (user-adjustable via the tune dialog). Temperature
  // defaults to 0 (greedy / deterministic) — the most reliable output; raise it
  // for more varied responses. Sampling is temperature-only (no top-p/top-k),
  // matching the native runner's GenerationConfig.
  double _temperature = 0;
  int _maxNewTokens = 512;

  bool get _loaded => _llm != null;

  Future<void> _load() async {
    final modelPath = _modelPathCtrl.text.trim();
    final tokenizerPath = _tokenizerPathCtrl.text.trim();
    if (modelPath.isEmpty || tokenizerPath.isEmpty) {
      setState(() => _error = 'Enter both a model path and a tokenizer path.');
      return;
    }
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      await _llm?.dispose();
      _llm = null;
      final metallibPath = _metallibPathCtrl.text.trim();
      _llm = await ExecuTorchLLM.load(
        modelPath: modelPath,
        tokenizerPath: tokenizerPath,
        // Required for MLX (Apple-GPU) models; harmless/ignored otherwise.
        mlxMetallibPath: metallibPath.isEmpty ? null : metallibPath,
      );
    } catch (e) {
      _error = '$e';
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  /// Build the Gemma 4 chat-templated prompt from the conversation so far.
  ///
  /// Gemma 4 (instruction-tuned) expects turn markers — without them the model
  /// emits degenerate output. The native runner tokenizes the prompt verbatim,
  /// so we format here. We re-send the full history each turn (and reset the KV
  /// cache) to keep the prompt self-contained.
  String _buildGemmaPrompt() {
    final sb = StringBuffer('<bos>');
    for (final m in _messages) {
      if (m.fromUser) {
        sb.write('<|turn>user\n${m.text}<turn|>\n');
      } else if (m.text.isNotEmpty) {
        sb.write('<|turn>model\n${m.text}<turn|>\n');
      }
    }
    sb.write('<|turn>model\n'); // generation prompt
    return sb.toString();
  }

  Future<void> _send() async {
    final llm = _llm;
    final text = _promptCtrl.text.trim();
    if (llm == null || text.isEmpty || _generating) return;
    _promptCtrl.clear();

    // Add the user turn, then build the templated prompt from full history.
    setState(() => _messages.add(_ChatMessage(fromUser: true, text: text)));
    final prompt = _buildGemmaPrompt();

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
            // Gemma ends a turn with <turn|> / <end_of_turn>. Some exports embed
            // it in get_eos_ids so the runner stops itself (then we never see it);
            // others (e.g. the XNNPACK export) only declare <eos> and keep
            // emitting <turn|>. So handle the turn-end at the app level: strip the
            // markers from display, and stop generation when one appears. This
            // keeps the package model-agnostic — turn handling lives in the app.
            final hadTurnEnd =
                piece.contains('<turn|>') || piece.contains('<end_of_turn>');
            final clean = piece
                .replaceAll('<turn|>', '')
                .replaceAll('<end_of_turn>', '');
            if (clean.isNotEmpty) {
              setState(() => assistant.text += clean);
              _scrollToBottom();
            }
            if (hadTurnEnd) _stop();
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
    if (mounted) setState(() => _generating = false);
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
        title: const Text('Gemma 4 Chat'),
        actions: [
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
      body: Column(
        children: [
          _buildLoader(context),
          if (_error != null) _buildError(context),
          const Divider(height: 1),
          Expanded(child: _buildMessages(context)),
          _buildComposer(context),
        ],
      ),
    );
  }

  Widget _buildLoader(BuildContext context) {
    return ExpansionTile(
      initiallyExpanded: !_loaded,
      leading: Icon(_loaded ? Icons.check_circle : Icons.folder_open),
      title: Text(_loaded ? 'Model loaded' : 'Load a model'),
      childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      children: [
        const _SetupHelp(),
        const SizedBox(height: 12),
        TextField(
          controller: _modelPathCtrl,
          decoration: InputDecoration(
            labelText: 'Model path (.pte)',
            hintText: '/path/to/gemma-4-E2B-it_xnnpack.pte',
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
            hintText: '/path/to/gemma-4-E2B-it_tokenizer.json',
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
            onPressed: _loading ? null : _load,
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
          _loaded ? 'Say something…' : 'Load a model to start chatting.',
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
        child: Text(text),
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
                  hintText: _loaded ? 'Message Gemma…' : 'Load a model first',
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

/// Explains where the (unbundled) LLM files come from.
///
/// Weights are >1 GB and carry their own license terms, so the package ships
/// neither the model nor the tokenizer. This panel points at the upstream
/// sources instead. Links are copyable rather than tappable to avoid pulling
/// url_launcher into the example just for this.
class _SetupHelp extends StatelessWidget {
  const _SetupHelp();

  static const _docsUrl =
      'https://github.com/abdelaziz-mahdy/executorch_flutter/blob/main/docs/LLM.md';
  static const _exportUrl =
      'https://github.com/abdelaziz-mahdy/executorch_flutter_models/tree/main/python';
  static const _tokenizerUrl = 'https://huggingface.co/google/gemma-4-E2B-it';
  static const _metallibUrl =
      'https://github.com/abdelaziz-mahdy/executorch_native/releases';

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
          Row(
            children: [
              Icon(Icons.info_outline, size: 18, color: scheme.primary),
              const SizedBox(width: 8),
              Text(
                "Don't have the files?",
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            'Gemma 4 weights are over 1 GB and carry their own license, so they '
            'are not bundled. Export them once with the scripts below, then '
            'point this screen at the resulting files.',
            style: theme.textTheme.bodySmall?.copyWith(
              color: scheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 10),
          const _LinkRow(label: 'Setup guide', url: _docsUrl),
          const _LinkRow(label: 'Export scripts', url: _exportUrl),
          const _LinkRow(label: 'Tokenizer (HF)', url: _tokenizerUrl),
          const _LinkRow(label: 'mlx.metallib (MLX only)', url: _metallibUrl),
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
