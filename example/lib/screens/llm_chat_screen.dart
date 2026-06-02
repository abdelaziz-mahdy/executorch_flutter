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
  final _promptCtrl = TextEditingController();
  final _scrollCtrl = ScrollController();
  final _messages = <_ChatMessage>[];

  ExecuTorchLLM? _llm;
  bool _loading = false;
  bool _generating = false;
  StreamSubscription<String>? _sub;
  String? _error;

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
      _llm = await ExecuTorchLLM.load(
        modelPath: modelPath,
        tokenizerPath: tokenizerPath,
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
        .generate(prompt, config: const GenConfig(maxNewTokens: 512))
        .listen(
          (piece) {
            setState(() => assistant.text += piece);
            _scrollToBottom();
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
      acceptedTypeGroups: [
        XTypeGroup(label: label, extensions: extensions),
      ],
    );
    if (file != null) {
      setState(() => controller.text = file.path);
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
    final color = m.fromUser ? scheme.primaryContainer : scheme.surfaceContainerHighest;
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
