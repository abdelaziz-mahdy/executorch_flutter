// How a chat model expects its conversation to be written out as text.
//
// The native runner tokenizes the prompt verbatim, so the app has to wrap each
// turn in the model's own markers. Instruction-tuned models produce degenerate
// output without them. A format is plain data: the hosted model catalog ships
// one per model, and the "use your own model" panel offers the presets below.
library;

/// One turn of a conversation.
class ChatTurn {
  const ChatTurn({required this.fromUser, required this.text});

  final bool fromUser;
  final String text;
}

/// A model's chat template, reduced to the parts the example needs.
///
/// Templates use `{text}` as the placeholder for the turn's content.
class ChatFormat {
  const ChatFormat({
    required this.id,
    required this.label,
    this.prefix = '',
    required this.user,
    required this.assistant,
    required this.generationPrompt,
    this.stop = const [],
  });

  factory ChatFormat.fromJson(Map<String, dynamic> json) => ChatFormat(
    id: json['id'] as String,
    label: json['label'] as String? ?? json['id'] as String,
    prefix: json['prefix'] as String? ?? '',
    user: json['user'] as String,
    assistant: json['assistant'] as String,
    generationPrompt: json['generationPrompt'] as String,
    stop: (json['stop'] as List<dynamic>? ?? const []).cast<String>(),
  );

  /// Stable identifier, e.g. `chatml-qwen3`.
  final String id;

  /// Name shown in the format picker.
  final String label;

  /// Written once, before the first turn (e.g. a BOS marker).
  final String prefix;

  /// A user turn.
  final String user;

  /// A finished assistant turn, used for the conversation history.
  final String assistant;

  /// Opens the assistant turn the model should now write.
  final String generationPrompt;

  /// Markers that end the assistant's turn. Some exports declare them as stop
  /// tokens, so the runner stops by itself; others keep emitting them, so the
  /// app strips them and stops generation when one appears.
  final List<String> stop;

  /// Builds the prompt for the whole conversation so far. The runner's KV
  /// cache is reset before every message, so the prompt carries the history.
  String build(List<ChatTurn> turns) {
    final sb = StringBuffer(prefix);
    for (final turn in turns) {
      if (turn.fromUser) {
        sb.write(user.replaceAll('{text}', turn.text));
      } else if (turn.text.isNotEmpty) {
        sb.write(assistant.replaceAll('{text}', turn.text));
      }
    }
    sb.write(generationPrompt);
    return sb.toString();
  }

  /// Removes stop markers from a generated [piece]. `ended` is true when the
  /// piece contained one, meaning the assistant's turn is over.
  ({String text, bool ended}) clean(String piece) {
    var text = piece;
    var ended = false;
    for (final marker in stop) {
      if (text.contains(marker)) {
        ended = true;
        text = text.replaceAll(marker, '');
      }
    }
    return (text: text, ended: ended);
  }

  /// ChatML as Qwen3 writes it, with thinking turned off: the empty think
  /// block in the generation prompt is what Qwen3's own template inserts for
  /// `enable_thinking=False`.
  static const chatmlQwen3 = ChatFormat(
    id: 'chatml-qwen3',
    label: 'ChatML (Qwen3)',
    user: '<|im_start|>user\n{text}<|im_end|>\n',
    assistant: '<|im_start|>assistant\n{text}<|im_end|>\n',
    generationPrompt: '<|im_start|>assistant\n<think>\n\n</think>\n\n',
    stop: ['<|im_end|>', '<|endoftext|>'],
  );

  /// Gemma 4. The turn markers are the literal `<|turn>` and `<turn|>`
  /// tokens, not `<start_of_turn>`/`<end_of_turn>`.
  static const gemma4 = ChatFormat(
    id: 'gemma4',
    label: 'Gemma 4',
    prefix: '<bos>',
    user: '<|turn>user\n{text}<turn|>\n',
    assistant: '<|turn>model\n{text}<turn|>\n',
    generationPrompt: '<|turn>model\n',
    stop: ['<turn|>', '<end_of_turn>'],
  );

  /// Formats offered for models the user brings themselves.
  static const presets = [chatmlQwen3, gemma4];
}
