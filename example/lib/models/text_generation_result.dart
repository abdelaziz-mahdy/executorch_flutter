/// Result from text generation inference
class TextGenerationResult {
  /// The generated text output
  final String generatedText;

  /// The original input prompt
  final String inputPrompt;

  /// Number of tokens generated
  final int tokensGenerated;

  /// Generation time per token (milliseconds)
  final double? timePerToken;

  /// Raw logits (optional, for debugging)
  final List<double>? logits;

  const TextGenerationResult({
    required this.generatedText,
    required this.inputPrompt,
    required this.tokensGenerated,
    this.timePerToken,
    this.logits,
  });

  /// Total generation time
  double? get totalTime =>
      timePerToken != null ? timePerToken! * tokensGenerated : null;

  /// Tokens per second
  double? get tokensPerSecond =>
      timePerToken != null ? 1000.0 / timePerToken! : null;

  @override
  String toString() {
    return 'TextGenerationResult(\n'
        '  input: "$inputPrompt"\n'
        '  output: "$generatedText"\n'
        '  tokens: $tokensGenerated\n'
        '  time/token: ${timePerToken?.toStringAsFixed(2)}ms\n'
        '  tokens/sec: ${tokensPerSecond?.toStringAsFixed(1)}\n'
        ')';
  }

  /// Create a copy with updated values
  TextGenerationResult copyWith({
    String? generatedText,
    String? inputPrompt,
    int? tokensGenerated,
    double? timePerToken,
    List<double>? logits,
  }) {
    return TextGenerationResult(
      generatedText: generatedText ?? this.generatedText,
      inputPrompt: inputPrompt ?? this.inputPrompt,
      tokensGenerated: tokensGenerated ?? this.tokensGenerated,
      timePerToken: timePerToken ?? this.timePerToken,
      logits: logits ?? this.logits,
    );
  }
}
