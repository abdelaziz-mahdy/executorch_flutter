import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' hide TextInput;
import '../processors/base_processor.dart';
import '../processors/gemma_input_processor.dart';
import '../processors/gemma_output_processor.dart';
import '../renderers/screens/text_generation_renderer.dart';
import 'model_definition.dart';
import 'model_input.dart';
import 'model_settings.dart';
import 'gemma_model_settings.dart';
import 'text_generation_result.dart';

/// Gemma Text Generation Model Definition
class GemmaModelDefinition
    extends ModelDefinition<TextPromptInput, TextGenerationResult> {
  const GemmaModelDefinition({
    required super.name,
    required super.displayName,
    required super.description,
    required super.assetPath,
    required super.inputSize, // Sequence length (e.g., 128)
    required this.vocabAssetPath,
  }) : super(icon: Icons.auto_awesome);

  final String vocabAssetPath;

  // Cache for vocabulary (loaded once)
  static final Map<String, Map<String, int>> _vocabCache = {};

  Future<Map<String, int>> _loadVocabulary() async {
    if (_vocabCache.containsKey(vocabAssetPath)) {
      return _vocabCache[vocabAssetPath]!;
    }

    final vocabString = await rootBundle.loadString(vocabAssetPath);
    final vocabJson = json.decode(vocabString) as Map<String, dynamic>;

    // Convert to Map<String, int>
    final vocab = vocabJson.map((key, value) => MapEntry(key, value as int));

    _vocabCache[vocabAssetPath] = vocab;
    return vocab;
  }

  // Helper to load vocabulary synchronously from cache
  Map<String, int> _loadVocabularySync() {
    if (_vocabCache.containsKey(vocabAssetPath)) {
      return _vocabCache[vocabAssetPath]!;
    }
    // Vocabulary should be preloaded by controller before creating processor
    throw StateError('Vocabulary not loaded. Call loadVocabulary() first.');
  }

  // Make _loadVocabulary public so controller can preload
  Future<Map<String, int>> loadVocabulary() => _loadVocabulary();

  @override
  ModelSettings createDefaultSettings() {
    return GemmaModelSettings();
  }

  @override
  Widget buildInputWidget({
    required BuildContext context,
    required Function(TextPromptInput) onInputSelected,
    VoidCallback? onCameraModeToggle,
    bool isCameraMode = false,
  }) {
    return _TextInputWidget(
      onTextSubmitted: (text) => onInputSelected(TextPromptInput(text)),
    );
  }

  @override
  InputProcessor<TextPromptInput> createInputProcessor(ModelSettings settings) {
    // Load vocabulary from assets
    final vocabulary = _loadVocabularySync();

    // Get special token IDs from vocabulary
    final padTokenId = vocabulary['<pad>'] ?? vocabulary['[PAD]'] ?? 0;
    final bosTokenId =
        vocabulary['<bos>'] ?? vocabulary['[BOS]'] ?? vocabulary['<s>'] ?? 2;
    final eosTokenId =
        vocabulary['<eos>'] ?? vocabulary['[EOS]'] ?? vocabulary['</s>'] ?? 3;

    return GemmaInputProcessor(
      maxLength: inputSize,
      vocabulary: vocabulary,
      padTokenId: padTokenId,
      bosTokenId: bosTokenId,
      eosTokenId: eosTokenId,
    );
  }

  @override
  OutputProcessor<TextGenerationResult> createOutputProcessor(
    ModelSettings settings,
  ) {
    // Load vocabulary from assets
    final vocabulary = _loadVocabularySync();

    // Create reverse vocabulary
    final reverseVocab = VocabularyHelper.reverseVocabulary(vocabulary);

    // Get special token IDs
    final padTokenId = vocabulary['<pad>'] ?? vocabulary['[PAD]'] ?? 0;
    final bosTokenId =
        vocabulary['<bos>'] ?? vocabulary['[BOS]'] ?? vocabulary['<s>'] ?? 2;
    final eosTokenId =
        vocabulary['<eos>'] ?? vocabulary['[EOS]'] ?? vocabulary['</s>'] ?? 3;

    return GemmaOutputProcessor(
      reverseVocabulary: reverseVocab,
      eosTokenId: eosTokenId,
      bosTokenId: bosTokenId,
      padTokenId: padTokenId,
      inputPrompt: '', // Will be set during inference
    );
  }

  @override
  Widget buildResultRenderer({
    required BuildContext context,
    required TextPromptInput input,
    required TextGenerationResult? result,
  }) {
    return TextGenerationRenderer(input: input, result: result);
  }

  @override
  Widget buildResultsDetailsSection({
    required BuildContext context,
    required TextGenerationResult result,
    required double? processingTime,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Generation Details',
          style: Theme.of(
            context,
          ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 16),
        _buildDetailRow(context, 'Input Prompt', result.inputPrompt),
        const SizedBox(height: 8),
        _buildDetailRow(
          context,
          'Generated Text',
          result.generatedText.isEmpty ? '<no output>' : result.generatedText,
        ),
        const SizedBox(height: 8),
        _buildDetailRow(
          context,
          'Tokens Generated',
          result.tokensGenerated.toString(),
        ),
        if (result.timePerToken != null) ...[
          const SizedBox(height: 8),
          _buildDetailRow(
            context,
            'Time per Token',
            '${result.timePerToken!.toStringAsFixed(1)}ms',
          ),
        ],
        if (result.tokensPerSecond != null) ...[
          const SizedBox(height: 8),
          _buildDetailRow(
            context,
            'Tokens per Second',
            result.tokensPerSecond!.toStringAsFixed(1),
          ),
        ],
      ],
    );
  }

  Widget _buildDetailRow(BuildContext context, String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 140,
          child: Text(
            label,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: Theme.of(
              context,
            ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
          ),
        ),
      ],
    );
  }

  @override
  Widget buildSettingsWidget({
    required BuildContext context,
    required ModelSettings settings,
    required Function(ModelSettings) onSettingsChanged,
  }) {
    final gemmaSettings = settings is GemmaModelSettings
        ? settings
        : GemmaModelSettings();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Performance Overlay Section
        _buildSettingsSection(
          context: context,
          title: 'Display',
          children: [
            SwitchListTile(
              title: const Text('Show Performance Overlay'),
              subtitle: const Text('Display timing metrics'),
              value: gemmaSettings.showPerformanceOverlay,
              onChanged: (value) {
                gemmaSettings.showPerformanceOverlay = value;
                onSettingsChanged(gemmaSettings);
              },
            ),
          ],
        ),

        const SizedBox(height: 16),

        // Generation Settings Section
        _buildSettingsSection(
          context: context,
          title: 'Generation',
          children: [
            ListTile(
              title: const Text('Max Length'),
              subtitle: Text('${gemmaSettings.maxLength} tokens'),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: gemmaSettings.maxLength.toDouble(),
                min: 32,
                max: 512,
                divisions: 16,
                label: '${gemmaSettings.maxLength}',
                onChanged: (value) {
                  gemmaSettings.maxLength = value.toInt();
                  onSettingsChanged(gemmaSettings);
                },
              ),
            ),
            const Divider(),
            ListTile(
              title: const Text('Temperature'),
              subtitle: Text(gemmaSettings.temperature.toStringAsFixed(2)),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: gemmaSettings.temperature,
                min: 0.1,
                max: 2.0,
                divisions: 19,
                label: gemmaSettings.temperature.toStringAsFixed(2),
                onChanged: (value) {
                  gemmaSettings.temperature = value;
                  onSettingsChanged(gemmaSettings);
                },
              ),
            ),
            const Divider(),
            ListTile(
              title: const Text('Top-p (Nucleus Sampling)'),
              subtitle: Text(gemmaSettings.topP.toStringAsFixed(2)),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: gemmaSettings.topP,
                min: 0.1,
                max: 1.0,
                divisions: 9,
                label: gemmaSettings.topP.toStringAsFixed(2),
                onChanged: (value) {
                  gemmaSettings.topP = value;
                  onSettingsChanged(gemmaSettings);
                },
              ),
            ),
            const Divider(),
            ListTile(
              title: const Text('Top-k Sampling'),
              subtitle: Text('${gemmaSettings.topK}'),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Slider(
                value: gemmaSettings.topK.toDouble(),
                min: 1,
                max: 100,
                divisions: 99,
                label: '${gemmaSettings.topK}',
                onChanged: (value) {
                  gemmaSettings.topK = value.toInt();
                  onSettingsChanged(gemmaSettings);
                },
              ),
            ),
          ],
        ),

        const SizedBox(height: 16),

        // Reset button
        Center(
          child: OutlinedButton.icon(
            onPressed: () {
              gemmaSettings.reset();
              onSettingsChanged(gemmaSettings);
            },
            icon: const Icon(Icons.refresh),
            label: const Text('Reset to Defaults'),
          ),
        ),
      ],
    );
  }

  Widget _buildSettingsSection({
    required BuildContext context,
    required String title,
    required List<Widget> children,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Text(
            title,
            style: Theme.of(context).textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.bold,
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
        ),
        Container(
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(children: children),
        ),
      ],
    );
  }

  @override
  String getExportCommand() {
    return 'python3 main.py export --gemma';
  }

  @override
  String? getSpecialSetupRequirements() {
    return 'Requires optimum-executorch and HuggingFace authentication.\n'
        'Run: ./install_executorch.sh\n'
        'Then: hf auth login';
  }
}

/// Text input widget for text generation models
class _TextInputWidget extends StatefulWidget {
  const _TextInputWidget({required this.onTextSubmitted});

  final Function(String) onTextSubmitted;

  @override
  State<_TextInputWidget> createState() => _TextInputWidgetState();
}

class _TextInputWidgetState extends State<_TextInputWidget> {
  final _textController = TextEditingController();
  final _focusNode = FocusNode();

  @override
  void dispose() {
    _textController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _submit() {
    final text = _textController.text.trim();
    if (text.isNotEmpty) {
      widget.onTextSubmitted(text);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Enter your prompt',
            style: Theme.of(
              context,
            ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _textController,
            focusNode: _focusNode,
            maxLines: 5,
            decoration: InputDecoration(
              hintText: 'Type your prompt here...',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              filled: true,
            ),
            onSubmitted: (_) => _submit(),
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: _submit,
              icon: const Icon(Icons.auto_awesome),
              label: const Text('Generate'),
            ),
          ),
        ],
      ),
    );
  }
}
