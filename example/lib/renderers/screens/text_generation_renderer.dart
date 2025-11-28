import 'package:flutter/material.dart';
import '../../models/model_input.dart';
import '../../models/text_generation_result.dart';

/// Renderer for text generation results
class TextGenerationRenderer extends StatefulWidget {
  const TextGenerationRenderer({
    super.key,
    required this.input,
    required this.result,
  });

  final ModelInput input;
  final TextGenerationResult? result;

  @override
  State<TextGenerationRenderer> createState() => _TextGenerationRendererState();
}

class _TextGenerationRendererState extends State<TextGenerationRenderer> {
  @override
  Widget build(BuildContext context) {
    final textInput = widget.input as TextPromptInput;

    return Container(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Input prompt section
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.blue.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: Colors.blue.withValues(alpha: 0.3),
                width: 1,
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.edit, size: 16, color: Colors.blue.shade700),
                    const SizedBox(width: 8),
                    Text(
                      'Prompt',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue.shade700,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                SelectableText(
                  textInput.text,
                  style: const TextStyle(fontSize: 14, height: 1.5),
                ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // Generated text section
          if (widget.result != null) ...[
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.green.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: Colors.green.withValues(alpha: 0.3),
                  width: 1,
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.auto_awesome,
                        size: 16,
                        color: Colors.green.shade700,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Generated Text',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: Colors.green.shade700,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  SelectableText(
                    widget.result!.generatedText.isEmpty
                        ? '<no output>'
                        : widget.result!.generatedText,
                    style: TextStyle(
                      fontSize: 14,
                      height: 1.5,
                      fontStyle: widget.result!.generatedText.isEmpty
                          ? FontStyle.italic
                          : FontStyle.normal,
                      color: widget.result!.generatedText.isEmpty
                          ? Colors.grey
                          : Colors.black,
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // Statistics
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: Colors.grey.withValues(alpha: 0.3),
                  width: 1,
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.analytics, size: 16, color: Colors.grey),
                      const SizedBox(width: 8),
                      Text(
                        'Statistics',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey.shade700,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _buildStatRow(
                    'Tokens Generated',
                    widget.result!.tokensGenerated.toString(),
                  ),
                  if (widget.result!.timePerToken != null) ...[
                    const SizedBox(height: 8),
                    _buildStatRow(
                      'Time per Token',
                      '${widget.result!.timePerToken!.toStringAsFixed(1)}ms',
                    ),
                  ],
                  if (widget.result!.tokensPerSecond != null) ...[
                    const SizedBox(height: 8),
                    _buildStatRow(
                      'Tokens per Second',
                      widget.result!.tokensPerSecond!.toStringAsFixed(1),
                    ),
                  ],
                ],
              ),
            ),
          ],

          // Loading indicator
          if (widget.result == null) ...[
            const Center(
              child: Padding(
                padding: EdgeInsets.all(32),
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text(
                      'Generating text...',
                      style: TextStyle(fontSize: 14, color: Colors.grey),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(fontSize: 13, color: Colors.grey)),
        Text(
          value,
          style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
        ),
      ],
    );
  }
}
