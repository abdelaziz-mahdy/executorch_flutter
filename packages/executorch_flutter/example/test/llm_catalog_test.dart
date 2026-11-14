import 'package:executorch_flutter_example/llm/chat_format.dart';
import 'package:executorch_flutter_example/llm/llm_catalog.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('ChatFormat', () {
    test('builds a Qwen3 prompt with history and thinking off', () {
      final prompt = ChatFormat.chatmlQwen3.build(const [
        ChatTurn(fromUser: true, text: 'Hi'),
        ChatTurn(fromUser: false, text: 'Hello!'),
        ChatTurn(fromUser: true, text: 'Capital of France?'),
      ]);
      expect(
        prompt,
        '<|im_start|>user\nHi<|im_end|>\n'
        '<|im_start|>assistant\nHello!<|im_end|>\n'
        '<|im_start|>user\nCapital of France?<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\n',
      );
    });

    test('skips the empty assistant turn being generated', () {
      final prompt = ChatFormat.gemma4.build(const [
        ChatTurn(fromUser: true, text: 'Hi'),
        ChatTurn(fromUser: false, text: ''),
      ]);
      expect(prompt, '<bos><|turn>user\nHi<turn|>\n<|turn>model\n');
    });

    test('strips stop markers and reports the end of the turn', () {
      expect(ChatFormat.chatmlQwen3.clean('Paris.<|im_end|>'), (
        text: 'Paris.',
        ended: true,
      ));
      expect(ChatFormat.chatmlQwen3.clean('Par'), (text: 'Par', ended: false));
    });
  });

  test('LlmEntry parses a catalog entry, with and without a metallib', () {
    Map<String, dynamic> file(String name) => {
      'name': name,
      'url': 'https://example.com/$name',
      'size': 10,
      'sha256': 'ab',
    };
    Map<String, dynamic> entry(String backend, {bool metallib = false}) => {
      'id': 'qwen3-$backend',
      'displayName': 'Qwen3 0.6B',
      'backend': backend,
      'license': 'Apache-2.0',
      'sourceUrl': 'https://huggingface.co/Qwen/Qwen3-0.6B',
      'chatFormat': {
        'id': 'chatml-qwen3',
        'user': '<|im_start|>user\n{text}<|im_end|>\n',
        'assistant': '<|im_start|>assistant\n{text}<|im_end|>\n',
        'generationPrompt': '<|im_start|>assistant\n',
        'stop': ['<|im_end|>'],
      },
      'files': {
        'model': file('m.pte'),
        'tokenizer': file('t.json'),
        if (metallib) 'metallib': file('mlx.metallib'),
      },
    };

    final cpu = LlmEntry.fromJson(entry('xnnpack'));
    expect(cpu.files.map((f) => f.name), ['m.pte', 't.json']);
    expect(cpu.totalSize, 20);
    expect(cpu.format.label, 'chatml-qwen3');

    final gpu = LlmEntry.fromJson(entry('mlx', metallib: true));
    expect(gpu.metallib?.name, 'mlx.metallib');
    expect(gpu.files, hasLength(3));
    expect(gpu.backendLabel, 'Apple GPU (MLX)');
  });
}
