import 'package:executorch_dart/src/llm_types.dart';
import 'package:test/test.dart';

void main() {
  test('accepts the defaults and the edges of the ranges', () {
    expect(validateGenConfig(const GenConfig()), isNull);
    expect(validateGenConfig(const GenConfig(temperature: 0)), isNull);
    expect(validateGenConfig(const GenConfig(temperature: 1)), isNull);
    expect(validateGenConfig(const GenConfig(maxNewTokens: 1)), isNull);
  });

  test('rejects temperatures outside [0, 1], including NaN', () {
    expect(validateGenConfig(const GenConfig(temperature: -0.1)), isNotNull);
    expect(validateGenConfig(const GenConfig(temperature: 1.5)), isNotNull);
    expect(
      validateGenConfig(const GenConfig(temperature: double.nan)),
      isNotNull,
    );
  });

  test('rejects maxNewTokens of 0 or below -1', () {
    expect(validateGenConfig(const GenConfig(maxNewTokens: 0)), isNotNull);
    expect(validateGenConfig(const GenConfig(maxNewTokens: -2)), isNotNull);
  });

  test('rejects counts that do not fit the native types', () {
    expect(
      validateGenConfig(const GenConfig(maxNewTokens: 0x80000000)),
      isNotNull,
    );
    expect(validateGenConfig(const GenConfig(seqLen: 0)), isNotNull);
    expect(validateGenConfig(const GenConfig(seqLen: -2)), isNotNull);
    expect(validateGenConfig(const GenConfig(seqLen: 512)), isNull);
    expect(validateGenConfig(const GenConfig(numBos: -1)), isNotNull);
    expect(validateGenConfig(const GenConfig(numEos: 128)), isNotNull);
    expect(validateGenConfig(const GenConfig(numBos: 1)), isNull);
  });
}
