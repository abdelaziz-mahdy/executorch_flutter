import 'package:executorch_flutter_example/services/model_download_service.dart';
import 'package:flutter_test/flutter_test.dart';

/// Run the override cases with:
///   flutter test --dart-define=MODEL_BASE_URL=models
void main() {
  const release =
      'https://github.com/abdelaziz-mahdy/executorch_flutter_models'
      '/releases/download/v1.3.1/mobilenet_v3_small_xnnpack.pte';
  const labels =
      'https://github.com/abdelaziz-mahdy/executorch_flutter_models'
      '/releases/download/v1.3.1/mobilenet-labels.txt';

  if (modelBaseUrlOverride.isEmpty) {
    test('without MODEL_BASE_URL, URLs pass through untouched', () {
      expect(resolveModelUrl(release), release);
      expect(resolveModelUrl(labels), labels);
    });
    return;
  }

  test('models resolve to the injected base', () {
    expect(
      resolveModelUrl(release),
      '$modelBaseUrlOverride/mobilenet_v3_small_xnnpack.pte',
    );
  });

  test('label files resolve to the injected base', () {
    // Labels come from the same release as the models and hit the same CORS
    // wall, so they must be redirected too.
    expect(
      resolveModelUrl(labels),
      '$modelBaseUrlOverride/mobilenet-labels.txt',
    );
  });

  test('an existing query string is dropped', () {
    expect(
      resolveModelUrl('$release?t=123'),
      '$modelBaseUrlOverride/mobilenet_v3_small_xnnpack.pte',
    );
  });
}
