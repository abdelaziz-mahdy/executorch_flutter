// Smoke test for the example app's root widget and category landing screen.
//
// The default `flutter create` template ships a counter-app test here; this
// app has no counter (it's an ExecuTorch model playground), so that template
// never matched the real app and didn't compile once ExecuTorchPlaygroundApp
// replaced MyApp. This test exercises the real root widget instead: it
// verifies the app builds and the home screen's two demo categories (vision
// models, language models — see lib/screens/home_screen.dart) are present.

import 'package:flutter_test/flutter_test.dart';

import 'package:executorch_flutter_example/main.dart';

void main() {
  testWidgets('App builds and shows both model categories', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const ExecuTorchPlaygroundApp());

    expect(find.text('Vision Models'), findsOneWidget);
    expect(find.text('Language Models'), findsOneWidget);
  });
}
