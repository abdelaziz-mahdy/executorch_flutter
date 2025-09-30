// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:executorch_flutter_example/main.dart';

void main() {
  testWidgets('ExecuTorch Playground App loads', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const ExecuTorchPlaygroundApp());

    // Verify that the app title is displayed.
    expect(find.text('ExecuTorch Playground'), findsOneWidget);

    // Verify that model selection screen is shown.
    expect(find.text('Choose a Model'), findsOneWidget);
  });
}
