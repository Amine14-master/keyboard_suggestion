// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:keayboard_predictor/main.dart';

void main() {
  testWidgets('Keyboard predictor app smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const KeyboardPredictorApp());

    // Verify that the text field is present
    expect(find.byType(TextField), findsOneWidget);

    // Verify that the language selector is present
    expect(find.byType(DropdownButton), findsOneWidget);

    // Verify app title
    expect(find.text('Next-word Suggestions (1-5 grams)'), findsOneWidget);
  });
}
