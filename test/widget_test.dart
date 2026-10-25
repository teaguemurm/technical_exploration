import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:technical_exploration/main.dart';

void main() {
  testWidgets('report cards keep every control reachable', (tester) async {
    tester.view.physicalSize = const Size(1200, 2400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);

    await tester.pumpWidget(const TechnicalExplorationApp());
    await tester.pumpAndSettle();

    expect(find.text('Setup'), findsOneWidget);
    expect(find.text('Telemetry'), findsOneWidget);
    expect(find.text('Cold load'), findsOneWidget);
    expect(find.text('Messages & output'), findsOneWidget);
    expect(find.text('LLM configuration'), findsOneWidget);
    expect(find.text('History'), findsOneWidget);
    expect(find.text('Diagnostics'), findsOneWidget);

    expect(find.text('Pick GGUF'), findsOneWidget);
    expect(find.text('Load'), findsOneWidget);
    expect(find.text('No file selected'), findsOneWidget);
    expect(find.text('None'), findsOneWidget);
    expect(find.text('Pick a model first'), findsOneWidget);

    expect(find.text('A · Base'), findsOneWidget);
    expect(find.text('B · Dataset'), findsOneWidget);
    expect(find.text('C · Constraints'), findsOneWidget);
    expect(find.text('Simulated RAG'), findsOneWidget);
    expect(find.text('Visitor question'), findsOneWidget);

    expect(find.text('Run'), findsOneWidget);
    expect(find.text('Run batch'), findsOneWidget);
    expect(find.text('More'), findsOneWidget);
    expect(find.text('Locked'), findsOneWidget);
    expect(find.text('Temperature'), findsOneWidget);

    await tester.tap(find.text('More'));
    await tester.pumpAndSettle();
    expect(find.text('Compare all 3'), findsOneWidget);
    expect(find.text('10-run heat test'), findsOneWidget);
    await tester.tapAt(const Offset(10, 10));
    await tester.pumpAndSettle();

    await tester.scrollUntilVisible(
      find.text('Diagnostics'),
      400,
      scrollable: find.byType(Scrollable).first,
    );
    await tester.tap(find.text('Diagnostics'));
    await tester.pumpAndSettle();
    expect(find.text('Verify model'), findsOneWidget);
    expect(find.text('Skip sanity check'), findsOneWidget);
    expect(find.text('Copy CSV'), findsOneWidget);
    expect(find.text('Copy JSON'), findsOneWidget);
    expect(find.text('Export files'), findsOneWidget);
    expect(find.text('Clear history'), findsOneWidget);
    expect(
      find.text('Long-press a card title to save a PNG.'),
      findsOneWidget,
    );
    expect(tester.takeException(), isNull);
  });

  testWidgets('phone, fold, and desktop widths do not overflow', (tester) async {
    addTearDown(tester.view.resetPhysicalSize);
    for (final size in const [
      Size(390, 844),
      Size(720, 900),
      Size(1280, 800),
    ]) {
      tester.view.physicalSize = size;
      tester.view.devicePixelRatio = 1.0;
      await tester.pumpWidget(const TechnicalExplorationApp());
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull, reason: '$size');
    }
  });
}
