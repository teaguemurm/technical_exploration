import 'dart:ui';
import 'package:flutter_test/flutter_test.dart';
import 'package:technical_exploration/main.dart';

void main() {
  testWidgets('the screen shows telemetry, model controls, conditions, and benchmark buttons', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1200, 2400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);

    await tester.pumpWidget(const TechnicalExplorationApp());
    await tester.pumpAndSettle();

    // Telemetry dashboard
    expect(find.text('Hardware & Inference Telemetry'), findsOneWidget);
    expect(find.text('Cold Load'), findsOneWidget);

    // Model controls
    expect(find.text('Pick GGUF'), findsOneWidget);
    expect(find.text('Load'), findsOneWidget);
    expect(find.text('No file selected'), findsOneWidget);

    // Three Condition Segments
    expect(find.text('A: Base'), findsOneWidget);
    expect(find.text('B: Dataset'), findsOneWidget);
    expect(find.text('C: Constraints'), findsOneWidget);

    // Run actions
    expect(find.text('Run'), findsOneWidget);
    expect(find.text('Compare All 3 (A vs B vs C)'), findsOneWidget);
    expect(find.text('3-Prompt Throttle Test'), findsOneWidget);
    expect(find.text('Stop'), findsOneWidget);
  });
}
