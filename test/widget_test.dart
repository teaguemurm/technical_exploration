import 'package:flutter_test/flutter_test.dart';
import 'package:technical_exploration/main.dart';

void main() {
  testWidgets('the screen shows pick, load, dataset, run, and stop', (
    tester,
  ) async {
    await tester.pumpWidget(const TechnicalExplorationApp());
    await tester.pumpAndSettle();

    expect(find.text('Pick GGUF'), findsOneWidget);
    expect(find.text('Load'), findsOneWidget);
    expect(find.text('Use dataset'), findsOneWidget);
    expect(find.text('Run'), findsOneWidget);
    expect(find.text('Stop'), findsOneWidget);
    expect(find.text('Choose prompt'), findsOneWidget);
    expect(find.text('No file selected'), findsOneWidget);
  });
}
