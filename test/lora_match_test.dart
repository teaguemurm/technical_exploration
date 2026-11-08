import 'package:flutter_test/flutter_test.dart';
import 'package:technical_exploration/lora_match.dart';

void main() {
  test('matching architectures are accepted', () {
    expect(
      adapterArchitectureRefusal(
        modelArchitecture: 'qwen2',
        adapterArchitecture: 'qwen2',
      ),
      isNull,
    );
    expect(
      adapterArchitectureRefusal(
        modelArchitecture: 'gemma4',
        adapterArchitecture: 'gemma4',
      ),
      isNull,
    );
  });

  test('a mismatched adapter is refused before load', () {
    expect(
      adapterArchitectureRefusal(
        modelArchitecture: 'qwen2',
        adapterArchitecture: 'gemma4',
      ),
      'This adapter is for gemma4, but the model is qwen2. '
      'Pick an adapter that matches the model.',
    );
  });

  test('a missing header is refused instead of guessing', () {
    expect(
      adapterArchitectureRefusal(
        modelArchitecture: null,
        adapterArchitecture: 'qwen2',
      ),
      contains('Could not read the model architecture'),
    );
    expect(
      adapterArchitectureRefusal(
        modelArchitecture: 'gemma4',
        adapterArchitecture: null,
      ),
      contains("Could not read this adapter's architecture"),
    );
  });
}
