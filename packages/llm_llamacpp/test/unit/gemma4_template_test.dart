import 'package:llm_llamacpp/src/chat_template_format.dart';
import 'package:llm_llamacpp/src/stop_token_resolver.dart';
import 'package:test/test.dart';

void main() {
  const system = 'You are a guide.';
  const notes = 'Note one.';
  const question = 'What animal?';
  const turns = [
    ChatTurn(role: 'system', content: system),
    ChatTurn(role: 'user', content: notes),
    ChatTurn(role: 'user', content: question),
  ];
  const gemmaTemplate = '<|turn>role<turn|> {{ bos_token }}';

  test('gemma 4 string keeps a system turn and ends on the model newline', () {
    final text = formatGemma4(turns);

    expect(
      text,
      '<bos><|turn>system\n'
      'You are a guide.<turn|>\n'
      '<|turn>user\n'
      'Note one.<turn|>\n'
      '<|turn>user\n'
      'What animal?<turn|>\n'
      '<|turn>model\n',
    );
    expect(text.endsWith('<|turn>model\n'), isTrue);
    expect(text.endsWith('<turn|>'), isFalse);
    expect(text.endsWith('<turn|>\n'), isFalse);
    expect(text.contains('<|think|>'), isFalse);
    expect(text.contains('<|channel>'), isFalse);
    expect(text.toLowerCase().contains('thought'), isFalse);
    expect(text.contains('You are a guide.\nNote one.'), isFalse);
  });

  test('assistant turns are written as model', () {
    final text = formatGemma4(const [
      ChatTurn(role: 'user', content: 'Hi'),
      ChatTurn(role: 'assistant', content: 'Hello'),
    ]);

    expect(text.contains('<|turn>model\nHello<turn|>\n'), isTrue);
    expect(text.endsWith('<|turn>model\n'), isTrue);
    expect(text.contains('assistant'), isFalse);
  });

  test('a tool turn is rejected', () {
    expect(
      () => formatGemma4(const [ChatTurn(role: 'tool', content: '{}')]),
      throwsA(isA<ChatTemplateException>()),
    );
  });

  test('gemma template is detected from turn markers, not the architecture', () {
    expect(isGemma4Template(gemmaTemplate), isTrue);
    expect(isGemma4Template('<start_of_turn>user'), isFalse);
    expect(isGemma4Template('<|im_start|>user'), isFalse);
    expect(isGemma4Template(null), isFalse);
  });

  test('an unusable template does not become ChatML', () {
    const turns = [ChatTurn(role: 'user', content: 'Hi')];
    expect(
      () => applyChatTemplateDecision(
        template: null,
        messages: turns,
        nativeText: null,
      ),
      throwsA(isA<ChatTemplateException>()),
    );

    ChatTemplateException? error;
    try {
      applyChatTemplateDecision(
        template: '{% for message in messages %}{{ message.content }}{% endfor %}',
        messages: turns,
        nativeText: null,
      );
    } on ChatTemplateException catch (caught) {
      error = caught;
    }
    expect(error, isNotNull);
    expect(error!.message.contains('<|im_start|>'), isFalse);
    expect(error.message.contains('chatml'), isFalse);
  });

  test('a failed ChatML template is labeled chatml-fallback', () {
    const turns = [ChatTurn(role: 'user', content: 'Hi')];
    final applied = applyChatTemplateDecision(
      template: '<|im_start|>{{ role }}',
      messages: turns,
      nativeText: null,
    );

    expect(applied.formatter, promptFormatterChatMlFallback);
    expect(applied.text, formatChatMlFallback(turns));
    expect(applied.bosIncludedInText, isFalse);
  });

  test('a successful native render is returned untouched', () {
    const native = '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n';
    final applied = applyChatTemplateDecision(
      template: '<|im_start|>{{ role }}\n{{ content }}<|im_end|>',
      messages: const [ChatTurn(role: 'user', content: 'Hi')],
      nativeText: native,
    );

    expect(applied.formatter, promptFormatterNative);
    expect(identical(applied.text, native), isTrue);
  });

  test('a gemma template ignores a native ChatML string', () {
    final applied = applyChatTemplateDecision(
      template: gemmaTemplate,
      messages: turns,
      nativeText: '<|im_start|>assistant\n',
    );

    expect(applied.formatter, promptFormatterGemma4);
    expect(applied.text.contains('<|im_start|>'), isFalse);
    expect(applied.text, formatGemma4(turns));
    expect(applied.bosIncludedInText, isTrue);
  });

  test('mismatch error names both formatters', () {
    final message = promptFormatterMismatchMessage(
      preflight: promptFormatterGemma4,
      sent: promptFormatterChatMlFallback,
    );

    expect(
      message,
      'Prompt formatter mismatch: preflight=gemma4, sent=chatml-fallback',
    );
    expect(message.contains(promptFormatterGemma4), isTrue);
    expect(message.contains(promptFormatterChatMlFallback), isTrue);
  });

  test('gemma stops are not added to a ChatML prompt', () {
    const chatMl =
        '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n';
    expect(resolveStopTokens(requested: const [], prompt: chatMl), [
      '<|im_end|>',
    ]);

    const gemma = '<bos><|turn>user\nHi<turn|>\n<|turn>model\n';
    expect(resolveStopTokens(requested: const [], prompt: gemma), [
      '<turn|>',
      '<end_of_turn>',
      '<eos>',
    ]);
    expect(
      resolveStopTokens(requested: const ['<turn|>'], prompt: gemma),
      ['<turn|>', '<end_of_turn>', '<eos>'],
    );
  });
}
