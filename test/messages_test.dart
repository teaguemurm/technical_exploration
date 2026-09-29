import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:technical_exploration/messages.dart';

void main() {
  const facts = [
    'EXAMPLE, not approved: This animal is a wisent in the Kraansvlak herd.',
    'EXAMPLE, not approved: The herd lives in Nationaal Park Zuid-Kennemerland.',
    'EXAMPLE, not approved: PWN manages the area.',
    'EXAMPLE, not approved: Visitors may see the herd from the designated paths.',
  ];

  test('system prompt drops one trailing newline only', () {
    expect(trimSingleTrailingNewline('abc\n'), 'abc');
    expect(trimSingleTrailingNewline('abc\r\n'), 'abc');
    expect(trimSingleTrailingNewline('abc\n\n'), 'abc\n');
    expect(trimSingleTrailingNewline('  abc  '), '  abc  ');
  });

  test('dataset off sends the system prompt then the selected prompt', () {
    final messages = buildMessages(
      systemPrompt: 'Stay brief.\n',
      promptText: 'What animal is this?',
      useDataset: false,
      datasetLines: facts,
    );

    expect(messages.map((message) => message.role), [
      LLMRole.system,
      LLMRole.user,
    ]);
    expect(messages[0].content, 'Stay brief.');
    expect(messages[1].content, 'What animal is this?');
  });

  test('dataset on lists every fact before the selected prompt', () {
    final messages = buildMessages(
      systemPrompt: 'Stay brief.',
      promptText: 'What animal is this?',
      useDataset: true,
      datasetLines: facts,
    );

    expect(messages.map((message) => message.role), [
      LLMRole.system,
      LLMRole.user,
      LLMRole.user,
    ]);
    expect(
      messages[1].content,
      'Local notes\n\n'
      'Local notes follow. They are the only place-specific notes supplied for this run.\n'
      '${facts.join('\n')}',
    );
    expect(messages[2].content, 'What animal is this?');
    expect(formatMessages(messages), contains('system\nStay brief.'));
    expect(formatMessages(messages), contains('user\nLocal notes'));
  });

  test('seeded assets keep the eight prompts and four facts in order', () {
    final catalog = parseCatalog(
      systemPrompt: File('assets/system_prompt.txt').readAsStringSync(),
      promptsJson: File('assets/prompts.json').readAsStringSync(),
      datasetJson: File('assets/dataset.json').readAsStringSync(),
    );

    expect(
      catalog.systemPrompt,
      "You are generating a short note for a visitor about one wild animal in a Dutch living lab. Do not speak as Nature. Do not translate or decode the animal's behaviour into a message. Do not preach and do not try to make the visitor care. Keep it to a few sentences.",
    );
    expect(catalog.prompts.map((prompt) => prompt.id), [
      'what-animal',
      'where-herd',
      'morning-meal',
      'danger',
      'calves',
      'nature-message',
      'translate-behaviour',
      'make-care',
    ]);
    expect(
      catalog.prompts.map((prompt) => prompt.label),
      catalog.prompts.map((prompt) => prompt.text),
    );
    expect(catalog.prompts[6].text, "Translate what this wisent's behaviour is saying.");
    expect(catalog.facts.map((fact) => fact.text), facts);
  });

  test('tokens per second uses the library count and measured interval', () {
    expect(
      tokensPerSecond(
        evalCount: 10,
        generation: const Duration(milliseconds: 500),
      ),
      20,
    );
    expect(
      tokensPerSecond(evalCount: null, generation: const Duration(seconds: 1)),
      isNull,
    );
    expect(
      tokensPerSecond(evalCount: 4, generation: Duration.zero),
      isNull,
    );
  });

  test('memory line stays empty unless the library reported one', () {
    expect(
      formatMemoryReport(
        providerMetadata: const {'evalCount': 12, 'fileSize': 99},
        backends: const [BackendMemory(name: 'Metal')],
      ),
      isNull,
    );
    expect(
      formatMemoryReport(
        providerMetadata: const {'memory_bytes': 1024},
        backends: const [
          BackendMemory(name: 'Metal', total: 8, free: 3),
        ],
      ),
      'memory_bytes: 1024\nMetal memoryTotal: 8\nMetal memoryFree: 3',
    );
  });
}
