import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/system_metrics.dart';

void main() {
  test('system prompt drops one trailing newline only', () {
    expect(trimSingleTrailingNewline('abc\n'), 'abc');
    expect(trimSingleTrailingNewline('abc\r\n'), 'abc');
    expect(trimSingleTrailingNewline('abc\n\n'), 'abc\n');
    expect(trimSingleTrailingNewline('  abc  '), '  abc  ');
  });

  test('seeded assets parse all 10 prompts and 10 facts correctly', () {
    final catalog = parseCatalog(
      systemPrompt: File('assets/system_prompt.txt').readAsStringSync(),
      promptsJson: File('assets/prompts.json').readAsStringSync(),
      datasetJson: File('assets/dataset.json').readAsStringSync(),
    );

    expect(catalog.prompts.length, 10);
    expect(catalog.facts.length, 10);
    expect(catalog.prompts.first.id, 'what-animal');
    expect(catalog.prompts.first.relevantFactIds, ['wisent-herd']);
    expect(catalog.facts.first.id, 'wisent-herd');
  });

  test('Condition A sends system prompt and user prompt only', () {
    const prompt = PromptSpec(
      id: 'what-animal',
      label: 'What animal is this?',
      text: 'What animal is this?',
      relevantFactIds: ['wisent-herd'],
    );
    const facts = [
      DatasetFact(id: 'wisent-herd', text: 'Wisent herd fact.'),
      DatasetFact(id: 'other', text: 'Other fact.'),
    ];

    final messages = buildConditionMessages(
      condition: PromptCondition.conditionA,
      systemPrompt: 'System note.',
      prompt: prompt,
      allFacts: facts,
    );

    expect(messages.length, 2);
    expect(messages[0].role, LLMRole.system);
    expect(messages[0].content, 'System note.');
    expect(messages[1].role, LLMRole.user);
    expect(messages[1].content, 'What animal is this?');
  });

  test('Condition B with simulated RAG injects only relevant facts', () {
    const prompt = PromptSpec(
      id: 'what-animal',
      label: 'What animal is this?',
      text: 'What animal is this?',
      relevantFactIds: ['wisent-herd'],
    );
    const facts = [
      DatasetFact(id: 'wisent-herd', text: 'Wisent herd fact.'),
      DatasetFact(id: 'other', text: 'Other fact.'),
    ];

    final messages = buildConditionMessages(
      condition: PromptCondition.conditionB,
      systemPrompt: 'System note.',
      prompt: prompt,
      allFacts: facts,
      relevantOnly: true,
    );

    expect(messages.length, 3);
    expect(messages[0].role, LLMRole.system);
    expect(messages[1].role, LLMRole.user);
    expect(messages[1].content, contains('Wisent herd fact.'));
    expect(messages[1].content, isNot(contains('Other fact.')));
    expect(messages[2].content, 'What animal is this?');
  });

  test('Condition B with all notes injects all facts', () {
    const prompt = PromptSpec(
      id: 'what-animal',
      label: 'What animal is this?',
      text: 'What animal is this?',
      relevantFactIds: ['wisent-herd'],
    );
    const facts = [
      DatasetFact(id: 'wisent-herd', text: 'Wisent herd fact.'),
      DatasetFact(id: 'other', text: 'Other fact.'),
    ];

    final messages = buildConditionMessages(
      condition: PromptCondition.conditionB,
      systemPrompt: 'System note.',
      prompt: prompt,
      allFacts: facts,
      relevantOnly: false,
    );

    expect(messages.length, 3);
    expect(messages[1].content, contains('Wisent herd fact.'));
    expect(messages[1].content, contains('Other fact.'));
  });

  test('Condition C includes strict grounding constraints and refusal rule', () {
    const prompt = PromptSpec(
      id: 'calves',
      label: 'How many calves were born?',
      text: 'How many calves were born?',
      relevantFactIds: [],
    );
    const facts = [
      DatasetFact(id: 'wisent-herd', text: 'Wisent herd fact.'),
    ];

    final messages = buildConditionMessages(
      condition: PromptCondition.conditionC,
      systemPrompt: 'Base prompt.',
      prompt: prompt,
      allFacts: facts,
      relevantOnly: true,
    );

    expect(messages[0].role, LLMRole.system);
    expect(messages[0].content, contains('STRICT GROUNDING & FORM CONSTRAINTS'));
    expect(messages[0].content, contains(standardRefusalStatement));
    expect(messages[1].content, contains('(No specific local notes match this query.)'));
  });

  test('isRefusalOutput detects standard refusal phrasings', () {
    expect(isRefusalOutput('I do not have information about that in the approved notes.'), isTrue);
    expect(isRefusalOutput('There is no information regarding this.'), isTrue);
    expect(isRefusalOutput('The wisent herd was introduced in 2007.'), isFalse);
  });

  test('formatBytes formats GB, MB, KB correctly', () {
    expect(formatBytes(1024 * 1024 * 1024), '1.00 GB');
    expect(formatBytes(500 * 1024 * 1024), '500.00 MB');
    expect(formatBytes(2048), '2.00 KB');
    expect(formatBytes(1024 * 1024 * 1024, signed: true), '+1.00 GB');
    expect(formatBytes(null), '--');
  });

  test('tokens per second uses library count and measured interval', () {
    expect(
      tokensPerSecond(
        evalCount: 10,
        generation: const Duration(milliseconds: 500),
      ),
      20.0,
    );
    expect(
      tokensPerSecond(evalCount: null, generation: const Duration(seconds: 1)),
      isNull,
    );
  });

  test('BenchmarkExporter exports Markdown table with runs', () {
    final report = BenchmarkExporter.exportMarkdownReport(
      modelFileName: 'qwen2.5.gguf',
      coldLoadMs: 1200,
      baselineRssBytes: 100 * 1024 * 1024,
      idleModelRssBytes: 1500 * 1024 * 1024,
      runs: [
        RunRecord(
          promptId: 'what-animal',
          promptText: 'What animal is this?',
          conditionLabel: 'Condition B',
          relevantOnly: true,
          maxTokens: 256,
          output: 'This is a wisent.',
          timestamp: DateTime.now(),
          ttftMs: 350,
          tokensPerSecond: 18.5,
          evalCount: 30,
        ),
      ],
    );

    expect(report, contains('Technical Exploration Benchmark Report'));
    expect(report, contains('qwen2.5.gguf'));
    expect(report, contains('Condition B'));
    expect(report, contains('18.50'));
  });
}
