import 'dart:convert';
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

  test('bundled JSON matches the locked prompts and notes (10 prompts, 10 notes)', () {
    final promptsFile = File('assets/prompts.json').readAsStringSync();
    final datasetFile = File('assets/dataset.json').readAsStringSync();
    final simRagFile = File('assets/simulated_rag.json').readAsStringSync();
    final systemPrompt = File('assets/system_prompt.txt').readAsStringSync();

    final catalog = parseCatalog(
      systemPrompt: systemPrompt,
      promptsJson: promptsFile,
      datasetJson: datasetFile,
      simRagJson: simRagFile,
    );

    expect(catalog.prompts.length, 10);
    expect(catalog.facts.length, 10);

    const expectedPromptIds = [
      'what-animal',
      'who-manages',
      'where-stand',
      'on-path',
      'since-when',
      'morning-meal',
      'calves',
      'nature-message',
      'speak-as',
      'make-care',
    ];
    expect(catalog.prompts.map((p) => p.id).toList(), expectedPromptIds);

    const expectedFactIds = [
      'wisent-herd',
      'other-grazers',
      'park',
      'pwn',
      'arrival',
      'distance',
      'paths',
      'wisentenpad',
      'on-path',
      'no-feeding',
    ];
    expect(catalog.facts.map((f) => f.id).toList(), expectedFactIds);

    expect(catalog.simRagConfig.k, 2);
    for (final id in expectedPromptIds) {
      expect(catalog.simRagConfig.retrieval.containsKey(id), isTrue,
          reason: 'simulated_rag.json must contain prompt $id');
      expect(catalog.simRagConfig.retrieval[id]!.length, 2,
          reason: 'prompt $id must have exactly k=2 notes');
    }

    // Near-miss checks
    expect(catalog.simRagConfig.retrieval['morning-meal'], contains('no-feeding'));
    expect(catalog.simRagConfig.retrieval['what-animal'], contains('other-grazers'));
  });

  test('Condition B and C contain all 10 notes, while sim-RAG variants contain exactly their fixed lists', () {
    final promptsFile = File('assets/prompts.json').readAsStringSync();
    final datasetFile = File('assets/dataset.json').readAsStringSync();
    final simRagFile = File('assets/simulated_rag.json').readAsStringSync();
    final systemPrompt = File('assets/system_prompt.txt').readAsStringSync();

    final catalog = parseCatalog(
      systemPrompt: systemPrompt,
      promptsJson: promptsFile,
      datasetJson: datasetFile,
      simRagJson: simRagFile,
    );

    final prompt = catalog.prompts.firstWhere((p) => p.id == 'morning-meal');

    // Condition B: all 10 notes
    final bMessages = buildConditionMessages(
      condition: PromptCondition.conditionB,
      systemPrompt: catalog.systemPrompt,
      prompt: prompt,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );
    expect(bMessages.length, 3);
    for (final fact in catalog.facts) {
      expect(bMessages[1].content, contains(fact.text));
    }
    // Nothing in B hints at abstaining
    expect(bMessages[1].content, isNot(contains(standardRefusalStatement)));
    expect(bMessages[1].content, isNot(contains('No specific local notes match')));

    // Condition C: all 10 notes
    final cMessages = buildConditionMessages(
      condition: PromptCondition.conditionC,
      systemPrompt: catalog.systemPrompt,
      prompt: prompt,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );
    expect(cMessages.length, 3);
    for (final fact in catalog.facts) {
      expect(cMessages[1].content, contains(fact.text));
    }

    // Condition B-sim-RAG: exactly k=2 notes
    final bSimMessages = buildConditionMessages(
      condition: PromptCondition.conditionBSimRag,
      systemPrompt: catalog.systemPrompt,
      prompt: prompt,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );
    expect(bSimMessages.length, 3);
    expect(bSimMessages[1].content, contains('The wisents are not fed by people and find their own food in the dunes.'));
    expect(bSimMessages[1].content, contains('The Kraansvlak herd is a herd of wisents (European bison).'));
    expect(bSimMessages[1].content, isNot(contains('PWN manages the area.')));

    // Condition C-sim-RAG: exactly k=2 notes
    final cSimMessages = buildConditionMessages(
      condition: PromptCondition.conditionCSimRag,
      systemPrompt: catalog.systemPrompt,
      prompt: prompt,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );
    expect(cSimMessages.length, 3);
    expect(cSimMessages[1].content, contains('The wisents are not fed by people and find their own food in the dunes.'));
    expect(cSimMessages[1].content, isNot(contains('PWN manages the area.')));

    // Unmapped prompt gets 0 notes and does not fallback to keyword matching
    const unmappedPrompt = PromptSpec(
      id: 'unmapped-query',
      label: 'Unmapped',
      text: 'What did the wisent eat?',
    );
    final unmappedMessages = buildConditionMessages(
      condition: PromptCondition.conditionBSimRag,
      systemPrompt: catalog.systemPrompt,
      prompt: unmappedPrompt,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );
    // When 0 notes, B variant has only system and user prompt (no abstention hint)
    expect(unmappedMessages.length, 2);
    expect(unmappedMessages[0].role, LLMRole.system);
    expect(unmappedMessages[1].role, LLMRole.user);
    expect(unmappedMessages[1].content, contains('What did the wisent eat?'));
  });

  test('Condition C bans appear only in the C variants', () {
    const baseSystemPrompt = 'Base system note.';
    const prompt = PromptSpec(
      id: 'what-animal',
      label: 'What animal',
      text: 'What animal is this?',
    );
    const facts = [
      DatasetFact(id: 'wisent-herd', text: 'Wisent herd fact.'),
    ];

    const bans = [
      'Do NOT speak as Nature or in the animal\'s first-person voice.',
      'Do NOT claim to decode or translate the animal\'s behaviour.',
      'Do NOT preach or moralise at the visitor.',
      standardRefusalStatement,
    ];

    for (final cond in [PromptCondition.conditionA, PromptCondition.conditionB, PromptCondition.conditionBSimRag]) {
      final msgs = buildConditionMessages(
        condition: cond,
        systemPrompt: baseSystemPrompt,
        prompt: prompt,
        allFacts: facts,
      );
      for (final ban in bans) {
        expect(msgs[0].content, isNot(contains(ban)),
            reason: '$ban should NOT appear in ${cond.code}');
      }
    }

    for (final cond in [PromptCondition.conditionC, PromptCondition.conditionCSimRag]) {
      final msgs = buildConditionMessages(
        condition: cond,
        systemPrompt: baseSystemPrompt,
        prompt: prompt,
        allFacts: facts,
      );
      for (final ban in bans) {
        expect(msgs[0].content, contains(ban),
            reason: '$ban MUST appear in ${cond.code}');
      }
    }
  });

  test('stripThinkingBlocks removes think, thought, and channel thought blocks', () {
    expect(
      stripThinkingBlocks('<think>Let me reason about this...</think>The herd is wisents.'),
      'The herd is wisents.',
    );
    expect(
      stripThinkingBlocks('<thought>Gemma internal reasoning...</thought>The herd is wisents.'),
      'The herd is wisents.',
    );
    expect(
      stripThinkingBlocks('<|thought|>Reasoning inside special tokens...</|thought|>The herd is wisents.'),
      'The herd is wisents.',
    );
    expect(
      stripThinkingBlocks('<|channel>thought\nInternal channel reflection\n<channel|>The herd is wisents.'),
      'The herd is wisents.',
    );
    expect(
      stripThinkingBlocks('<think>Unclosed thinking block...'),
      '',
    );
    expect(
      stripThinkingBlocks('<thought>Unclosed thought block...'),
      '',
    );
    expect(
      stripThinkingBlocks('Direct answer with no thinking.'),
      'Direct answer with no thinking.',
    );
  });

  test('a run record round-trips to CSV and JSON with every field', () {
    final now = DateTime.now();
    final record = RunRecord(
      runIndex: 1,
      runIndexSinceLoad: 1,
      isColdLoad: true,
      loadDurationMs: 1450,
      timestamp: now,
      deviceModel: 'Fairphone 5',
      osVersion: 'Android 14',
      modelFileName: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
      architecture: 'qwen2',
      quantization: 'Q4_K_M',
      backend: 'CPU',
      nGpuLayers: 0,
      condition: 'B',
      notesMode: 'all',
      k: 0,
      injectedNoteIds: ['wisent-herd', 'park'],
      promptId: 'what-animal',
      promptText: 'What animal is this?',
      decodingSettings: const DecodingSettings(
        temperature: 0.0,
        seed: 42,
        topK: 40,
        topP: 0.9,
        repeatPenalty: 1.1,
        maxTokens: 256,
        contextSize: 2048,
      ),
      systemPrompt: 'System prompt content',
      messagesSent: 'system\n...\nuser\n...',
      output: 'The Kraansvlak herd consists of wisents.',
      hitMaxTokens: false,
      ramBeforeLoadBytes: 85 * 1024 * 1024,
      ramIdleBytes: 1320 * 1024 * 1024,
      ramPeakBytes: 1450 * 1024 * 1024,
      promptTokenCount: 142,
      generatedTokenCount: 22,
      ttftMs: 320,
      generationDurationMs: 1100,
      tokensPerSecond: 20.0,
      batteryLevelBefore: 88,
      batteryLevelAfter: 87,
      batteryTempBeforeCelsius: 32.5,
      batteryTempAfterCelsius: 33.1,
      throttlePercentage: null,
      isHeatTest: false,
      thinkingDisabled: true,
    );

    // JSON round-trip
    final jsonMap = record.toJson();
    final jsonStr = jsonEncode(jsonMap);
    final deserialized = RunRecord.fromJson(jsonDecode(jsonStr) as Map<String, dynamic>);

    expect(deserialized.runIndex, record.runIndex);
    expect(deserialized.runIndexSinceLoad, record.runIndexSinceLoad);
    expect(deserialized.isColdLoad, record.isColdLoad);
    expect(deserialized.loadType, 'Cold');
    expect(deserialized.loadDurationMs, record.loadDurationMs);
    expect(deserialized.deviceModel, 'Fairphone 5');
    expect(deserialized.modelFileName, 'qwen2.5-1.5b-instruct-q4_k_m.gguf');
    expect(deserialized.architecture, 'qwen2');
    expect(deserialized.quantization, 'Q4_K_M');
    expect(deserialized.condition, 'B');
    expect(deserialized.promptTokenCount, 142);
    expect(deserialized.injectedNoteIds, ['wisent-herd', 'park']);
    expect(deserialized.decodingSettings.temperature, 0.0);
    expect(deserialized.decodingSettings.seed, 42);
    expect(deserialized.output, record.output);
    expect(deserialized.hitMaxTokens, false);
    expect(deserialized.tokensPerSecond, 20.0);
    expect(deserialized.batteryLevelBefore, 88);
    expect(deserialized.batteryLevelAfter, 87);
    expect(deserialized.batteryTempBeforeCelsius, 32.5);
    expect(deserialized.batteryTempAfterCelsius, 33.1);

    // CSV format verification
    final csv = BenchmarkExporter.exportCsv([record]);
    expect(csv, contains('# Tab: Load and idle'));
    expect(csv, contains('# Tab: Inference'));
    expect(csv, contains('Load Type'));
    expect(csv, isNot(contains('Is Cold Load')));
    expect(csv, contains('Prompt Tokens'));
    expect(csv, contains('Fairphone 5'));
    expect(csv, contains('qwen2.5-1.5b-instruct-q4_k_m.gguf'));
    expect(csv, contains('Q4_K_M'));
    expect(csv, contains('The Kraansvlak herd consists of wisents.'));
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

  test('exact messages sent for one Condition A, one Condition B, and one Condition C run', () {
    final promptsFile = File('assets/prompts.json').readAsStringSync();
    final datasetFile = File('assets/dataset.json').readAsStringSync();
    final simRagFile = File('assets/simulated_rag.json').readAsStringSync();
    final systemPrompt = File('assets/system_prompt.txt').readAsStringSync();

    final catalog = parseCatalog(
      systemPrompt: systemPrompt,
      promptsJson: promptsFile,
      datasetJson: datasetFile,
      simRagJson: simRagFile,
    );

    final prompt = catalog.prompts.firstWhere((p) => p.id == 'what-animal');

    final aMessages = buildConditionMessages(
      condition: PromptCondition.conditionA,
      systemPrompt: catalog.systemPrompt,
      prompt: prompt,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );

    final bMessages = buildConditionMessages(
      condition: PromptCondition.conditionB,
      systemPrompt: catalog.systemPrompt,
      prompt: prompt,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );

    final cMessages = buildConditionMessages(
      condition: PromptCondition.conditionC,
      systemPrompt: catalog.systemPrompt,
      prompt: prompt,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );

    // Print exact messages sent
    // ignore: avoid_print
    print('==================== EXACT MESSAGES: CONDITION A ====================');
    for (var i = 0; i < aMessages.length; i++) {
      // ignore: avoid_print
      print('--- Message ${i + 1} [role: ${aMessages[i].role.name}] ---');
      // ignore: avoid_print
      print(aMessages[i].content);
    }

    // ignore: avoid_print
    print('==================== EXACT MESSAGES: CONDITION B ====================');
    for (var i = 0; i < bMessages.length; i++) {
      // ignore: avoid_print
      print('--- Message ${i + 1} [role: ${bMessages[i].role.name}] ---');
      // ignore: avoid_print
      print(bMessages[i].content);
    }

    // ignore: avoid_print
    print('==================== EXACT MESSAGES: CONDITION C ====================');
    for (var i = 0; i < cMessages.length; i++) {
      // ignore: avoid_print
      print('--- Message ${i + 1} [role: ${cMessages[i].role.name}] ---');
      // ignore: avoid_print
      print(cMessages[i].content);
    }
    // ignore: avoid_print
    print('====================================================================');

    // Structural assertions
    // Condition A: 2 messages (system, user prompt)
    expect(aMessages.length, 2);
    expect(aMessages[0].role, LLMRole.system);
    expect(aMessages[0].content, trimSingleTrailingNewline(catalog.systemPrompt));
    expect(aMessages[1].role, LLMRole.user);
    expect(aMessages[1].content, prompt.text);
    expect(aMessages[0].content, isNot(contains('/no_think')));
    expect(aMessages[1].content, isNot(contains('/no_think')));

    // Condition B: 3 messages (system, local notes with all 10 facts, user prompt)
    expect(bMessages.length, 3);
    expect(bMessages[0].role, LLMRole.system);
    expect(bMessages[0].content, trimSingleTrailingNewline(catalog.systemPrompt));
    expect(bMessages[1].role, LLMRole.user);
    for (final fact in catalog.facts) {
      expect(bMessages[1].content, contains(fact.text));
    }
    expect(bMessages[2].role, LLMRole.user);
    expect(bMessages[2].content, prompt.text);
    // No refusal or abstention hints in B
    expect(bMessages[0].content, isNot(contains(standardRefusalStatement)));
    expect(bMessages[1].content, isNot(contains(standardRefusalStatement)));
    expect(bMessages[0].content, isNot(contains('/no_think')));
    expect(bMessages[1].content, isNot(contains('/no_think')));
    expect(bMessages[2].content, isNot(contains('/no_think')));

    // Condition C: 3 messages (system with constraints + refusal + 3 bans, local notes with all 10 facts, user prompt)
    expect(cMessages.length, 3);
    expect(cMessages[0].role, LLMRole.system);
    expect(cMessages[0].content, contains(standardRefusalStatement));
    expect(cMessages[0].content, contains('1 to 3 concise, factual sentences'));
    expect(cMessages[0].content, contains("Do NOT speak as Nature or in the animal's first-person voice."));
    expect(cMessages[0].content, contains("Do NOT claim to decode or translate the animal's behaviour."));
    expect(cMessages[0].content, contains('Do NOT preach or moralise at the visitor.'));
    expect(cMessages[1].role, LLMRole.user);
    for (final fact in catalog.facts) {
      expect(cMessages[1].content, contains(fact.text));
    }
    expect(cMessages[2].role, LLMRole.user);
    expect(cMessages[2].content, prompt.text);
    expect(cMessages[0].content, isNot(contains('/no_think')));
    expect(cMessages[1].content, isNot(contains('/no_think')));
    expect(cMessages[2].content, isNot(contains('/no_think')));
  });

  test('BenchmarkExporter.exportToFiles saves files locally to /exports directory', () async {
    final record = RunRecord(
      runIndex: 1,
      runIndexSinceLoad: 1,
      isColdLoad: true,
      timestamp: DateTime.now(),
      deviceModel: 'Test Device',
      osVersion: 'Test OS',
      modelFileName: 'test-model.gguf',
      architecture: 'test-arch',
      quantization: 'Q4_K_M',
      backend: 'CPU',
      nGpuLayers: 0,
      condition: 'A',
      notesMode: 'none',
      k: 0,
      injectedNoteIds: const [],
      promptId: 'what-animal',
      promptText: 'What animal is this?',
      decodingSettings: const DecodingSettings(),
      systemPrompt: 'System',
      messagesSent: 'Messages',
      output: 'Output text',
      hitMaxTokens: false,
    );

    final paths = await BenchmarkExporter.exportToFiles([record]);
    final csvPath = paths['csv']!;
    final jsonPath = paths['json']!;

    expect(csvPath, contains('exports'));
    expect(jsonPath, contains('exports'));

    final csvFile = File(csvPath);
    final jsonFile = File(jsonPath);

    expect(csvFile.existsSync(), isTrue);
    expect(jsonFile.existsSync(), isTrue);

    expect(csvFile.readAsStringSync(), contains('Test Device'));
    expect(jsonFile.readAsStringSync(), contains('test-model.gguf'));

    // Clean up test export files
    if (csvFile.existsSync()) csvFile.deleteSync();
    if (jsonFile.existsSync()) jsonFile.deleteSync();
  });

  test('sanitizeModelOutput truncates at stop sequences and strips thinking blocks', () {
    expect(
      sanitizeModelOutput('The wisents live here.<|im_end|>'),
      'The wisents live here.',
    );
    expect(
      sanitizeModelOutput(
        'The wisents live here.<|im_end|>\n<|im_start|>user\nWho are they?<|im_start|>assistant\nThey are wild bison.',
      ),
      'The wisents live here.',
    );
    expect(
      sanitizeModelOutput(
        '<think>Thinking about wisents...</think>The herd is wisents.<end_of_turn>',
      ),
      'The herd is wisents.',
    );
    expect(
      sanitizeModelOutput('Clean response without stop markers.'),
      'Clean response without stop markers.',
    );
    expect(
      sanitizeModelOutput('Sentence one.<|eot_id|>Sentence two.<|end|><turn|>'),
      'Sentence one.',
    );
  });

  test('RunPersistenceManager.loadRuns loads persisted runs', () async {
    final runs = await RunPersistenceManager.loadRuns();
    // Should be able to query without throwing
    expect(runs, isA<List<RunRecord>>());
  });

  test('isColdLoad and loadType behavior across first and subsequent runs', () {
    final run1 = RunRecord(
      runIndex: 1,
      runIndexSinceLoad: 1,
      isColdLoad: true,
      timestamp: DateTime.now(),
      deviceModel: 'Test Device',
      osVersion: 'Test OS',
      modelFileName: 'model.gguf',
      architecture: 'gemma',
      quantization: 'Q4_K_M',
      backend: 'CPU',
      nGpuLayers: 0,
      condition: 'A',
      notesMode: 'none',
      k: 0,
      injectedNoteIds: const [],
      promptId: 'what-animal',
      promptText: 'What animal is this?',
      decodingSettings: const DecodingSettings(),
      systemPrompt: 'System',
      messagesSent: 'Msg',
      output: 'Output',
      hitMaxTokens: false,
    );

    final run2 = RunRecord(
      runIndex: 2,
      runIndexSinceLoad: 2,
      isColdLoad: false,
      timestamp: DateTime.now(),
      deviceModel: 'Test Device',
      osVersion: 'Test OS',
      modelFileName: 'model.gguf',
      architecture: 'gemma',
      quantization: 'Q4_K_M',
      backend: 'CPU',
      nGpuLayers: 0,
      condition: 'A',
      notesMode: 'none',
      k: 0,
      injectedNoteIds: const [],
      promptId: 'what-animal',
      promptText: 'What animal is this?',
      decodingSettings: const DecodingSettings(),
      systemPrompt: 'System',
      messagesSent: 'Msg',
      output: 'Output',
      hitMaxTokens: false,
    );

    expect(run1.isColdLoad, isTrue);
    expect(run1.loadType, 'Cold');
    expect(run2.isColdLoad, isFalse);
    expect(run2.loadType, 'Warm');
  });

  test('thinkingDisabled is null/N/A when no toggle applied, and true/Yes when applied', () {
    final recordNoToggle = RunRecord(
      runIndex: 1,
      runIndexSinceLoad: 1,
      isColdLoad: true,
      timestamp: DateTime.now(),
      deviceModel: 'Test Device',
      osVersion: 'Test OS',
      modelFileName: 'gemma-4.gguf',
      architecture: 'gemma',
      quantization: 'Q4_K_M',
      backend: 'CPU',
      nGpuLayers: 0,
      condition: 'A',
      notesMode: 'none',
      k: 0,
      injectedNoteIds: const [],
      promptId: 'what-animal',
      promptText: 'What animal is this?',
      decodingSettings: const DecodingSettings(),
      systemPrompt: 'System',
      messagesSent: 'Msg',
      output: 'Output',
      hitMaxTokens: false,
      thinkingDisabled: null,
    );

    final recordWithToggle = RunRecord(
      runIndex: 2,
      runIndexSinceLoad: 2,
      isColdLoad: false,
      timestamp: DateTime.now(),
      deviceModel: 'Test Device',
      osVersion: 'Test OS',
      modelFileName: 'qwen3.gguf',
      architecture: 'qwen3',
      quantization: 'Q4_K_M',
      backend: 'CPU',
      nGpuLayers: 0,
      condition: 'A',
      notesMode: 'none',
      k: 0,
      injectedNoteIds: const [],
      promptId: 'what-animal',
      promptText: 'What animal is this?',
      decodingSettings: const DecodingSettings(),
      systemPrompt: 'System',
      messagesSent: 'Msg',
      output: 'Output',
      hitMaxTokens: false,
      thinkingDisabled: true,
    );

    expect(recordNoToggle.thinkingDisabled, isNull);
    final jsonNoToggle = recordNoToggle.toJson();
    expect(jsonNoToggle['thinkingDisabled'], isNull);
    final roundTrippedNoToggle = RunRecord.fromJson(jsonNoToggle);
    expect(roundTrippedNoToggle.thinkingDisabled, isNull);

    final csvNoToggle = BenchmarkExporter.exportCsv([recordNoToggle]);
    expect(csvNoToggle, contains('N/A'));

    expect(recordWithToggle.thinkingDisabled, isTrue);
    final jsonWithToggle = recordWithToggle.toJson();
    expect(jsonWithToggle['thinkingDisabled'], isTrue);
    final roundTrippedWithToggle = RunRecord.fromJson(jsonWithToggle);
    expect(roundTrippedWithToggle.thinkingDisabled, isTrue);

    final csvWithToggle = BenchmarkExporter.exportCsv([recordWithToggle]);
    expect(csvWithToggle, contains('Yes'));
  });

  test('gemma output keeps a thought block and still cuts end markers', () {
    const raw =
        '<|channel>thought\nsecret\n<channel|>The herd is wisents.<turn|>';
    expect(outputContainsThought(raw), isTrue);
    expect(
      sanitizeModelOutput(raw, keepThoughts: true),
      '<|channel>thought\nsecret\n<channel|>The herd is wisents.',
    );
    expect(
      sanitizeModelOutput('Answer<eos>more', keepThoughts: true),
      'Answer',
    );
    expect(sanitizeModelOutput('Answer<eos>more'), 'Answer<eos>more');
    expect(
      sanitizeModelOutput(
        '<think>Thinking about wisents...</think>The herd is wisents.<end_of_turn>',
      ),
      'The herd is wisents.',
    );
  });

  test('export records the formatter and whether a thought was emitted', () {
    final record = RunRecord(
      runIndex: 1,
      runIndexSinceLoad: 1,
      isColdLoad: false,
      timestamp: DateTime.utc(2026, 10, 20),
      deviceModel: 'Test Device',
      osVersion: 'Test OS',
      modelFileName: 'gemma-4.gguf',
      architecture: 'gemma4',
      quantization: 'Q3_K_M',
      backend: 'GPU',
      nGpuLayers: 99,
      condition: 'A',
      notesMode: 'none',
      k: 0,
      injectedNoteIds: const [],
      promptId: 'what-animal',
      promptText: 'What animal is this?',
      decodingSettings: const DecodingSettings(),
      systemPrompt: 'System',
      messagesSent: 'Msg',
      output: '<|channel>thought\nsecret\n<channel|>Wisents.',
      hitMaxTokens: false,
      promptFormatter: 'gemma4',
      thoughtEmitted: true,
    );

    final json = record.toJson();
    expect(json['promptFormatter'], 'gemma4');
    expect(json['thoughtEmitted'], isTrue);
    final roundTripped = RunRecord.fromJson(json);
    expect(roundTripped.promptFormatter, 'gemma4');
    expect(roundTripped.thoughtEmitted, isTrue);

    final legacy = Map<String, dynamic>.from(json)
      ..remove('promptFormatter')
      ..remove('thoughtEmitted');
    final legacyRecord = RunRecord.fromJson(legacy);
    expect(legacyRecord.promptFormatter, isNull);
    expect(legacyRecord.thoughtEmitted, isFalse);

    final csv = BenchmarkExporter.exportCsv([record]);
    expect(
      csv,
      contains(
        'Thinking Disabled,Prompt Formatter,Thought Emitted,Output',
      ),
    );
    expect(csv, contains('gemma4'));
  });
}

