// ignore_for_file: implementation_imports

import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:llm_llamacpp/src/bindings/llama_bindings.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/prompt_tokens.dart';

void main() {
  test('prompt token counts are the full formatted prompt', () {
    final catalog = parseCatalog(
      systemPrompt: File('assets/system_prompt.txt').readAsStringSync(),
      promptsJson: File('assets/prompts.json').readAsStringSync(),
      datasetJson: File('assets/dataset.json').readAsStringSync(),
      simRagJson: File('assets/simulated_rag.json').readAsStringSync(),
    );

    final pSinceWhen = catalog.prompts.firstWhere((p) => p.id == 'since-when');
    final pCalves = catalog.prompts.firstWhere((p) => p.id == 'calves');

    final factsSinceWhen = selectFactsForCondition(
      condition: PromptCondition.conditionBSimRag,
      prompt: pSinceWhen,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );
    final factsCalves = selectFactsForCondition(
      condition: PromptCondition.conditionBSimRag,
      prompt: pCalves,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );

    expect(factsSinceWhen.map((f) => f.id).toList(), [
      'arrival',
      'wisent-herd',
    ]);
    expect(factsCalves.map((f) => f.id).toList(), ['arrival', 'wisent-herd']);

    List<LLMMessage> messagesFor(PromptSpec prompt, PromptCondition condition) {
      return buildConditionMessages(
        condition: condition,
        systemPrompt: catalog.systemPrompt,
        prompt: prompt,
        allFacts: catalog.facts,
        simRagConfig: catalog.simRagConfig,
      );
    }

    final sinceWhenBsim = messagesFor(
      pSinceWhen,
      PromptCondition.conditionBSimRag,
    );
    final calvesBsim = messagesFor(pCalves, PromptCondition.conditionBSimRag);
    expect(sinceWhenBsim[0].content, calvesBsim[0].content);
    expect(sinceWhenBsim[1].content, calvesBsim[1].content);

    final bindings = resolveLlamaBindings();
    expect(bindings, isNotNull, reason: 'llama bindings are required');

    // Model paths come from the environment. A missing variable or file skips
    // that model's token counts; the message checks above still run.
    // GEMMA_GGUF, QWEN_GGUF
    final gemmaPath = Platform.environment['GEMMA_GGUF'];
    if (gemmaPath != null &&
        gemmaPath.isNotEmpty &&
        File(gemmaPath).existsSync()) {
      final counts = _countsFor(bindings!, gemmaPath, {
        'since-when-bsim': sinceWhenBsim,
        'calves-bsim': calvesBsim,
        'since-when-a': messagesFor(pSinceWhen, PromptCondition.conditionA),
      });
      // ignore: avoid_print
      print(
        'gemma B-sim since-when=${counts['since-when-bsim']!.tokenCount} '
        'calves=${counts['calves-bsim']!.tokenCount} '
        'A=${counts['since-when-a']!.tokenCount}',
      );
      final sinceWhen = counts['since-when-bsim']!.tokenCount;
      final calves = counts['calves-bsim']!.tokenCount;
      final conditionA = counts['since-when-a']!.tokenCount;
      expect(counts['since-when-bsim']!.formatter, promptFormatterGemma4);
      expect((sinceWhen - calves).abs(), lessThanOrEqualTo(5));
      expect(sinceWhen, greaterThan(conditionA + 15));
      _expectGemmaProtocolCounts(bindings, gemmaPath, catalog, messagesFor);
    }

    final qwenPath = Platform.environment['QWEN_GGUF'];
    if (qwenPath != null &&
        qwenPath.isNotEmpty &&
        File(qwenPath).existsSync()) {
      final whatAnimal = catalog.prompts.firstWhere(
        (p) => p.id == 'what-animal',
      );
      final counts = _countsFor(bindings!, qwenPath, {
        'A': messagesFor(whatAnimal, PromptCondition.conditionA),
        'B': messagesFor(whatAnimal, PromptCondition.conditionB),
        'Bsim': messagesFor(whatAnimal, PromptCondition.conditionBSimRag),
        'C': messagesFor(whatAnimal, PromptCondition.conditionC),
      });
      // ignore: avoid_print
      print(
        'qwen what-animal A=${counts['A']!.tokenCount} '
        'B=${counts['B']!.tokenCount} '
        'B-sim=${counts['Bsim']!.tokenCount} C=${counts['C']!.tokenCount}',
      );
      expect(counts['A']!.formatter, promptFormatterNative);
      expect(counts['A']!.tokenCount, 58);
      expect(counts['B']!.tokenCount, 274);
      expect(counts['Bsim']!.tokenCount, 115);
      expect(counts['C']!.tokenCount, 420);
    }
  });
}

Map<String, CountedPrompt> _countsFor(
  LlamaBindings bindings,
  String modelPath,
  Map<String, List<LLMMessage>> labeled,
) {
  final modelParams = bindings.llama_model_default_params();
  final pathPtr = modelPath.toNativeUtf8();
  final model = bindings.llama_model_load_from_file(
    pathPtr.cast(),
    modelParams,
  );
  calloc.free(pathPtr);
  expect(model.address, isNot(0), reason: 'failed to load $modelPath');
  try {
    final vocab = bindings.llama_model_get_vocab(model);
    return {
      for (final entry in labeled.entries)
        entry.key: countPromptTokens(
          model: model,
          vocab: vocab,
          messages: entry.value,
          bindings: bindings,
        )!,
    };
  } finally {
    bindings.llama_model_free(model);
  }
}

/// what-animal on this GGUF.
///
/// Condition A is 60, the Gemma 4 count for the 11 Oct prompt. Conditions B
/// and C are 280 and 434. The 11 Oct figures 330 and 484 counted the same
/// turns while every note still began with "EXAMPLE, not approved: ". That
/// prefix is 50 tokens and is not in the current notes, so B and C are each
/// 50 below those figures. The ChatML counts for the same messages are
/// 87/318/472; the 11 Oct log's 368/522 is the same 50-token prefix.
void _expectGemmaProtocolCounts(
  LlamaBindings bindings,
  String modelPath,
  CatalogData catalog,
  List<LLMMessage> Function(PromptSpec prompt, PromptCondition condition)
  messagesFor,
) {
  final prompt = catalog.prompts.firstWhere((item) => item.id == 'what-animal');
  final counted = _countsFor(bindings, modelPath, {
    'A': messagesFor(prompt, PromptCondition.conditionA),
    'B': messagesFor(prompt, PromptCondition.conditionB),
    'C': messagesFor(prompt, PromptCondition.conditionC),
  });
  // ignore: avoid_print
  print(
    'gemma what-animal A=${counted['A']!.tokenCount} '
    'B=${counted['B']!.tokenCount} C=${counted['C']!.tokenCount} '
    'formatter=${counted['A']!.formatter}',
  );
  expect(counted['A']!.formatter, promptFormatterGemma4);
  expect(counted['B']!.formatter, promptFormatterGemma4);
  expect(counted['C']!.formatter, promptFormatterGemma4);
  expect(counted['A']!.tokenCount, 60);
  expect(counted['B']!.tokenCount, 280);
  expect(counted['C']!.tokenCount, 434);
}
