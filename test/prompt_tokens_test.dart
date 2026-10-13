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

    expect(factsSinceWhen.map((f) => f.id).toList(), ['arrival', 'wisent-herd']);
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

    final sinceWhenBsim = messagesFor(pSinceWhen, PromptCondition.conditionBSimRag);
    final calvesBsim = messagesFor(pCalves, PromptCondition.conditionBSimRag);
    expect(sinceWhenBsim[0].content, calvesBsim[0].content);
    expect(sinceWhenBsim[1].content, calvesBsim[1].content);

    final bindings = resolveLlamaBindings();
    expect(bindings, isNotNull, reason: 'llama bindings are required');

    final gemmaPath = '/Users/teaguemm/Desktop/gemma-4-E2B-it-Q3_K_M.gguf';
    if (File(gemmaPath).existsSync()) {
      final counts = _countsFor(bindings!, gemmaPath, {
        'since-when-bsim': sinceWhenBsim,
        'calves-bsim': calvesBsim,
        'since-when-a': messagesFor(pSinceWhen, PromptCondition.conditionA),
      });
      // ignore: avoid_print
      print(
        'gemma B-sim since-when=${counts['since-when-bsim']} '
        'calves=${counts['calves-bsim']} A=${counts['since-when-a']}',
      );
      final sinceWhen = counts['since-when-bsim']!;
      final calves = counts['calves-bsim']!;
      final conditionA = counts['since-when-a']!;
      expect((sinceWhen - calves).abs(), lessThanOrEqualTo(5));
      expect(sinceWhen, greaterThan(conditionA + 15));
    }

    final qwenPath = '/Users/teaguemm/Desktop/qwen2.5-1.5b-instruct-q4_k_m.gguf';
    if (File(qwenPath).existsSync()) {
      final whatAnimal = catalog.prompts.firstWhere((p) => p.id == 'what-animal');
      final counts = _countsFor(bindings!, qwenPath, {
        'A': messagesFor(whatAnimal, PromptCondition.conditionA),
        'B': messagesFor(whatAnimal, PromptCondition.conditionB),
        'Bsim': messagesFor(whatAnimal, PromptCondition.conditionBSimRag),
        'C': messagesFor(whatAnimal, PromptCondition.conditionC),
      });
      // ignore: avoid_print
      print(
        'qwen what-animal A=${counts['A']} B=${counts['B']} '
        'B-sim=${counts['Bsim']} C=${counts['C']}',
      );
      expect(counts['A'], 58);
      expect(counts['B'], 274);
      expect(counts['Bsim'], 115);
      expect(counts['C'], 420);
    }
  });
}

Map<String, int> _countsFor(
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
