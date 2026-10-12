import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_llamacpp/src/bindings/llama_bindings.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/prompt_tokens.dart';

void main() {
  test('B-sim-RAG since-when and calves get the same notes and differ by only a few tokens', () {
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

    // Verify both prompts receive the exact same 2 notes under B-sim-RAG
    expect(factsSinceWhen.map((f) => f.id).toList(), ['arrival', 'wisent-herd']);
    expect(factsCalves.map((f) => f.id).toList(), ['arrival', 'wisent-herd']);

    final msgsSinceWhen = buildConditionMessages(
      condition: PromptCondition.conditionBSimRag,
      systemPrompt: catalog.systemPrompt,
      prompt: pSinceWhen,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );

    final msgsCalves = buildConditionMessages(
      condition: PromptCondition.conditionBSimRag,
      systemPrompt: catalog.systemPrompt,
      prompt: pCalves,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );

    // Both prompts have 3 messages: system, local notes, question
    expect(msgsSinceWhen.length, 3);
    expect(msgsCalves.length, 3);
    expect(msgsSinceWhen[0].content, msgsCalves[0].content); // identical system prompt
    expect(msgsSinceWhen[1].content, msgsCalves[1].content); // identical injected notes

    // If local test models exist, verify full prompt token counts differ by only a few tokens
    final modelPaths = [
      '/Users/teaguemm/Desktop/gemma-4-E2B-it-Q3_K_M.gguf',
      '/Users/teaguemm/Desktop/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    ];

    for (final modelPath in modelPaths) {
      if (!File(modelPath).existsSync()) continue;

      final bindings = resolveLlamaBindings();
      if (bindings == null) continue;

      final modelParams = bindings.llama_model_default_params();
      final pathPtr = modelPath.toNativeUtf8();
      final model = bindings.llama_model_load_from_file(pathPtr.cast(), modelParams);
      calloc.free(pathPtr);
      if (model.address == 0) continue;

      final vocab = bindings.llama_model_get_vocab(model);

      final tokensSinceWhen = countPromptTokens(
        model: model,
        vocab: vocab,
        messages: msgsSinceWhen,
        bindings: bindings,
      );

      final tokensCalves = countPromptTokens(
        model: model,
        vocab: vocab,
        messages: msgsCalves,
        bindings: bindings,
      );

      bindings.llama_model_free(model);

      expect(tokensSinceWhen, isNotNull);
      expect(tokensCalves, isNotNull);

      // Verify that full prompt length difference is only a few tokens (e.g. 3 tokens)
      final diff = (tokensSinceWhen! - tokensCalves!).abs();
      expect(diff, lessThanOrEqualTo(5),
          reason: 'Prompt tokens should differ by only a few tokens, got $tokensSinceWhen vs $tokensCalves (diff $diff)');
    }
  });
}

