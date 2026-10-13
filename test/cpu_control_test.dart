// ignore_for_file: implementation_imports

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:llm_llamacpp/src/bindings/llama_bindings.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/prompt_tokens.dart';
import 'package:technical_exploration/sanity_check.dart';

/// Mac CPU control for the Android garbage-output investigation.
///
/// The Flutter test runner does not load the app's native-asset bundle, so
/// this uses the same bindings fallback as the token-count test and asks
/// llama.cpp for `n_gpu_layers = 0`.
///
/// Not part of the default suite. Run with:
/// `RUN_CPU_CONTROL=1 flutter test test/cpu_control_test.dart`
void main() {
  test('n_gpu_layers 0 answers the sanity prompts and what-animal A', () {
    if (Platform.environment['RUN_CPU_CONTROL'] != '1') return;

    const path = '/Users/teaguemm/Desktop/qwen2.5-1.5b-instruct-q4_k_m.gguf';
    expect(File(path).existsSync(), isTrue);
    final bindings = resolveLlamaBindings();
    expect(bindings, isNotNull);

    final info = bindings!
        .llama_print_system_info()
        .cast<Utf8>()
        .toDartString();
    // ignore: avoid_print
    print('cpu-control system: ${info.replaceAll('\n', ' | ')}');

    final modelParams = bindings.llama_model_default_params();
    modelParams.n_gpu_layers = 0;
    final pathPtr = path.toNativeUtf8();
    final model = bindings.llama_model_load_from_file(
      pathPtr.cast(),
      modelParams,
    );
    calloc.free(pathPtr);
    expect(model.address, isNot(0));
    final vocab = bindings.llama_model_get_vocab(model);

    try {
      for (final prompt in sanityPrompts) {
        final text = _generate(
          bindings: bindings,
          model: model,
          vocab: vocab,
          messages: [LLMMessage(role: LLMRole.user, content: prompt.text)],
          maxTokens: sanityMaxTokens,
        );
        final reason = sanityFailureReason(
          output: text.output,
          hitMaxTokens: text.tokens >= sanityMaxTokens,
          requireDigitFour: prompt.requireDigitFour,
          requireAnimalWord: prompt.requireAnimalWord,
        );
        expect(reason, isNull, reason: 'sanity failed: ${text.output}');
      }

      final catalog = parseCatalog(
        systemPrompt: File('assets/system_prompt.txt').readAsStringSync(),
        promptsJson: File('assets/prompts.json').readAsStringSync(),
        datasetJson: File('assets/dataset.json').readAsStringSync(),
        simRagJson: File('assets/simulated_rag.json').readAsStringSync(),
      );
      final whatAnimal = catalog.prompts.firstWhere(
        (p) => p.id == 'what-animal',
      );
      final answer = _generate(
        bindings: bindings,
        model: model,
        vocab: vocab,
        messages: buildConditionMessages(
          condition: PromptCondition.conditionA,
          systemPrompt: catalog.systemPrompt,
          prompt: whatAnimal,
          allFacts: catalog.facts,
          simRagConfig: catalog.simRagConfig,
        ),
        maxTokens: 64,
      );
      expect(answer.tokens, lessThan(64));
      expect(looksLikeTokenSoup(answer.output), isFalse);
      expect(answer.output.trim().length, greaterThan(10));
    } finally {
      bindings.llama_model_free(model);
    }
  }, timeout: const Timeout(Duration(minutes: 10)));
}

class _Gen {
  const _Gen(this.output, this.tokens);
  final String output;
  final int tokens;
}

_Gen _generate({
  required LlamaBindings bindings,
  required ffi.Pointer<llama_model> model,
  required ffi.Pointer<llama_vocab> vocab,
  required List<LLMMessage> messages,
  required int maxTokens,
}) {
  final formatted = _format(bindings, model, messages);
  final promptPtr = formatted.toNativeUtf8();
  final promptBytes = promptPtr.length;
  final tokBuf = calloc<ffi.Int32>(promptBytes + 32);
  final nPrompt = bindings.llama_tokenize(
    vocab,
    promptPtr.cast(),
    promptBytes,
    tokBuf,
    promptBytes + 32,
    true,
    true,
  );
  calloc.free(promptPtr);
  expect(nPrompt, greaterThan(0));

  final ctxParams = bindings.llama_context_default_params();
  ctxParams.n_ctx = 2048;
  ctxParams.n_batch = 2048;
  final ctx = bindings.llama_init_from_model(model, ctxParams);
  expect(ctx.address, isNot(0));

  final sampler = _sampler(bindings, vocab);
  try {
    final batch = bindings.llama_batch_get_one(tokBuf, nPrompt);
    expect(bindings.llama_decode(ctx, batch), 0);

    final pieceBuf = calloc<ffi.Char>(256);
    final one = calloc<ffi.Int32>(1);
    final bytes = <int>[];
    var generated = 0;
    try {
      while (generated < maxTokens) {
        final id = bindings.llama_sampler_sample(sampler, ctx, -1);
        if (bindings.llama_vocab_is_eog(vocab, id)) break;
        final n = bindings.llama_token_to_piece(
          vocab,
          id,
          pieceBuf,
          256,
          0,
          true,
        );
        if (n > 0) {
          bytes.addAll(pieceBuf.cast<ffi.Uint8>().asTypedList(n));
        }
        final text = utf8.decode(bytes, allowMalformed: true);
        if (text.contains('<|im_end|>')) break;
        one[0] = id;
        final next = bindings.llama_batch_get_one(one, 1);
        if (bindings.llama_decode(ctx, next) != 0) break;
        generated++;
      }
    } finally {
      calloc.free(pieceBuf);
      calloc.free(one);
    }
    final output = utf8
        .decode(bytes, allowMalformed: true)
        .split('<|im_end|>')
        .first;
    // ignore: avoid_print
    print('cpu-control tokens=$generated text=$output');
    return _Gen(output.trim(), generated);
  } finally {
    calloc.free(tokBuf);
    bindings.llama_sampler_free(sampler);
    bindings.llama_free(ctx);
  }
}

ffi.Pointer<llama_sampler> _sampler(
  LlamaBindings bindings,
  ffi.Pointer<llama_vocab> vocab,
) {
  final params = bindings.llama_sampler_chain_default_params();
  final sampler = bindings.llama_sampler_chain_init(params);
  bindings.llama_sampler_chain_add(
    sampler,
    bindings.llama_sampler_init_temp(0),
  );
  bindings.llama_sampler_chain_add(
    sampler,
    bindings.llama_sampler_init_top_k(40),
  );
  bindings.llama_sampler_chain_add(
    sampler,
    bindings.llama_sampler_init_top_p(0.9, 1),
  );
  bindings.llama_sampler_chain_add(
    sampler,
    bindings.llama_sampler_init_penalties(
      bindings.llama_vocab_n_tokens(vocab),
      64,
      1.1,
      0,
      0,
    ),
  );
  bindings.llama_sampler_chain_add(
    sampler,
    bindings.llama_sampler_init_dist(42),
  );
  return sampler;
}

String _format(
  LlamaBindings bindings,
  ffi.Pointer<llama_model> model,
  List<LLMMessage> messages,
) {
  final chat = calloc<llama_chat_message>(messages.length);
  final allocated = <ffi.Pointer<Utf8>>[];
  try {
    for (var i = 0; i < messages.length; i++) {
      final role = messages[i].role.name.toNativeUtf8();
      final content = (messages[i].content ?? '').toNativeUtf8();
      allocated.add(role);
      allocated.add(content);
      chat[i].role = role.cast();
      chat[i].content = content.cast();
    }
    final tmpl = bindings.llama_model_chat_template(model, ffi.nullptr);
    final size = bindings.llama_chat_apply_template(
      tmpl,
      chat,
      messages.length,
      true,
      ffi.nullptr,
      0,
    );
    expect(size, greaterThan(0), reason: 'Qwen template should apply');
    final buf = calloc<ffi.Char>(size + 1);
    try {
      final n = bindings.llama_chat_apply_template(
        tmpl,
        chat,
        messages.length,
        true,
        buf,
        size + 1,
      );
      expect(n, greaterThan(0));
      return buf.cast<Utf8>().toDartString(length: n);
    } finally {
      calloc.free(buf);
    }
  } finally {
    for (final p in allocated) {
      calloc.free(p);
    }
    calloc.free(chat);
  }
}
