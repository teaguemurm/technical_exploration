part of 'persistent_inference_isolate.dart';

void _handleInferenceRequest(
  _InferenceRequestMessage request,
  SendPort mainSendPort,
  ffi.DynamicLibrary lib,
  LlamaBindings bindings,
) {
  ffi.Pointer<llama_adapter_lora>? loraAdapter;

  try {
    final modelParams = bindings.llama_model_default_params();
    modelParams.n_gpu_layers = request.nGpuLayers;
    // The nommap APK passes --dart-define=NO_MMAP=true. Other builds leave
    // llama.cpp's AUTO load mode, which is what the failed Fold runs used.
    const noMmap = bool.fromEnvironment('NO_MMAP');
    if (noMmap) {
      modelParams.load_mode = llama_load_mode.LLAMA_LOAD_MODE_NONE;
    }

    final modelPathPtr = request.modelPath.toNativeUtf8();
    final model = bindings.llama_model_load_from_file(
      modelPathPtr.cast(),
      modelParams,
    );
    calloc.free(modelPathPtr);

    if (model.address == 0) {
      mainSendPort.send(
        _IsolateResponse(
          requestId: request.requestId,
          payload: InferenceError(
            'Failed to load model from ${request.modelPath}',
          ),
          isComplete: true,
        ),
      );
      return;
    }

    if (request.loraPath != null) {
      final loraPathPtr = request.loraPath!.toNativeUtf8();
      loraAdapter = bindings.llama_adapter_lora_init(model, loraPathPtr.cast());
      calloc.free(loraPathPtr);

      if (loraAdapter.address == 0) {
        bindings.llama_model_free(model);
        mainSendPort.send(
          _IsolateResponse(
            requestId: request.requestId,
            payload: InferenceError('Failed to load LoRA adapter'),
            isComplete: true,
          ),
        );
        return;
      }
    }

    final vocab = bindings.llama_model_get_vocab(model);

    // Log vocab/special-token diagnostics so we can quickly see whether the
    // tokenizer is auto-prepending BOS, what BOS/EOS ids are, etc. These
    // numbers are essential when the formatted chat prompt does not produce
    // coherent output (a sign the chat template / special-token plumbing is
    // mismatched against this particular model).
    try {
      final bosId = bindings.llama_vocab_bos(vocab);
      final eosId = bindings.llama_vocab_eos(vocab);
      final eotId = bindings.llama_vocab_eot(vocab);
      final addBos = bindings.llama_vocab_get_add_bos(vocab);
      final addEos = bindings.llama_vocab_get_add_eos(vocab);
      final nTokensInVocab = bindings.llama_vocab_n_tokens(vocab);
      // ignore: avoid_print
      print(
        '[inference_isolate_handler] Vocab: nTokens=$nTokensInVocab '
        'bos=$bosId eos=$eosId eot=$eotId '
        'add_bos_token=$addBos add_eos_token=$addEos',
      );
    } catch (e) {
      // ignore: avoid_print
      print(
        '[inference_isolate_handler] Could not query vocab diagnostics: $e',
      );
    }

    final ctxParams = bindings.llama_context_default_params();
    ctxParams.n_ctx = request.contextSize;
    ctxParams.n_batch = request.batchSize;
    if (request.threads != null) {
      ctxParams.n_threads = request.threads!;
      ctxParams.n_threads_batch = request.threads!;
    }

    final ctx = bindings.llama_init_from_model(model, ctxParams);
    if (ctx.address == 0) {
      if (loraAdapter != null) {
        bindings.llama_adapter_lora_free(loraAdapter);
      }
      bindings.llama_model_free(model);
      mainSendPort.send(
        _IsolateResponse(
          requestId: request.requestId,
          payload: InferenceError('Failed to create context'),
          isComplete: true,
        ),
      );
      return;
    }

    if (loraAdapter != null) {
      final result = setSingleContextLoraAdapter(
        bindings,
        ctx,
        loraAdapter,
        request.loraScale,
      );
      if (result != 0) {
        bindings.llama_free(ctx);
        bindings.llama_adapter_lora_free(loraAdapter);
        bindings.llama_model_free(model);
        mainSendPort.send(
          _IsolateResponse(
            requestId: request.requestId,
            payload: InferenceError('Failed to apply LoRA adapter'),
            isComplete: true,
          ),
        );
        return;
      }
    }

    try {
      final usingChatTemplate =
          request.messages != null && request.messages!.isNotEmpty;
      final FormattedPrompt formatted;
      if (usingChatTemplate) {
        formatted = _applyNativeChatTemplate(
          bindings,
          model,
          request.messages!,
          toolSchemasJson: request.toolSchemasJson,
        );
      } else {
        // Raw prompt. No chat formatter ran, so BOS follows the tokenizer.
        formatted = FormattedPrompt(
          text: request.prompt,
          formatter: promptFormatterNative,
          bosIncludedInText: false,
          template: readModelChatTemplate(bindings, model),
        );
      }
      final prompt = formatted.text;
      final promptFormatter = usingChatTemplate ? formatted.formatter : null;

      final List<int> tokenIds;
      try {
        tokenIds = tokenizeFormattedPrompt(
          bindings: bindings,
          vocab: vocab,
          formatted: formatted,
          usingChatTemplate: usingChatTemplate,
        );
      } on ChatTemplateException catch (error) {
        // ignore: avoid_print
        print('[inference_isolate_handler] ${error.message}');
        mainSendPort.send(
          _IsolateResponse(
            requestId: request.requestId,
            payload: InferenceError(error.message),
            isComplete: true,
          ),
        );
        return;
      }

      final nTokens = tokenIds.length;
      final tokensPtr = calloc<ffi.Int32>(nTokens);
      for (var i = 0; i < nTokens; i++) {
        tokensPtr[i] = tokenIds[i];
      }

      // ignore: avoid_print
      print(
        '[inference_isolate_handler] Tokenized prompt: $nTokens tokens '
        '(${prompt.length} chars, contextSize=${request.contextSize})',
      );

      // Decode the first few tokens so we can see whether ChatML markers like
      // `<|im_start|>` are tokenizing as single special tokens or being split
      // into many literal-character tokens. The latter is a strong sign that
      // the model's vocab does not contain those markers and the chat
      // template we used is wrong for this model.
      try {
        final previewCount = nTokens < 15 ? nTokens : 15;
        final pieceBuf = calloc<ffi.Char>(64);
        final preview = StringBuffer();
        try {
          for (var i = 0; i < previewCount; i++) {
            final id = tokensPtr[i];
            final pieceLen = bindings.llama_token_to_piece(
              vocab,
              id,
              pieceBuf,
              64,
              0,
              true,
            );
            String text;
            if (pieceLen <= 0) {
              text = '?';
            } else {
              text = pieceBuf
                  .cast<Utf8>()
                  .toDartString(length: pieceLen)
                  .replaceAll('\n', r'\n')
                  .replaceAll('\r', r'\r');
            }
            preview.write('  [$i] id=$id "$text"\n');
          }
        } finally {
          calloc.free(pieceBuf);
        }
        // ignore: avoid_print
        print(
          '[inference_isolate_handler] First $previewCount token(s):\n$preview',
        );
      } catch (e) {
        // ignore: avoid_print
        print('[inference_isolate_handler] Could not preview tokens: $e');
      }

      final batch = bindings.llama_batch_get_one(tokensPtr, nTokens);
      if (bindings.llama_decode(ctx, batch) != 0) {
        calloc.free(tokensPtr);
        mainSendPort.send(
          _IsolateResponse(
            requestId: request.requestId,
            payload: InferenceError('Failed to evaluate prompt'),
            isComplete: true,
          ),
        );
        return;
      }

      final sampler = _configureSampler(
        bindings,
        request.options,
        bindings.llama_vocab_n_tokens(bindings.llama_model_get_vocab(model)),
      );

      // Caller-supplied stops plus the markers detected from the rendered
      // template. See `resolveStopTokens` for why detection is additive.
      final effectiveStopTokens = resolveStopTokens(
        requested: request.stopTokens,
        prompt: prompt,
        onDiagnostic: (message) {
          // ignore: avoid_print
          print('[inference_isolate_handler] $message');
        },
      );

      final generatedTokens = _generateTokens(
        bindings,
        model,
        ctx,
        sampler,
        vocab,
        request.options,
        effectiveStopTokens,
        request.requestId,
        mainSendPort,
      );

      bindings.llama_sampler_free(sampler);
      calloc.free(tokensPtr);

      mainSendPort.send(
        _IsolateResponse(
          requestId: request.requestId,
          payload: InferenceComplete(
            promptTokens: nTokens,
            generatedTokens: generatedTokens,
            promptFormatter: promptFormatter,
          ),
          isComplete: true,
        ),
      );
    } finally {
      if (loraAdapter != null) {
        clearContextLoraAdapters(bindings, ctx);
        bindings.llama_adapter_lora_free(loraAdapter);
      }
      bindings.llama_free(ctx);
      bindings.llama_model_free(model);
    }
  } catch (e) {
    mainSendPort.send(
      _IsolateResponse(
        requestId: request.requestId,
        payload: InferenceError(e.toString()),
        isComplete: true,
      ),
    );
  }
}
