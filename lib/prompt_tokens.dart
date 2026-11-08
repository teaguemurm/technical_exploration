// ignore_for_file: implementation_imports

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:llm_llamacpp/src/backend_initializer.dart';
import 'package:llm_llamacpp/src/bindings/llama_bindings.dart';
import 'package:llm_llamacpp/src/chat_template_apply.dart';

/// Prompt text that was actually formatted, plus the formatter name.
class CountedPrompt {
  const CountedPrompt({required this.tokenCount, required this.formatter});

  final int tokenCount;

  /// `native`, `gemma4`, or `chatml-fallback`.
  final String formatter;
}

LlamaBindings? _cachedBindings;

/// Helper to get or initialize [LlamaBindings] safely across both app and test environments.
LlamaBindings? resolveLlamaBindings([LlamaBindings? existing]) {
  if (existing != null) return existing;
  if (_cachedBindings != null) return _cachedBindings;
  try {
    _cachedBindings = BackendInitializer.initializeBackend().$2;
    return _cachedBindings;
  } catch (_) {
    try {
      const fallbackPath = 'build/native_assets/macos/libllama.dylib';
      if (File(fallbackPath).existsSync()) {
        final lib = ffi.DynamicLibrary.open(File(fallbackPath).absolute.path);
        _cachedBindings = LlamaBindings(lib);
        return _cachedBindings;
      }
    } catch (_) {}
    return null;
  }
}

/// Computes the exact prompt token count for the given [messages] by applying
/// the model's chat template and tokenizing the full formatted prompt with the model's vocabulary.
///
/// Uses the same formatter and BOS rule as the inference isolate. A template
/// that cannot be applied throws [ChatTemplateException] instead of returning
/// null, so the caller does not fall through to a length estimate.
CountedPrompt? countPromptTokens({
  required ffi.Pointer<dynamic>? model,
  required ffi.Pointer<dynamic>? vocab,
  required List<LLMMessage> messages,
  LlamaBindings? bindings,
}) {
  if (model == null || vocab == null || messages.isEmpty) {
    return null;
  }
  if (model.address == 0 || vocab.address == 0) {
    return null;
  }

  final resolvedBindings = bindings ?? resolveLlamaBindings();
  if (resolvedBindings == null) return null;

  try {
    final formatted = formatWithModelTemplate(
      bindings: resolvedBindings,
      model: model.cast<llama_model>(),
      messages: [
        for (final message in messages)
          ChatTurn(role: message.role.name, content: message.content ?? ''),
      ],
    );
    final tokenIds = tokenizeFormattedPrompt(
      bindings: resolvedBindings,
      vocab: vocab.cast<llama_vocab>(),
      formatted: formatted,
      usingChatTemplate: true,
    );
    return CountedPrompt(
      tokenCount: tokenIds.length,
      formatter: formatted.formatter,
    );
  } on ChatTemplateException {
    rethrow;
  } catch (_) {
    return null;
  }
}

