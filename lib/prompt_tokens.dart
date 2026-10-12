// ignore_for_file: implementation_imports

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:llm_llamacpp/src/backend_initializer.dart';
import 'package:llm_llamacpp/src/bindings/llama_bindings.dart';

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
int? countPromptTokens({
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
    final modelPtr = model.cast<llama_model>();
    final vocabPtr = vocab.cast<llama_vocab>();

    final chatMessages = calloc<llama_chat_message>(messages.length);
    final allocated = <ffi.Pointer<Utf8>>[];
    try {
      for (var i = 0; i < messages.length; i++) {
        final rolePtr = messages[i].role.name.toNativeUtf8();
        final contentPtr = (messages[i].content ?? '').toNativeUtf8();
        allocated.add(rolePtr);
        allocated.add(contentPtr);
        chatMessages[i].role = rolePtr.cast();
        chatMessages[i].content = contentPtr.cast();
      }

      final tmplPtr = resolvedBindings.llama_model_chat_template(modelPtr, ffi.nullptr);
      final hasTemplate = tmplPtr.address != 0;
      final tmplArg = hasTemplate ? tmplPtr : ffi.nullptr;

      String? modelTemplateStr;
      if (hasTemplate) {
        try {
          modelTemplateStr = tmplPtr.cast<Utf8>().toDartString();
        } catch (_) {
          modelTemplateStr = null;
        }
      }

      final reqSize = resolvedBindings.llama_chat_apply_template(
        tmplArg,
        chatMessages,
        messages.length,
        true,
        ffi.nullptr,
        0,
      );

      String formatted;
      if (reqSize <= 0) {
        final buffer = StringBuffer();
        for (final msg in messages) {
          buffer.writeln('<|im_start|>${msg.role.name}');
          buffer.writeln(msg.content ?? '');
          buffer.writeln('<|im_end|>');
        }
        buffer.write('<|im_start|>assistant\n');
        formatted = buffer.toString();
      } else {
        final buf = calloc<ffi.Char>(reqSize + 1);
        try {
          final actualSize = resolvedBindings.llama_chat_apply_template(
            tmplArg,
            chatMessages,
            messages.length,
            true,
            buf,
            reqSize + 1,
          );
          if (actualSize <= 0) {
            final buffer = StringBuffer();
            for (final msg in messages) {
              buffer.writeln('<|im_start|>${msg.role.name}');
              buffer.writeln(msg.content ?? '');
              buffer.writeln('<|im_end|>');
            }
            buffer.write('<|im_start|>assistant\n');
            formatted = buffer.toString();
          } else {
            formatted = buf.cast<Utf8>().toDartString(length: actualSize);
          }
        } finally {
          calloc.free(buf);
        }
      }

      final addBosByTokenizer = resolvedBindings.llama_vocab_get_add_bos(vocabPtr);
      final templateRefersToBos = modelTemplateStr != null &&
          (modelTemplateStr.contains('bos_token') ||
              modelTemplateStr.contains('<|begin_of_text|>') ||
              modelTemplateStr.contains('<|startoftext|>'));
      final shouldManuallyPrependBos = templateRefersToBos && !addBosByTokenizer;

      final promptPtr = formatted.toNativeUtf8();
      final promptByteLen = promptPtr.length;
      final maxTokens = promptByteLen + 256;
      final tokBuf = calloc<ffi.Int32>(maxTokens);
      try {
        final nTok = resolvedBindings.llama_tokenize(
          vocabPtr,
          promptPtr.cast(),
          promptByteLen,
          tokBuf,
          maxTokens,
          !shouldManuallyPrependBos,
          true,
        );
        if (nTok < 0) return null;
        return shouldManuallyPrependBos ? nTok + 1 : nTok;
      } finally {
        calloc.free(promptPtr);
        calloc.free(tokBuf);
      }
    } finally {
      for (final p in allocated) {
        calloc.free(p);
      }
      calloc.free(chatMessages);
    }
  } catch (_) {
    return null;
  }
}

