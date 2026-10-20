import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:llm_llamacpp/src/bindings/llama_bindings.dart';
import 'package:llm_llamacpp/src/chat_template_format.dart';

export 'package:llm_llamacpp/src/chat_template_format.dart';

/// Reads the GGUF chat template. The pointer is owned by the model.
String? readModelChatTemplate(
  LlamaBindings bindings,
  ffi.Pointer<llama_model> model,
) {
  final templatePtr = bindings.llama_model_chat_template(model, ffi.nullptr);
  if (templatePtr.address == 0) return null;
  try {
    return templatePtr.cast<Utf8>().toDartString();
  } catch (_) {
    return null;
  }
}

/// Formats [messages] with the model's own template.
///
/// Gemma 4 is rendered in Dart. Every other recognised template goes through
/// `llama_chat_apply_template`. A template that cannot be applied throws
/// [ChatTemplateException] instead of silently writing another family's text.
FormattedPrompt formatWithModelTemplate({
  required LlamaBindings bindings,
  required ffi.Pointer<llama_model> model,
  required List<ChatTurn> messages,
}) {
  final templatePtr = bindings.llama_model_chat_template(model, ffi.nullptr);
  if (templatePtr.address == 0) {
    throw ChatTemplateException(
      'Model has no embedded chat template. Refusing to substitute ChatML.',
    );
  }

  String? template;
  try {
    template = templatePtr.cast<Utf8>().toDartString();
  } catch (_) {
    template = null;
  }

  if (isGemma4Template(template)) {
    final formatted = formatGemma4(messages);
    // ignore: avoid_print
    print(
      '[chat_template] formatter=$promptFormatterGemma4 '
      '(llama_chat_apply_template skipped) '
      '${_preview(formatted)}',
    );
    return FormattedPrompt(
      text: formatted,
      formatter: promptFormatterGemma4,
      bosIncludedInText: true,
      template: template,
    );
  }

  final nativeText = _tryNativeChatApply(bindings, templatePtr, messages);
  if (template == null) {
    if (nativeText != null) {
      // ignore: avoid_print
      print(
        '[chat_template] formatter=$promptFormatterNative '
        '(template bytes did not decode) ${_preview(nativeText)}',
      );
      return FormattedPrompt(
        text: nativeText,
        formatter: promptFormatterNative,
        bosIncludedInText: false,
        template: null,
      );
    }
    throw ChatTemplateException(
      'Chat template bytes could not be decoded and llama_chat_apply_template '
      'failed. Refusing to substitute ChatML.',
    );
  }

  final applied = applyChatTemplateDecision(
    template: template,
    messages: messages,
    nativeText: nativeText,
  );
  if (applied.formatter == promptFormatterChatMlFallback) {
    // ignore: avoid_print
    print(
      '[chat_template] formatter=$promptFormatterChatMlFallback '
      'llama_chat_apply_template failed for a ChatML template',
    );
  } else {
    // ignore: avoid_print
    print(
      '[chat_template] formatter=${applied.formatter} ${_preview(applied.text)}',
    );
  }
  return applied;
}

String _preview(String text) {
  final preview = text.length > 200 ? '${text.substring(0, 200)}...' : text;
  return '(${text.length} chars): $preview';
}

String? _tryNativeChatApply(
  LlamaBindings bindings,
  ffi.Pointer<ffi.Char> templatePtr,
  List<ChatTurn> messages,
) {
  if (messages.isEmpty) return null;
  final chatMessages = calloc<llama_chat_message>(messages.length);
  final allocatedPointers = <ffi.Pointer<Utf8>>[];
  try {
    for (var i = 0; i < messages.length; i++) {
      final rolePtr = messages[i].role.toNativeUtf8();
      final contentPtr = messages[i].content.toNativeUtf8();
      allocatedPointers
        ..add(rolePtr)
        ..add(contentPtr);
      chatMessages[i].role = rolePtr.cast();
      chatMessages[i].content = contentPtr.cast();
    }

    final requiredSize = bindings.llama_chat_apply_template(
      templatePtr,
      chatMessages,
      messages.length,
      true,
      ffi.nullptr,
      0,
    );
    if (requiredSize <= 0) {
      // ignore: avoid_print
      print(
        '[chat_template] llama_chat_apply_template sizing call '
        'returned $requiredSize',
      );
      return null;
    }

    final buffer = calloc<ffi.Char>(requiredSize + 1);
    try {
      final actualSize = bindings.llama_chat_apply_template(
        templatePtr,
        chatMessages,
        messages.length,
        true,
        buffer,
        requiredSize + 1,
      );
      if (actualSize <= 0) {
        // ignore: avoid_print
        print(
          '[chat_template] llama_chat_apply_template render call '
          'returned $actualSize',
        );
        return null;
      }
      return buffer.cast<Utf8>().toDartString(length: actualSize);
    } finally {
      calloc.free(buffer);
    }
  } finally {
    for (final ptr in allocatedPointers) {
      calloc.free(ptr);
    }
    calloc.free(chatMessages);
  }
}

/// Token ids for [formatted], using the same BOS rule as generation.
///
/// Gemma 4 text already starts with `<bos>`. It is tokenized with specials
/// parsed and with the tokenizer's own BOS addition turned off, and the manual
/// BOS prepend used for other families is skipped. Each Gemma marker must be
/// one vocab token, and the prompt must contain that BOS id exactly once.
List<int> tokenizeFormattedPrompt({
  required LlamaBindings bindings,
  required ffi.Pointer<llama_vocab> vocab,
  required FormattedPrompt formatted,
  required bool usingChatTemplate,
}) {
  if (formatted.formatter == promptFormatterGemma4) {
    _assertGemma4MarkersAreSingleTokens(bindings, vocab);
  }

  final addBosByTokenizer = bindings.llama_vocab_get_add_bos(vocab);
  final template = formatted.template;
  final templateRefersToBos =
      template != null &&
      (template.contains('bos_token') ||
          template.contains('<|begin_of_text|>') ||
          template.contains('<|startoftext|>'));
  final shouldManuallyPrependBos =
      !formatted.bosIncludedInText &&
      usingChatTemplate &&
      templateRefersToBos &&
      !addBosByTokenizer;
  final addSpecial = !shouldManuallyPrependBos && !formatted.bosIncludedInText;
  // ignore: avoid_print
  print(
    '[chat_template] BOS handling: '
    'formatter=${formatted.formatter} '
    'add_bos_token=$addBosByTokenizer '
    'template_refers_to_bos=$templateRefersToBos '
    'bos_in_text=${formatted.bosIncludedInText} '
    'manual_bos_prepend=$shouldManuallyPrependBos '
    'add_special=$addSpecial',
  );

  final ids = _tokenize(
    bindings,
    vocab,
    formatted.text,
    addSpecial: addSpecial,
    parseSpecial: true,
  );
  if (shouldManuallyPrependBos) {
    final bosId = bindings.llama_vocab_bos(vocab);
    if (bosId >= 0) {
      ids.insert(0, bosId);
      // ignore: avoid_print
      print(
        '[chat_template] Manually prepended BOS token id=$bosId; '
        'prompt now ${ids.length} tokens',
      );
    } else {
      // ignore: avoid_print
      print(
        '[chat_template] WARNING: wanted to prepend BOS but '
        'llama_vocab_bos returned $bosId',
      );
    }
  }

  if (formatted.formatter == promptFormatterGemma4) {
    final bosId = bindings.llama_vocab_bos(vocab);
    final bosCount = ids.where((id) => id == bosId).length;
    if (bosId < 0 || ids.isEmpty || ids.first != bosId || bosCount != 1) {
      throw ChatTemplateException(
        'Gemma 4 prompt must start with a single BOS token '
        '(bos=$bosId, first=${ids.isEmpty ? 'none' : ids.first}, '
        'count=$bosCount).',
      );
    }
  }
  return ids;
}

void _assertGemma4MarkersAreSingleTokens(
  LlamaBindings bindings,
  ffi.Pointer<llama_vocab> vocab,
) {
  const markers = <String>['<bos>', '<|turn>', '<turn|>'];
  for (final marker in markers) {
    final ids = _tokenize(
      bindings,
      vocab,
      marker,
      addSpecial: false,
      parseSpecial: true,
    );
    if (ids.length != 1) {
      throw ChatTemplateException(
        'Gemma 4 marker $marker tokenized as ${ids.length} tokens, '
        'not one special token.',
      );
    }
  }
}

List<int> _tokenize(
  LlamaBindings bindings,
  ffi.Pointer<llama_vocab> vocab,
  String text, {
  required bool addSpecial,
  required bool parseSpecial,
}) {
  final promptPtr = text.toNativeUtf8();
  final promptByteLen = promptPtr.length;
  final maxTokens = promptByteLen + 256;
  final tokBuf = calloc<ffi.Int32>(maxTokens);
  try {
    final nTok = bindings.llama_tokenize(
      vocab,
      promptPtr.cast(),
      promptByteLen,
      tokBuf,
      maxTokens,
      addSpecial,
      parseSpecial,
    );
    if (nTok < 0) {
      throw ChatTemplateException(
        'Failed to tokenize prompt (llama_tokenize returned $nTok).',
      );
    }
    return [for (var i = 0; i < nTok; i++) tokBuf[i]];
  } finally {
    calloc.free(promptPtr);
    calloc.free(tokBuf);
  }
}
