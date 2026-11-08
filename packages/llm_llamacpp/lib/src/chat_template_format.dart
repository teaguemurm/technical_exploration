/// Chat-template decisions that do not need the native library.
///
/// llama.cpp's `llama_chat_apply_template` only knows a fixed list of
/// templates. Gemma 4's Jinja uses `<|turn>` / `<turn|>`, which that function
/// rejects. The decision here is shared by the inference isolate and the
/// prompt-token counter so both format the same text.
library;

/// `llama_chat_apply_template` succeeded.
const String promptFormatterNative = 'native';

/// Dart renderer for a Gemma 4 template. Thinking is left off.
const String promptFormatterGemma4 = 'gemma4';

/// ChatML text used only when the embedded template itself is ChatML and the
/// native apply still failed. Never used for another model family.
const String promptFormatterChatMlFallback = 'chatml-fallback';

/// End-of-turn strings Gemma 4 may emit. `<turn|>` is the marker this prompt
/// asks for. Some checkpoints still emit the older `<end_of_turn>` or `<eos>`,
/// and waiting only for `<turn|>` runs the answer out to the token cap.
const List<String> gemma4GenerationStopMarkers = <String>[
  '<turn|>',
  '<end_of_turn>',
  '<eos>',
];

/// The generation prompt ends on this newline. It does not close the turn.
const String gemma4GenerationSuffix = '<|turn>model\n';

/// Thrown when a chat template cannot be applied without substituting another
/// model family's format, or when Gemma 4 markers do not tokenize as specials.
class ChatTemplateException implements Exception {
  ChatTemplateException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Status line when the counter and the isolate did not use the same formatter.
///
/// Both names are in the string so a mismatch can be diagnosed from the UI.
String promptFormatterMismatchMessage({
  required String preflight,
  required String sent,
}) {
  return 'Prompt formatter mismatch: preflight=$preflight, sent=$sent';
}

/// One chat message after tool injection, before family-specific rendering.
class ChatTurn {
  const ChatTurn({required this.role, required this.content});

  final String role;
  final String content;
}

/// The rendered prompt and which formatter produced it.
class FormattedPrompt {
  const FormattedPrompt({
    required this.text,
    required this.formatter,
    required this.bosIncludedInText,
    required this.template,
  });

  final String text;
  final String formatter;

  /// True when [text] already contains the BOS special, so the tokenizer must
  /// not add a second one.
  final bool bosIncludedInText;

  /// Embedded GGUF template, when it decoded. Used for the non-Gemma BOS
  /// heuristic. Not interpreted as Jinja.
  final String? template;
}

/// True when the embedded template uses Gemma 4's turn markers.
///
/// Both markers are required. Older Gemma `<start_of_turn>` templates stay on
/// the native applier. The architecture name is not consulted.
bool isGemma4Template(String? template) {
  if (template == null || template.isEmpty) return false;
  return template.contains('<|turn>') && template.contains('<turn|>');
}

/// True when the template is ChatML, so a failed native apply may use the
/// ChatML fallback without changing family.
bool isChatMlTemplate(String? template) {
  if (template == null) return false;
  final trimmed = template.trim();
  if (trimmed.isEmpty) return false;
  if (trimmed == 'chatml') return true;
  return template.contains('<|im_start|>');
}

/// Renders Gemma 4 turns with thinking off.
///
/// `<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n`
/// and the same shape for every further message, then `<|turn>model\n`.
/// System stays its own turn. Nothing is inserted between content and
/// `<turn|>`. The result ends on the generation suffix, not on `<turn|>`.
String formatGemma4(List<ChatTurn> messages) {
  final buffer = StringBuffer('<bos>');
  for (final message in messages) {
    buffer
      ..write('<|turn>')
      ..write(_gemmaRole(message.role))
      ..write('\n')
      ..write(message.content)
      ..write('<turn|>\n');
  }
  buffer.write(gemma4GenerationSuffix);
  return buffer.toString();
}

String _gemmaRole(String role) {
  switch (role) {
    case 'system':
      return 'system';
    case 'user':
      return 'user';
    case 'assistant':
    case 'model':
      return 'model';
    case 'tool':
      throw ChatTemplateException(
        'Gemma 4 formatter cannot render a tool turn.',
      );
    default:
      throw ChatTemplateException(
        'Gemma 4 formatter does not know role "$role".',
      );
  }
}

/// Byte-compatible with the previous internal ChatML fallback.
String formatChatMlFallback(List<ChatTurn> messages) {
  final buffer = StringBuffer();
  for (final message in messages) {
    buffer
      ..writeln('<|im_start|>${message.role}')
      ..writeln(message.content)
      ..writeln('<|im_end|>');
  }
  buffer.write('<|im_start|>assistant\n');
  return buffer.toString();
}

/// Picks the formatter for a decoded template.
///
/// A missing template throws. A Gemma 4 template never uses [nativeText] or
/// the ChatML fallback. A successful native render is returned untouched.
FormattedPrompt applyChatTemplateDecision({
  required String? template,
  required List<ChatTurn> messages,
  required String? nativeText,
}) {
  if (template == null || template.isEmpty) {
    throw ChatTemplateException(
      'Model has no embedded chat template. Refusing to substitute ChatML.',
    );
  }
  if (isGemma4Template(template)) {
    return FormattedPrompt(
      text: formatGemma4(messages),
      formatter: promptFormatterGemma4,
      bosIncludedInText: true,
      template: template,
    );
  }
  if (nativeText != null) {
    return FormattedPrompt(
      text: nativeText,
      formatter: promptFormatterNative,
      bosIncludedInText: false,
      template: template,
    );
  }
  if (isChatMlTemplate(template)) {
    return FormattedPrompt(
      text: formatChatMlFallback(messages),
      formatter: promptFormatterChatMlFallback,
      bosIncludedInText: false,
      template: template,
    );
  }
  throw ChatTemplateException(
    'Chat template could not be applied and is not Gemma 4 or ChatML. '
    'Refusing to substitute another format.',
  );
}
