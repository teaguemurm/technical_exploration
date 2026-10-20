part of 'persistent_inference_isolate.dart';

FormattedPrompt _applyNativeChatTemplate(
  LlamaBindings bindings,
  ffi.Pointer<llama_model> model,
  List<IsolateMessage> messages, {
  List<String> toolSchemasJson = const [],
}) {
  // Tool definitions are written into the system message before formatting.
  // The C chat-template API takes only role/content pairs, so a template's
  // `{%- if tools -%}` branch can never fire from here.
  final templateForTools = readModelChatTemplate(bindings, model);
  final toolFormat = _detectToolCallFormat(bindings, model, templateForTools);
  messages = injectToolDefinitions(
    messages,
    toolSchemasJson,
    format: toolFormat,
  );
  if (toolSchemasJson.isNotEmpty) {
    // ignore: avoid_print
    print(
      '[native_template_applier] Advertised ${toolSchemasJson.length} tool(s) '
      'as ${toolFormat.name}',
    );
  }

  return formatWithModelTemplate(
    bindings: bindings,
    model: model,
    messages: [
      for (final message in messages)
        ChatTurn(role: message.role, content: message.content),
    ],
  );
}
