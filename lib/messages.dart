import 'dart:convert';

import 'package:llm_llamacpp/llm_llamacpp.dart';

/// Drops one trailing newline. Other whitespace stays, so an edited asset
/// keeps the spacing its author typed.
String trimSingleTrailingNewline(String raw) {
  if (raw.endsWith('\r\n')) {
    return raw.substring(0, raw.length - 2);
  }
  if (raw.endsWith('\n')) {
    return raw.substring(0, raw.length - 1);
  }
  return raw;
}

class PromptSpec {
  const PromptSpec({
    required this.id,
    required this.label,
    required this.text,
  });

  final String id;
  final String label;
  final String text;
}

class DatasetFact {
  const DatasetFact({required this.id, required this.text});

  final String id;
  final String text;
}

class CatalogData {
  const CatalogData({
    required this.systemPrompt,
    required this.prompts,
    required this.facts,
  });

  final String systemPrompt;
  final List<PromptSpec> prompts;
  final List<DatasetFact> facts;
}

CatalogData parseCatalog({
  required String systemPrompt,
  required String promptsJson,
  required String datasetJson,
}) {
  return CatalogData(
    systemPrompt: trimSingleTrailingNewline(systemPrompt),
    prompts: parsePrompts(promptsJson),
    facts: parseDataset(datasetJson),
  );
}

List<PromptSpec> parsePrompts(String jsonText) {
  final decoded = jsonDecode(jsonText);
  if (decoded is! List) {
    throw const FormatException('prompts.json must be a list');
  }
  return [
    for (var i = 0; i < decoded.length; i++)
      _promptAt(decoded[i], i),
  ];
}

List<DatasetFact> parseDataset(String jsonText) {
  final decoded = jsonDecode(jsonText);
  if (decoded is! List) {
    throw const FormatException('dataset.json must be a list');
  }
  return [
    for (var i = 0; i < decoded.length; i++)
      _factAt(decoded[i], i),
  ];
}

PromptSpec _promptAt(Object? value, int index) {
  if (value is! Map) {
    throw FormatException('prompts.json[$index] must be an object');
  }
  return PromptSpec(
    id: _requiredString(value, 'id', 'prompts.json[$index]'),
    label: _requiredString(value, 'label', 'prompts.json[$index]'),
    text: _requiredString(value, 'text', 'prompts.json[$index]'),
  );
}

DatasetFact _factAt(Object? value, int index) {
  if (value is! Map) {
    throw FormatException('dataset.json[$index] must be an object');
  }
  return DatasetFact(
    id: _requiredString(value, 'id', 'dataset.json[$index]'),
    text: _requiredString(value, 'text', 'dataset.json[$index]'),
  );
}

String _requiredString(Map<dynamic, dynamic> value, String key, String where) {
  final field = value[key];
  if (field is! String || field.isEmpty) {
    throw FormatException('$where.$key must be a non-empty string');
  }
  return field;
}

const localNotesTitle = 'Local notes';
const localNotesIntro =
    'Local notes follow. They are the only place-specific notes supplied for this run.';

/// System prompt, then the selected prompt. With [useDataset], one user
/// message listing every fact is inserted before the selected prompt.
List<LLMMessage> buildMessages({
  required String systemPrompt,
  required String promptText,
  required bool useDataset,
  required List<String> datasetLines,
}) {
  final messages = <LLMMessage>[
    LLMMessage(
      role: LLMRole.system,
      content: trimSingleTrailingNewline(systemPrompt),
    ),
  ];
  if (useDataset) {
    messages.add(
      LLMMessage(role: LLMRole.user, content: localNotesMessage(datasetLines)),
    );
  }
  messages.add(LLMMessage(role: LLMRole.user, content: promptText));
  return messages;
}

String localNotesMessage(List<String> datasetLines) {
  return [
    localNotesTitle,
    '',
    localNotesIntro,
    ...datasetLines,
  ].join('\n');
}

String formatMessages(List<LLMMessage> messages) {
  return messages
      .map((message) => '${message.role.name}\n${message.content ?? ''}')
      .join('\n\n');
}

/// Generated tokens divided by the time from the first content chunk to the
/// final chunk. Null when the library did not report [evalCount], or when the
/// measured interval is empty so a rate would not be finite.
double? tokensPerSecond({
  required int? evalCount,
  required Duration generation,
}) {
  if (evalCount == null) {
    return null;
  }
  final seconds = generation.inMicroseconds / Duration.microsecondsPerSecond;
  if (seconds <= 0) {
    return null;
  }
  return evalCount / seconds;
}

class BackendMemory {
  const BackendMemory({required this.name, this.total, this.free});

  final String name;
  final int? total;
  final int? free;
}

final _memoryKey = RegExp(r'memory|rss|vram|\bram\b', caseSensitive: false);

/// Library-reported memory only. File size and parameter estimates are not
/// memory figures and are left out.
String? formatMemoryReport({
  required Map<String, dynamic> providerMetadata,
  required Iterable<BackendMemory> backends,
}) {
  final lines = <String>[];
  for (final entry in providerMetadata.entries) {
    if (entry.value is num && _memoryKey.hasMatch(entry.key)) {
      lines.add('${entry.key}: ${entry.value}');
    }
  }
  for (final backend in backends) {
    if (backend.total != null) {
      lines.add('${backend.name} memoryTotal: ${backend.total}');
    }
    if (backend.free != null) {
      lines.add('${backend.name} memoryFree: ${backend.free}');
    }
  }
  if (lines.isEmpty) {
    return null;
  }
  return lines.join('\n');
}
