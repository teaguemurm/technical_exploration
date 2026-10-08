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

/// The three prompt conditions defined in the Technical Exploration Project Plan.
enum PromptCondition {
  /// Condition A: System prompt only. Dataset is not used.
  /// Tests how much the base model already knows, and what kind of output it defaults to.
  conditionA,

  /// Condition B: Include data retrieval (simulated RAG or all notes).
  /// Tests whether output can be grounded using the supplied facts.
  conditionB,

  /// Condition C: Specific output constraints and grounding boundaries on top of B.
  /// Tests whether form can be constrained and whether the model says it doesn't
  /// know / says nothing when the dataset does not cover the query.
  conditionC,
}

extension PromptConditionExt on PromptCondition {
  String get code => switch (this) {
    PromptCondition.conditionA => 'Condition A',
    PromptCondition.conditionB => 'Condition B',
    PromptCondition.conditionC => 'Condition C',
  };

  String get label => switch (this) {
    PromptCondition.conditionA => 'Condition A (Base only)',
    PromptCondition.conditionB => 'Condition B (Dataset)',
    PromptCondition.conditionC => 'Condition C (Constraints)',
  };

  String get shortDescription => switch (this) {
    PromptCondition.conditionA => 'System prompt only. No dataset.',
    PromptCondition.conditionB => 'Grounded with local notes / simulated RAG.',
    PromptCondition.conditionC => 'Strict grounding, 1-3 sentences, and explicit refusal if uncovered.',
  };
}

class PromptSpec {
  const PromptSpec({
    required this.id,
    required this.label,
    required this.text,
    this.relevantFactIds = const [],
  });

  final String id;
  final String label;
  final String text;
  final List<String> relevantFactIds;
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
  final relevantList = value['relevantFactIds'];
  final List<String> relevantFactIds = relevantList is List
      ? relevantList.map((e) => e.toString()).toList()
      : const [];
  return PromptSpec(
    id: _requiredString(value, 'id', 'prompts.json[$index]'),
    label: _requiredString(value, 'label', 'prompts.json[$index]'),
    text: _requiredString(value, 'text', 'prompts.json[$index]'),
    relevantFactIds: relevantFactIds,
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

/// Standard refusal statement expected when Condition C detects uncovered knowledge.
const standardRefusalStatement =
    'I do not have information about that in the approved notes.';

/// Selects facts for Condition B or C.
/// When [relevantOnly] is true, uses prompt.relevantFactIds (or keyword matching fallback).
/// When [relevantOnly] is false, returns all facts.
List<DatasetFact> selectFacts(
  PromptSpec prompt,
  List<DatasetFact> allFacts, {
  required bool relevantOnly,
}) {
  if (!relevantOnly) {
    return allFacts;
  }
  if (prompt.relevantFactIds.isNotEmpty) {
    final targetIds = prompt.relevantFactIds.toSet();
    return allFacts.where((fact) => targetIds.contains(fact.id)).toList();
  }
  // If no explicit relevant IDs are set, do basic keyword matching
  final words = prompt.text
      .toLowerCase()
      .split(RegExp(r'\W+'))
      .where((w) => w.length >= 4)
      .toSet();
  final matched = allFacts.where((fact) {
    final factLower = fact.text.toLowerCase();
    return words.any((word) => factLower.contains(word));
  }).toList();
  return matched;
}

/// Builds the system prompt for Condition C, incorporating grounding refusal
/// boundaries and concise output format constraints.
String buildConditionCSystemPrompt(String baseSystemPrompt) {
  return '${trimSingleTrailingNewline(baseSystemPrompt)}\n\n'
      'STRICT GROUNDING & FORM CONSTRAINTS:\n'
      '1. Answer strictly and solely based on the Local notes provided below.\n'
      '2. If the Local notes do not explicitly provide the information needed to answer the question, '
      'you MUST reply with: "$standardRefusalStatement" Do not guess, extrapolate, or invent details.\n'
      '3. Limit your response to 1 to 3 concise, factual sentences. '
      'Do not include conversational pleasantries, introductory phrases, or meta-commentary.';
}

/// Builds the LLM messages for any of the three experimental conditions.
List<LLMMessage> buildConditionMessages({
  required PromptCondition condition,
  required String systemPrompt,
  required PromptSpec prompt,
  required List<DatasetFact> allFacts,
  bool relevantOnly = true,
}) {
  switch (condition) {
    case PromptCondition.conditionA:
      return [
        LLMMessage(
          role: LLMRole.system,
          content: trimSingleTrailingNewline(systemPrompt),
        ),
        LLMMessage(role: LLMRole.user, content: prompt.text),
      ];

    case PromptCondition.conditionB:
      final facts = selectFacts(prompt, allFacts, relevantOnly: relevantOnly);
      final messages = <LLMMessage>[
        LLMMessage(
          role: LLMRole.system,
          content: trimSingleTrailingNewline(systemPrompt),
        ),
      ];
      if (facts.isNotEmpty) {
        messages.add(
          LLMMessage(
            role: LLMRole.user,
            content: localNotesMessage(facts.map((f) => f.text).toList()),
          ),
        );
      } else {
        messages.add(
          LLMMessage(
            role: LLMRole.user,
            content: '$localNotesTitle\n\n$localNotesIntro\n(No specific local notes match this query.)',
          ),
        );
      }
      messages.add(LLMMessage(role: LLMRole.user, content: prompt.text));
      return messages;

    case PromptCondition.conditionC:
      final facts = selectFacts(prompt, allFacts, relevantOnly: relevantOnly);
      final messages = <LLMMessage>[
        LLMMessage(
          role: LLMRole.system,
          content: buildConditionCSystemPrompt(systemPrompt),
        ),
      ];
      if (facts.isNotEmpty) {
        messages.add(
          LLMMessage(
            role: LLMRole.user,
            content: localNotesMessage(facts.map((f) => f.text).toList()),
          ),
        );
      } else {
        messages.add(
          LLMMessage(
            role: LLMRole.user,
            content: '$localNotesTitle\n\n$localNotesIntro\n(No specific local notes match this query.)',
          ),
        );
      }
      messages.add(LLMMessage(role: LLMRole.user, content: prompt.text));
      return messages;
  }
}

/// Legacy message builder for backward compatibility.
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

/// Checks if an output text matches the expected refusal pattern in Condition C.
bool isRefusalOutput(String output) {
  final lower = output.toLowerCase();
  return lower.contains('do not have information') ||
      lower.contains('approved notes') ||
      lower.contains('no information') ||
      lower.contains('not mention') ||
      lower.contains('notes do not provide');
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
