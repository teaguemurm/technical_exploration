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

/// The prompt conditions defined for the empirical exploration protocol.
enum PromptCondition {
  /// Condition A: System prompt only. Dataset is not used.
  conditionA,

  /// Condition B: All 10 dataset notes injected into the prompt.
  conditionB,

  /// Condition C: All 10 dataset notes + output constraints + refusal rule + bans.
  conditionC,

  /// Condition B-sim-RAG: Fixed note selection (k=2) approximation.
  conditionBSimRag,

  /// Condition C-sim-RAG: Fixed note selection (k=2) + output constraints + refusal rule + bans.
  conditionCSimRag,
}

extension PromptConditionExt on PromptCondition {
  String get code => switch (this) {
    PromptCondition.conditionA => 'Condition A',
    PromptCondition.conditionB => 'Condition B',
    PromptCondition.conditionC => 'Condition C',
    PromptCondition.conditionBSimRag => 'Condition B-sim-RAG',
    PromptCondition.conditionCSimRag => 'Condition C-sim-RAG',
  };

  String get shortCode => switch (this) {
    PromptCondition.conditionA => 'A',
    PromptCondition.conditionB => 'B',
    PromptCondition.conditionC => 'C',
    PromptCondition.conditionBSimRag => 'B-sim-RAG',
    PromptCondition.conditionCSimRag => 'C-sim-RAG',
  };

  String get label => switch (this) {
    PromptCondition.conditionA => 'Condition A (Base only)',
    PromptCondition.conditionB => 'Condition B (Dataset)',
    PromptCondition.conditionC => 'Condition C (Constraints)',
    PromptCondition.conditionBSimRag => 'Condition B-sim-RAG (Simulated RAG)',
    PromptCondition.conditionCSimRag => 'Condition C-sim-RAG (Simulated RAG + Constraints)',
  };

  String get notesMode => switch (this) {
    PromptCondition.conditionA => 'none',
    PromptCondition.conditionB => 'all',
    PromptCondition.conditionC => 'all',
    PromptCondition.conditionBSimRag => 'sim-rag',
    PromptCondition.conditionCSimRag => 'sim-rag',
  };

  bool get isSimulatedRag =>
      this == PromptCondition.conditionBSimRag ||
      this == PromptCondition.conditionCSimRag;

  bool get isConditionC =>
      this == PromptCondition.conditionC ||
      this == PromptCondition.conditionCSimRag;

  String get shortDescription => switch (this) {
    PromptCondition.conditionA => 'System prompt only. No dataset.',
    PromptCondition.conditionB => 'All 10 notes included. No abstention hints.',
    PromptCondition.conditionC => 'All 10 notes + strict refusal, 1-3 sentences, and behavioural bans.',
    PromptCondition.conditionBSimRag => 'Fixed top-k notes (retrieval approximation). No abstention hints.',
    PromptCondition.conditionCSimRag => 'Fixed top-k notes + strict refusal, 1-3 sentences, and behavioural bans.',
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

class SimulatedRagConfig {
  const SimulatedRagConfig({
    this.k = 2,
    this.retrieval = const {},
  });

  final int k;
  final Map<String, List<String>> retrieval;

  factory SimulatedRagConfig.fromJson(Map<String, dynamic> json) {
    final k = (json['k'] as num?)?.toInt() ?? 2;
    final retMap = <String, List<String>>{};
    final rawRetrieval = json['retrieval'];
    if (rawRetrieval is Map) {
      for (final entry in rawRetrieval.entries) {
        if (entry.value is List) {
          retMap[entry.key.toString()] =
              (entry.value as List).map((e) => e.toString()).toList();
        }
      }
    }
    return SimulatedRagConfig(k: k, retrieval: retMap);
  }
}

class CatalogData {
  const CatalogData({
    required this.systemPrompt,
    required this.prompts,
    required this.facts,
    required this.simRagConfig,
  });

  final String systemPrompt;
  final List<PromptSpec> prompts;
  final List<DatasetFact> facts;
  final SimulatedRagConfig simRagConfig;
}

CatalogData parseCatalog({
  required String systemPrompt,
  required String promptsJson,
  required String datasetJson,
  String? simRagJson,
}) {
  final simRag = simRagJson != null && simRagJson.isNotEmpty
      ? SimulatedRagConfig.fromJson(jsonDecode(simRagJson) as Map<String, dynamic>)
      : const SimulatedRagConfig();

  return CatalogData(
    systemPrompt: trimSingleTrailingNewline(systemPrompt),
    prompts: parsePrompts(promptsJson),
    facts: parseDataset(datasetJson),
    simRagConfig: simRag,
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

/// Selects facts for a condition according to the locked protocol rules:
/// - Condition A: 0 notes
/// - Condition B & C: All 10 notes
/// - Condition B-sim-RAG & C-sim-RAG: Fixed list from [SimulatedRagConfig] ($k=2$).
///   No keyword-matching fallback: if no list exists, returns 0 notes and logs.
List<DatasetFact> selectFactsForCondition({
  required PromptCondition condition,
  required PromptSpec prompt,
  required List<DatasetFact> allFacts,
  required SimulatedRagConfig simRagConfig,
}) {
  switch (condition) {
    case PromptCondition.conditionA:
      return const [];

    case PromptCondition.conditionB:
    case PromptCondition.conditionC:
      // B and C always get all 10 notes as locked
      return allFacts;

    case PromptCondition.conditionBSimRag:
    case PromptCondition.conditionCSimRag:
      final noteIds = simRagConfig.retrieval[prompt.id];
      if (noteIds == null || noteIds.isEmpty) {
        // No word-matching fallback: a prompt without a list gets no notes, and this is logged.
        // ignore: avoid_print
        print('[sim-RAG] Prompt "${prompt.id}" has no fixed note list; injecting 0 notes.');
        return const [];
      }
      final factMap = {for (final f in allFacts) f.id: f};
      final selected = <DatasetFact>[];
      for (final id in noteIds.take(simRagConfig.k)) {
        final fact = factMap[id];
        if (fact != null) {
          selected.add(fact);
        }
      }
      return selected;
  }
}

/// Backward compatibility helper for legacy calls.
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
  return const [];
}

/// Builds the system prompt for Condition C, incorporating grounding refusal
/// boundaries, concise output format constraints, and behavioural bans.
String buildConditionCSystemPrompt(String baseSystemPrompt) {
  return '${trimSingleTrailingNewline(baseSystemPrompt)}\n\n'
      'STRICT GROUNDING & FORM CONSTRAINTS:\n'
      '1. Answer strictly and solely based on the Local notes provided below.\n'
      '2. If the Local notes do not explicitly provide the information needed to answer the question, '
      'you MUST reply with: "$standardRefusalStatement" Do not guess, extrapolate, or invent details.\n'
      '3. Limit your response to 1 to 3 concise, factual sentences. '
      'Do not include conversational pleasantries, introductory phrases, or meta-commentary.\n'
      '4. Do NOT speak as Nature or in the animal\'s first-person voice.\n'
      '5. Do NOT claim to decode or translate the animal\'s behaviour.\n'
      '6. Do NOT preach or moralise at the visitor.';
}

/// Builds the LLM messages for any experimental condition.
/// Ensures B variants never hint at abstaining.
List<LLMMessage> buildConditionMessages({
  required PromptCondition condition,
  required String systemPrompt,
  required PromptSpec prompt,
  required List<DatasetFact> allFacts,
  SimulatedRagConfig simRagConfig = const SimulatedRagConfig(),
}) {
  final userText = prompt.text;

  switch (condition) {
    case PromptCondition.conditionA:
      return [
        LLMMessage(
          role: LLMRole.system,
          content: trimSingleTrailingNewline(systemPrompt),
        ),
        LLMMessage(role: LLMRole.user, content: userText),
      ];

    case PromptCondition.conditionB:
    case PromptCondition.conditionBSimRag:
      final facts = selectFactsForCondition(
        condition: condition,
        prompt: prompt,
        allFacts: allFacts,
        simRagConfig: simRagConfig,
      );
      final messages = <LLMMessage>[
        LLMMessage(
          role: LLMRole.system,
          content: trimSingleTrailingNewline(systemPrompt),
        ),
      ];
      // Nothing in the B variants hints at abstaining!
      if (facts.isNotEmpty) {
        messages.add(
          LLMMessage(
            role: LLMRole.user,
            content: localNotesMessage(facts.map((f) => f.text).toList()),
          ),
        );
      }
      messages.add(LLMMessage(role: LLMRole.user, content: userText));
      return messages;

    case PromptCondition.conditionC:
    case PromptCondition.conditionCSimRag:
      final facts = selectFactsForCondition(
        condition: condition,
        prompt: prompt,
        allFacts: allFacts,
        simRagConfig: simRagConfig,
      );
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
            content:
                '$localNotesTitle\n\n$localNotesIntro\n(No specific local notes match this query.)',
          ),
        );
      }
      messages.add(LLMMessage(role: LLMRole.user, content: userText));
      return messages;
  }
}

/// Patterns that mark a thought block, including a block the model never closed.
final List<RegExp> _thoughtPatterns = [
  RegExp(r'<(think|thought)>[\s\S]*?</\1>', caseSensitive: false),
  RegExp(r'<\|thought\|>[\s\S]*?</\|thought\|>', caseSensitive: false),
  RegExp(
    r'<\|channel\>thought[\s\S]*?(<channel\|>|<\|channel>|$)',
    caseSensitive: false,
  ),
  RegExp(r'<(think|thought)>[\s\S]*$', caseSensitive: false),
  RegExp(r'<\|thought\|>[\s\S]*$', caseSensitive: false),
];

/// True when [text] contains a thought block. The text is not modified.
bool outputContainsThought(String text) {
  for (final pattern in _thoughtPatterns) {
    if (pattern.hasMatch(text)) return true;
  }
  return false;
}

/// Strips reasoning/thinking tags (e.g. `<think>...</think>`, `<thought>...</thought>`,
/// `<|thought|>...</|thought|>`, and `<|channel>thought...`) from model outputs.
String stripThinkingBlocks(String text) {
  var cleaned = text;
  for (final pattern in _thoughtPatterns) {
    cleaned = cleaned.replaceAll(pattern, '');
  }
  return cleaned.trim();
}

/// Recognized end-of-turn or role boundary markers across model families.
const stopSequences = [
  '<|im_end|>',
  '<end_of_turn>',
  '<|eot_id|>',
  '<turn|>',
  '<|end|>',
  '<|im_start|>',
  '<start_of_turn>',
];

/// Sanitizes model output by truncating at the earliest stop sequence.
///
/// Thought blocks are removed unless [keepThoughts] is set. Gemma 4 sets that
/// so a thought the model still emitted stays in the stored output. `<eos>` is
/// cut only in that case; the other stop strings were already recognised.
String sanitizeModelOutput(String text, {bool keepThoughts = false}) {
  var cleaned = keepThoughts ? text : stripThinkingBlocks(text);
  final stops = <String>[
    ...stopSequences,
    if (keepThoughts) '<eos>',
  ];

  int? earliestStop;
  for (final stop in stops) {
    final idx = cleaned.indexOf(stop);
    if (idx != -1) {
      if (earliestStop == null || idx < earliestStop) {
        earliestStop = idx;
      }
    }
  }

  if (earliestStop != null) {
    cleaned = cleaned.substring(0, earliestStop);
  }

  return cleaned.trim();
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
