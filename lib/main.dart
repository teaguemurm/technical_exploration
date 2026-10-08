import 'dart:async';
import 'dart:math';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/system_metrics.dart';
import 'package:technical_exploration/widgets/comparison_view.dart';
import 'package:technical_exploration/widgets/metrics_dashboard.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const TechnicalExplorationApp());
}

class TechnicalExplorationApp extends StatelessWidget {
  const TechnicalExplorationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'technical_exploration',
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF3F6B4A),
        useMaterial3: true,
      ),
      home: const NoteScreen(),
    );
  }
}

class NoteScreen extends StatefulWidget {
  const NoteScreen({super.key});

  @override
  State<NoteScreen> createState() => _NoteScreenState();
}

class _NoteScreenState extends State<NoteScreen> {
  final _outputController = TextEditingController();
  final _messagesController = TextEditingController();

  LlamaCppChatRepository? _repo;
  CatalogData? _catalog;
  PromptSpec? _selected;

  String? _modelPath;
  String? _modelFileName;
  int? _coldLoadMs;
  int? _baselineRssBytes;
  int? _idleModelRssBytes;
  int? _peakRssBytes;
  int? _currentRssBytes;

  String? _loadError;
  String? _catalogError;
  bool _loading = false;
  bool _loaded = false;
  bool _running = false;
  bool _hasRun = false;
  bool _stopRequested = false;

  // Condition & parameters
  PromptCondition _condition = PromptCondition.conditionA;
  bool _relevantNotesOnly = true;
  int _maxTokens = 256;

  int? _ttftMs;
  double? _tokensPerSecond;
  int? _evalCount;
  Duration? _generationDuration;
  double? _batteryTempCelsius;
  double? _throttlePercentage;
  double? _firstRunTps;
  int _sessionRunCount = 0;
  String? _status;

  final List<RunRecord> _history = [];
  final Map<PromptCondition, RunRecord?> _comparisonResults = {};

  StreamSubscription<LLMChunk>? _subscription;
  Stopwatch? _runWatch;
  Duration? _firstTokenAt;
  Completer<RunRecord?>? _runCompleter;

  @override
  void initState() {
    super.initState();
    _currentRssBytes = MemorySnapshot.capture().currentRssBytes;
    unawaited(_reloadCatalog());
  }

  @override
  void dispose() {
    unawaited(_subscription?.cancel());
    _repo?.dispose();
    _outputController.dispose();
    _messagesController.dispose();
    super.dispose();
  }

  bool get _canLoad => _modelPath != null && !_loading && !_running;

  bool get _canRun =>
      _loaded && _selected != null && !_running && !_loading && _catalog != null;

  Future<void> _reloadCatalog() async {
    try {
      final catalog = parseCatalog(
        systemPrompt: await rootBundle.loadString('assets/system_prompt.txt'),
        promptsJson: await rootBundle.loadString('assets/prompts.json'),
        datasetJson: await rootBundle.loadString('assets/dataset.json'),
      );
      if (!mounted) {
        return;
      }
      setState(() {
        _catalog = catalog;
        _catalogError = null;
        if (_selected != null) {
          final match = catalog.prompts.where((prompt) => prompt.id == _selected!.id);
          _selected = match.isEmpty ? null : match.first;
        } else if (catalog.prompts.isNotEmpty) {
          _selected = catalog.prompts.first;
        }
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() => _catalogError = error.toString());
    }
  }

  Future<void> _pickModel() async {
    final file = await FilePicker.pickFile(
      dialogTitle: 'Choose a GGUF model',
      type: FileType.custom,
      allowedExtensions: const ['gguf'],
    );
    if (!mounted || file == null) {
      return;
    }
    final path = file.path;
    if (path == null) {
      setState(() {
        _loadError = 'The picker did not return a local path.';
      });
      return;
    }
    setState(() {
      if (_modelPath != path) {
        _loaded = false;
        _coldLoadMs = null;
        _loadError = null;
      }
      _modelPath = path;
      _modelFileName = file.name;
    });
  }

  Future<void> _loadModel() async {
    final path = _modelPath;
    if (path == null || _loading) {
      return;
    }

    // Capture baseline RAM before allocating model
    final preSnap = MemorySnapshot.capture();
    setState(() {
      _loading = true;
      _loadError = null;
      _loaded = false;
      _coldLoadMs = null;
      _baselineRssBytes = preSnap.currentRssBytes;
      _idleModelRssBytes = null;
    });

    final gpuLayers = BackendDetector.getRecommendedGpuLayers();
    final repo = _repo ??= LlamaCppChatRepository(
      contextSize: 2048,
      nGpuLayers: gpuLayers,
    );
    final watch = Stopwatch()..start();
    try {
      // ignore: deprecated_member_use
      await repo.loadModel(
        path,
        options: ModelLoadOptions(
          nGpuLayers: gpuLayers,
          useMemoryMap: true,
        ),
      );
      watch.stop();

      final postSnap = MemorySnapshot.capture();
      final idleDiff = postSnap.currentRssBytes - preSnap.currentRssBytes;

      if (!mounted) {
        return;
      }
      setState(() {
        _loaded = true;
        _coldLoadMs = watch.elapsedMilliseconds;
        _currentRssBytes = postSnap.currentRssBytes;
        _idleModelRssBytes = idleDiff > 0 ? idleDiff : null;
        _loading = false;
      });
    } catch (error) {
      watch.stop();
      if (!mounted) {
        return;
      }
      setState(() {
        _loaded = false;
        _coldLoadMs = null;
        _loadError = error.toString();
        _loading = false;
      });
    }
  }

  Future<void> _choosePrompt() async {
    final catalog = _catalog;
    if (catalog == null) {
      return;
    }
    final picked = await showDialog<PromptSpec>(
      context: context,
      builder: (context) {
        return SimpleDialog(
          title: const Text('Prompts'),
          children: [
            for (final prompt in catalog.prompts)
              SimpleDialogOption(
                onPressed: () => Navigator.pop(context, prompt),
                child: Text(prompt.label),
              ),
          ],
        );
      },
    );
    if (picked != null && mounted) {
      setState(() {
        _selected = picked;
        _comparisonResults.clear();
      });
    }
  }

  /// Runs inference for a given condition and updates state.
  Future<RunRecord?> _runCondition(PromptCondition condition) async {
    final repo = _repo;
    final selected = _selected;
    final fileName = _modelFileName;
    final catalog = _catalog;
    if (repo == null || selected == null || fileName == null || catalog == null || _running) {
      return null;
    }

    final completer = Completer<RunRecord?>();
    _runCompleter = completer;

    final messages = buildConditionMessages(
      condition: condition,
      systemPrompt: catalog.systemPrompt,
      prompt: selected,
      allFacts: catalog.facts,
      relevantOnly: _relevantNotesOnly,
    );

    final initialSnap = MemorySnapshot.capture();

    setState(() {
      _condition = condition;
      _running = true;
      _hasRun = true;
      _stopRequested = false;
      _ttftMs = null;
      _tokensPerSecond = null;
      _evalCount = null;
      _generationDuration = null;
      _status = null;
      _firstTokenAt = null;
      _peakRssBytes = initialSnap.currentRssBytes;
      _currentRssBytes = initialSnap.currentRssBytes;
      _outputController.text = '';
      _messagesController.text = formatMessages(messages);
    });

    final watch = Stopwatch()..start();
    _runWatch = watch;

    final stream = repo.streamChatWithGenerationOptions(
      fileName,
      messages: messages,
      generationOptions: GenerationOptions(maxTokens: _maxTokens),
    );

    _subscription = stream.listen(
      _onChunk,
      onError: (Object error, StackTrace _) {
        if (!mounted) {
          completer.complete(null);
          return;
        }
        setState(() {
          _running = false;
          _status = error.toString();
        });
        completer.complete(null);
      },
      onDone: () {
        _onRunDone(condition, completer);
      },
    );

    return completer.future;
  }

  void _onChunk(LLMChunk chunk) {
    final text = chunk.message?.content;
    if (text != null && text.isNotEmpty) {
      _firstTokenAt ??= _runWatch?.elapsed;
      _outputController.text += text;
    }

    // Sample RAM during generation to track peak memory usage
    final currentRss = MemorySnapshot.capture().currentRssBytes;
    if (_peakRssBytes == null || currentRss > _peakRssBytes!) {
      _peakRssBytes = currentRss;
    }

    if (chunk.done == true) {
      final first = _firstTokenAt;
      final elapsed = _runWatch?.elapsed;
      final generation = first == null || elapsed == null
          ? Duration.zero
          : elapsed - first;

      final rate = tokensPerSecond(
        evalCount: chunk.evalCount,
        generation: generation,
      );

      if (!mounted) {
        return;
      }
      setState(() {
        if (first != null) {
          _ttftMs = first.inMilliseconds;
        }
        _tokensPerSecond = rate;
        _evalCount = chunk.evalCount;
        _generationDuration = generation;
      });
    }
  }

  void _onRunDone(PromptCondition condition, Completer<RunRecord?> completer) {
    if (!mounted) {
      completer.complete(null);
      return;
    }

    final temp = readDeviceTemperatureCelsius();
    _sessionRunCount++;

    if (_tokensPerSecond != null) {
      _firstRunTps ??= _tokensPerSecond;
      if (_firstRunTps != null && _firstRunTps! > 0) {
        _throttlePercentage =
            ((_firstRunTps! - _tokensPerSecond!) / _firstRunTps!) * 100.0;
      }
    }

    final record = RunRecord(
      promptId: _selected!.id,
      promptText: _selected!.text,
      conditionLabel: condition.code,
      relevantOnly: _relevantNotesOnly,
      maxTokens: _maxTokens,
      output: _outputController.text,
      timestamp: DateTime.now(),
      coldLoadMs: _coldLoadMs,
      ttftMs: _ttftMs,
      tokensPerSecond: _tokensPerSecond,
      evalCount: _evalCount,
      generationDuration: _generationDuration,
      peakRssBytes: _peakRssBytes,
      batteryTempCelsius: temp,
    );

    _history.add(record);
    _comparisonResults[condition] = record;

    setState(() {
      _running = false;
      _subscription = null;
      _batteryTempCelsius = temp;
      if (_stopRequested && _tokensPerSecond == null) {
        _status = 'Stopped before token count reported.';
      }
    });

    completer.complete(record);
  }

  Future<void> _stop() async {
    _stopRequested = true;
    final firstTokenAt = _firstTokenAt;
    await _subscription?.cancel();
    _subscription = null;
    if (!mounted) {
      return;
    }
    setState(() {
      _running = false;
      if (firstTokenAt != null) {
        _ttftMs = firstTokenAt.inMilliseconds;
      }
      if (_tokensPerSecond == null) {
        _status = 'Stopped before the library reported a token count.';
      }
    });
    _runCompleter?.complete(null);
  }

  /// Runs Condition A, B, and C sequentially for the currently selected prompt.
  Future<void> _runCompareAll() async {
    if (!_canRun) return;

    for (final condition in PromptCondition.values) {
      if (!mounted || _stopRequested) break;
      await _runCondition(condition);
      // Small pause between runs for thermal/memory settling
      await Future<void>.delayed(const Duration(milliseconds: 300));
    }
  }

  /// Runs 3 standard prompts back-to-back to directly test thermal throttling.
  Future<void> _runThrottleTest() async {
    final catalog = _catalog;
    if (!_canRun || catalog == null || catalog.prompts.isEmpty) return;

    final promptsToTest = catalog.prompts.take(3).toList();
    for (var i = 0; i < promptsToTest.length; i++) {
      if (!mounted || _stopRequested) break;
      setState(() => _selected = promptsToTest[i]);
      await _runCondition(_condition);
      await Future<void>.delayed(const Duration(milliseconds: 200));
    }

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '3-Prompt Throttle Test complete! Throttle: ${_throttlePercentage != null ? "${_throttlePercentage!.toStringAsFixed(1)}%" : "--"}',
          ),
        ),
      );
    }
  }

  void _copyBenchmarkReport() {
    final report = BenchmarkExporter.exportMarkdownReport(
      modelFileName: _modelFileName,
      coldLoadMs: _coldLoadMs,
      baselineRssBytes: _baselineRssBytes,
      idleModelRssBytes: _idleModelRssBytes,
      runs: _history,
    );
    Clipboard.setData(ClipboardData(text: report));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Benchmark report copied to clipboard!')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Visitor note & Benchmark'),
        actions: [
          IconButton(
            icon: const Icon(Icons.copy_all),
            tooltip: 'Copy Benchmark Report',
            onPressed: _history.isNotEmpty ? _copyBenchmarkReport : null,
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            // Telemetry dashboard
            MetricsDashboard(
              coldLoadMs: _coldLoadMs,
              baselineRssBytes: _baselineRssBytes,
              idleModelRssBytes: _idleModelRssBytes,
              peakRssBytes: _peakRssBytes,
              currentRssBytes: _currentRssBytes,
              ttftMs: _ttftMs,
              tokensPerSecond: _tokensPerSecond,
              evalCount: _evalCount,
              generationDuration: _generationDuration,
              runCount: _sessionRunCount,
              throttlePercentage: _throttlePercentage,
              batteryTempCelsius: _batteryTempCelsius,
            ),
            const SizedBox(height: 20),

            // Model Selection
            Text('Model', style: textTheme.titleMedium),
            const SizedBox(height: 6),
            Text(
              _modelFileName ?? 'No file selected',
              style: TextStyle(
                fontWeight: _modelFileName != null ? FontWeight.w600 : FontWeight.normal,
              ),
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                FilledButton.tonal(
                  onPressed: _running || _loading ? null : _pickModel,
                  child: const Text('Pick GGUF'),
                ),
                FilledButton(
                  onPressed: _canLoad ? _loadModel : null,
                  child: const Text('Load'),
                ),
                if (_loading) const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2.5)),
              ],
            ),
            if (_coldLoadMs != null) ...[
              const SizedBox(height: 6),
              Text('Cold load: $_coldLoadMs ms', style: const TextStyle(fontWeight: FontWeight.w500)),
            ],
            if (_loadError != null) ...[
              const SizedBox(height: 6),
              Text(_loadError!, style: TextStyle(color: colorScheme.error)),
            ],

            const Divider(height: 32),

            // Prompt & Condition Section
            Text('Evaluation & Prompting', style: textTheme.titleMedium),
            const SizedBox(height: 10),
            Align(
              alignment: Alignment.centerLeft,
              child: FilledButton.tonal(
                onPressed: _catalog == null || _running ? null : _choosePrompt,
                child: Text(_selected?.label ?? 'Choose prompt'),
              ),
            ),
            if (_catalogError != null) ...[
              const SizedBox(height: 8),
              Text(_catalogError!, style: TextStyle(color: colorScheme.error)),
            ],
            const SizedBox(height: 14),

            // Three Condition Selector
            Text('Prompt Condition', style: textTheme.labelLarge),
            const SizedBox(height: 6),
            SegmentedButton<PromptCondition>(
              segments: const [
                ButtonSegment(
                  value: PromptCondition.conditionA,
                  label: Text('A: Base'),
                  tooltip: 'System prompt only',
                ),
                ButtonSegment(
                  value: PromptCondition.conditionB,
                  label: Text('B: Dataset'),
                  tooltip: 'Include local notes retrieval',
                ),
                ButtonSegment(
                  value: PromptCondition.conditionC,
                  label: Text('C: Constraints'),
                  tooltip: 'Constrained output + refusal',
                ),
              ],
              selected: {_condition},
              onSelectionChanged: _running
                  ? null
                  : (set) => setState(() => _condition = set.first),
            ),
            const SizedBox(height: 6),
            Text(
              _condition.shortDescription,
              style: textTheme.bodySmall?.copyWith(color: colorScheme.onSurfaceVariant),
            ),

            if (_condition != PromptCondition.conditionA) ...[
              const SizedBox(height: 10),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Simulated RAG (Relevant notes only)'),
                subtitle: Text(
                  _relevantNotesOnly
                      ? 'Injects only prompt-matched facts'
                      : 'Stuffs all 10 local notes into context',
                  style: textTheme.bodySmall,
                ),
                value: _relevantNotesOnly,
                onChanged: _running
                    ? null
                    : (val) => setState(() => _relevantNotesOnly = val),
              ),
            ],

            const SizedBox(height: 10),

            // Max Tokens / Generation Budget
            Row(
              children: [
                Text('Max Tokens:', style: textTheme.labelLarge),
                const SizedBox(width: 8),
                Text('$_maxTokens tok', style: const TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 4),
            Wrap(
              spacing: 8,
              children: [
                for (final tokens in [64, 128, 256, 512, 1024])
                  ChoiceChip(
                    label: Text('$tokens'),
                    selected: _maxTokens == tokens,
                    onSelected: _running
                        ? null
                        : (selected) {
                            if (selected) setState(() => _maxTokens = tokens);
                          },
                  ),
              ],
            ),

            const SizedBox(height: 16),

            // Action Run Buttons
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                FilledButton(
                  onPressed: _canRun ? () => _runCondition(_condition) : null,
                  child: const Text('Run'),
                ),
                FilledButton.tonal(
                  onPressed: _canRun ? _runCompareAll : null,
                  child: const Text('Compare All 3 (A vs B vs C)'),
                ),
                OutlinedButton(
                  onPressed: _canRun ? _runThrottleTest : null,
                  child: const Text('3-Prompt Throttle Test'),
                ),
                OutlinedButton(
                  onPressed: _running ? _stop : null,
                  child: const Text('Stop'),
                ),
              ],
            ),

            const SizedBox(height: 24),

            // Live Output Box
            TextField(
              controller: _outputController,
              readOnly: true,
              minLines: 6,
              maxLines: 16,
              decoration: InputDecoration(
                labelText: 'Live Output (${_condition.code})',
                alignLabelWithHint: true,
                border: const OutlineInputBorder(),
              ),
            ),

            if (_hasRun && !_running) ...[
              const SizedBox(height: 10),
              Wrap(
                spacing: 16,
                runSpacing: 6,
                children: [
                  if (_ttftMs != null) Text('TTFT: $_ttftMs ms', style: const TextStyle(fontWeight: FontWeight.w600)),
                  if (_tokensPerSecond != null)
                    Text('TPS: ${_tokensPerSecond!.toStringAsFixed(2)} tok/s', style: const TextStyle(fontWeight: FontWeight.w600)),
                  if (_evalCount != null) Text('Tokens: $_evalCount'),
                  if (_status != null) Text(_status!, style: TextStyle(color: colorScheme.error)),
                ],
              ),
            ],

            // Condition Comparison View
            if (_selected != null && _comparisonResults.isNotEmpty) ...[
              ComparisonView(
                prompt: _selected!,
                results: _comparisonResults,
              ),
            ],

            const SizedBox(height: 24),
            Text('Messages sent', style: textTheme.titleMedium),
            const SizedBox(height: 8),
            TextField(
              controller: _messagesController,
              readOnly: true,
              minLines: 6,
              maxLines: 14,
              decoration: const InputDecoration(
                alignLabelWithHint: true,
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
