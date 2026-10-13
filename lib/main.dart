import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:technical_exploration/device_probe.dart';
import 'package:technical_exploration/kernel_diag.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/model_integrity.dart';
import 'package:technical_exploration/prompt_tokens.dart';
import 'package:technical_exploration/run_flags.dart';
import 'package:technical_exploration/sanity_check.dart';
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
  final _pathInputController = TextEditingController();

  LlamaCppChatRepository? _repo;
  CatalogData? _catalog;
  PromptSpec? _selected;

  String? _modelPath;
  String? _modelFileName;
  String _architecture = 'unknown';
  String _quantization = 'unknown';
  String _backend = 'CPU';
  int _nGpuLayers = 0;
  final bool _useMemoryMap = true;
  bool _sanityOk = false;
  String? _sanityFailure;
  String? _kernelLine;
  bool _finalizingRun = false;

  bool _hasLoadedEver = false;
  bool _isColdLoad = true;
  int? _loadDurationMs;
  int? _coldLoadMs;
  int? _ramBeforeLoadBytes;
  int? _ramIdleBytes;
  int? _peakRssBytes;
  int? _currentRssBytes;

  String? _loadError;
  String? _catalogError;
  bool _loading = false;
  bool _loaded = false;
  bool _running = false;
  bool _hasRun = false;
  bool _stopRequested = false;

  // Protocol condition & parameters
  PromptCondition _selectedCondition = PromptCondition.conditionA;
  bool _simulatedRagEnabled = false;

  // Reproducible decoding settings (protocol defaults)
  final double _temperature = 0.0; // Greedy decoding
  final int _seed = 42;
  final int _topK = 40;
  final double _topP = 0.9;
  final double _repeatPenalty = 1.1;
  final int _maxTokens = 256;
  final int _contextSize = 2048;

  int _sessionRunCount = 0;
  int _runIndexSinceLoad = 0;
  int? _ttftMs;
  double? _tokensPerSecond;
  int? _evalCount;
  int? _promptTokenCount;
  Duration? _generationDuration;
  double? _batteryTempCelsius;
  double? _heatTestThrottlePercentage;
  double? _heatTestMaxThrottlePercentage;
  double? _heatTestFirstRunTps;
  String? _status;

  final List<RunRecord> _history = [];
  final Map<PromptCondition, RunRecord?> _comparisonResults = {};

  StreamSubscription<LLMChunk>? _subscription;
  Stopwatch? _runWatch;
  Duration? _firstTokenAt;
  Completer<RunRecord?>? _runCompleter;

  // Batch progress tracking
  String? _batchProgressStatus;

  @override
  void initState() {
    super.initState();
    _currentRssBytes = MemorySnapshot.capture().currentRssBytes;
    _nGpuLayers = BackendDetector.getRecommendedGpuLayers();
    _backend = _nGpuLayers > 0 ? 'GPU' : 'CPU';
    unawaited(refreshDeviceIdentity());
    unawaited(_initApp());
  }

  Future<void> _initApp() async {
    await _reloadCatalog();
    final saved = await RunPersistenceManager.loadRuns();
    if (mounted && saved.isNotEmpty) {
      setState(() {
        _history.addAll(saved);
        _sessionRunCount = saved.length;
      });
    }
  }

  @override
  void dispose() {
    unawaited(_subscription?.cancel());
    _repo?.dispose();
    _outputController.dispose();
    _messagesController.dispose();
    _pathInputController.dispose();
    super.dispose();
  }

  bool get _canLoad =>
      _modelPath != null && _modelPath!.isNotEmpty && !_loading && !_running;

  bool get _canRun =>
      _loaded &&
      _sanityOk &&
      _selected != null &&
      !_running &&
      !_loading &&
      _catalog != null;

  DecodingSettings get _currentDecodingSettings => DecodingSettings(
    temperature: _temperature,
    seed: _seed,
    topK: _topK,
    topP: _topP,
    repeatPenalty: _repeatPenalty,
    maxTokens: _maxTokens,
    contextSize: _contextSize,
  );

  PromptCondition get _effectiveCondition {
    if (_selectedCondition == PromptCondition.conditionB) {
      return _simulatedRagEnabled
          ? PromptCondition.conditionBSimRag
          : PromptCondition.conditionB;
    }
    if (_selectedCondition == PromptCondition.conditionC) {
      return _simulatedRagEnabled
          ? PromptCondition.conditionCSimRag
          : PromptCondition.conditionC;
    }
    return PromptCondition.conditionA;
  }

  Future<void> _reloadCatalog() async {
    try {
      final simRagJson = await rootBundle
          .loadString('assets/simulated_rag.json')
          .catchError((_) => '');
      final catalog = parseCatalog(
        systemPrompt: await rootBundle.loadString('assets/system_prompt.txt'),
        promptsJson: await rootBundle.loadString('assets/prompts.json'),
        datasetJson: await rootBundle.loadString('assets/dataset.json'),
        simRagJson: simRagJson.isNotEmpty ? simRagJson : null,
      );
      if (!mounted) return;
      setState(() {
        _catalog = catalog;
        _catalogError = null;
        if (_selected != null) {
          final match =
              catalog.prompts.where((prompt) => prompt.id == _selected!.id);
          _selected = match.isEmpty ? null : match.first;
        } else if (catalog.prompts.isNotEmpty) {
          _selected = catalog.prompts.first;
        }
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _catalogError = error.toString());
    }
  }

  Future<void> _pickModel() async {
    // Use FileType.any so Android file picker does not hide .gguf files
    final file = await FilePicker.pickFile(
      dialogTitle: 'Choose a GGUF model',
      type: FileType.any,
    );
    if (!mounted || file == null) return;

    final path = file.path;
    if (path == null) {
      setState(() => _loadError = 'The picker did not return a local path.');
      return;
    }

    setState(() {
      _modelPath = path;
      _modelFileName = file.name;
      _pathInputController.text = path;
      _loaded = false;
      _loadError = null;
    });
  }

  Future<void> _loadModel() async {
    final path = _modelPath;
    if (path == null || _loading) return;

    setState(() {
      _loading = true;
      _loadError = null;
      _loaded = false;
      _sanityOk = false;
      _sanityFailure = null;
      _kernelLine = null;
      _status = 'Checking GGUF sha256...';
    });

    // If another model is currently loaded, unload it first
    if (_repo != null) {
      _repo?.dispose();
      _repo = null;
      // Brief pause to allow memory release before measuring pre-load RAM
      await Future<void>.delayed(const Duration(milliseconds: 150));
    }

    final ModelFileReport fileReport;
    try {
      fileReport = await inspectModelFile(path);
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _status = null;
        _loadError = 'Could not hash the GGUF: $error';
      });
      return;
    }
    // ignore: avoid_print
    print('[technical_exploration] ${fileReport.logLine}');
    if (fileReport.hashMismatch) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _loaded = false;
        _sanityOk = false;
        _status = fileReport.mismatchMessage;
        _loadError = fileReport.mismatchMessage;
      });
      return;
    }

    // Capture baseline RAM before allocating model
    final preSnap = MemorySnapshot.capture();
    _ramBeforeLoadBytes = preSnap.currentRssBytes;

    final isCold = !_hasLoadedEver;
    _hasLoadedEver = true;
    _isColdLoad = isCold;

    // Inspect GGUF header for architecture & quantization
    GgufMetadata? ggufMeta;
    try {
      ggufMeta = GgufMetadata.fromFileSync(path);
      _architecture = ggufMeta.architecture ?? 'unknown';
      _quantization = ggufMeta.quantizationType ?? 'unknown';
    } catch (_) {
      _architecture = 'qwen2.5';
      _quantization = 'Q4_K_M';
    }
    // ignore: avoid_print
    print(
      '[technical_exploration] gguf arch=$_architecture quant=$_quantization '
      'tensors=${ggufMeta?.tensorCount ?? 'n/a'}',
    );

    final repo = LlamaCppChatRepository(
      contextSize: _contextSize,
      batchSize: _contextSize,
      stopTokens: const [
        '<end_of_turn>',
        '<turn|>',
        '<|end|>',
        '<|im_end|>',
        '<|eot_id|>',
      ],
      nGpuLayers: _nGpuLayers,
    );
    _repo = repo;

    final watch = Stopwatch()..start();
    try {
      // ignore: deprecated_member_use
      await repo.loadModel(
        path,
        options: ModelLoadOptions(
          nGpuLayers: _nGpuLayers,
          useMemoryMap: _useMemoryMap,
        ),
      );
      watch.stop();

      final postSnap = MemorySnapshot.capture();
      final duration = watch.elapsedMilliseconds;
      _runIndexSinceLoad = 0;

      if (!mounted) return;
      setState(() {
        _loaded = true;
        _loadDurationMs = duration;
        if (isCold) _coldLoadMs = duration;
        _ramIdleBytes = postSnap.currentRssBytes;
        _currentRssBytes = postSnap.currentRssBytes;
        _status = 'Running sanity check...';
      });
      await _runSanityChecks();
      final kernel = collectKernelReport(
        useMemoryMap: _useMemoryMap,
        nGpuLayers: _nGpuLayers,
      );
      // ignore: avoid_print
      print('[technical_exploration] after load ${kernel.line}');
      // ignore: avoid_print
      print('[technical_exploration] after load ${kernel.detail}');
      if (!mounted) return;
      setState(() {
        _loading = false;
        _kernelLine = kernel.line;
        _status = _sanityOk
            ? 'Sanity check passed.'
            : (_sanityFailure ?? 'Sanity check failed. Batch not started.');
      });
    } catch (error) {
      watch.stop();
      if (!mounted) return;
      setState(() {
        _loaded = false;
        _sanityOk = false;
        _loadDurationMs = null;
        _loadError = error.toString();
        _status = error.toString();
        _loading = false;
      });
    }
  }

  /// Two fixed prompts. Both must pass before a protocol run can start.
  /// The answers are logged and shown. They are not protocol rows.
  Future<void> _runSanityChecks() async {
    final repo = _repo;
    final fileName = _modelFileName;
    if (repo == null || fileName == null) {
      _sanityOk = false;
      _sanityFailure = 'Sanity check did not run because the model is not loaded.';
      return;
    }

    final transcript = StringBuffer();
    final failures = <String>[];
    for (var i = 0; i < sanityPrompts.length; i++) {
      final prompt = sanityPrompts[i];
      if (mounted) {
        setState(() {
          _status = 'Sanity check ${i + 1}/${sanityPrompts.length}...';
        });
      }
      try {
        final result = await _probe(
          repo: repo,
          fileName: fileName,
          prompt: prompt.text,
        );
        final reason = sanityFailureReason(
          output: result.text,
          hitMaxTokens: result.hitMax,
          requireDigitFour: prompt.requireDigitFour,
        );
        transcript.writeln('sanity ${i + 1}: ${prompt.text}');
        transcript.writeln(result.text.isEmpty ? '(empty)' : result.text);
        transcript.writeln();
        // ignore: avoid_print
        print(
          '[technical_exploration] sanity ${i + 1} '
          'tokens=${result.tokens} hitMax=${result.hitMax} '
          'reason=${reason ?? 'pass'}',
        );
        // ignore: avoid_print
        print('[technical_exploration] sanity ${i + 1} output: ${result.text}');
        if (reason != null) {
          failures.add('"${prompt.text}" failed ($reason)');
        }
      } catch (error) {
        failures.add('"${prompt.text}" failed ($error)');
        // ignore: avoid_print
        print('[technical_exploration] sanity ${i + 1} error: $error');
      }
    }

    _sanityOk = failures.isEmpty;
    _sanityFailure = failures.isEmpty
        ? null
        : '${failures.join(' ')} Batch not started.';
    _outputController.text = transcript.toString().trimRight();
  }

  Future<({String text, int? tokens, bool hitMax})> _probe({
    required LlamaCppChatRepository repo,
    required String fileName,
    required String prompt,
  }) async {
    final buffer = StringBuffer();
    int? evalCount;
    final stream = repo.streamChatWithGenerationOptions(
      fileName,
      messages: [LLMMessage(role: LLMRole.user, content: prompt)],
      generationOptions: GenerationOptions(
        temperature: _temperature,
        topP: _topP,
        topK: _topK,
        maxTokens: sanityMaxTokens,
        seed: _seed,
        repeatPenalty: _repeatPenalty,
      ),
    );
    await for (final chunk in stream) {
      final text = chunk.message?.content;
      if (text != null && text.isNotEmpty) buffer.write(text);
      if (chunk.done == true) evalCount = chunk.evalCount;
    }
    final cleaned = sanitizeModelOutput(buffer.toString());
    final hitMax = evalCount != null && evalCount >= sanityMaxTokens;
    return (text: cleaned, tokens: evalCount, hitMax: hitMax);
  }

  void _logKernel(String when) {
    final report = collectKernelReport(
      useMemoryMap: _useMemoryMap,
      nGpuLayers: _nGpuLayers,
    );
    // ignore: avoid_print
    print('[technical_exploration] $when ${report.line}');
    // ignore: avoid_print
    print('[technical_exploration] $when ${report.detail}');
    if (mounted) setState(() => _kernelLine = report.line);
  }

  Future<void> _logSocThermal(String when) async {
    final line = await readSocThermalLine();
    if (line == null) return;
    // ignore: avoid_print
    print('[technical_exploration] soc-thermal $when $line');
  }

  Future<void> _choosePrompt() async {
    final catalog = _catalog;
    if (catalog == null) return;

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
  Future<RunRecord?> _runCondition(
    PromptCondition condition, {
    bool isHeatTest = false,
  }) async {
    final repo = _repo;
    final selected = _selected;
    final fileName = _modelFileName;
    final catalog = _catalog;
    if (repo == null ||
        selected == null ||
        fileName == null ||
        catalog == null ||
        _running) {
      return null;
    }
    if (!_sanityOk) {
      setState(() {
        _status = _sanityFailure ?? 'Sanity check has not passed. Run not started.';
      });
      return null;
    }

    final completer = Completer<RunRecord?>();
    _runCompleter = completer;

    final messages = buildConditionMessages(
      condition: condition,
      systemPrompt: catalog.systemPrompt,
      prompt: selected,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );

    // Pre-flight check: ensure estimated prompt tokens do not overflow context size
    final promptEstimate = _estimatePromptTokens(formatMessages(messages));
    if (promptEstimate > _contextSize) {
      setState(() {
        _status =
            'Prompt tokens (~$promptEstimate) exceed context size ($_contextSize).';
      });
      return null;
    }

    final initialSnap = MemorySnapshot.capture();
    await refreshDeviceIdentity();
    final batteryBefore = await readBattery();
    final batteryLevelBefore = batteryBefore.levelPercent;
    final batteryTempBefore = batteryBefore.tempCelsius;
    await _logSocThermal('before-run');
    _finalizingRun = false;

    // ignore: deprecated_member_use
    final modelRef = repo.model;
    final nativeTokens = countPromptTokens(
      model: modelRef?.pointer,
      vocab: modelRef?.vocab,
      messages: messages,
    );

    setState(() {
      _selectedCondition = condition;
      _running = true;
      _hasRun = true;
      _stopRequested = false;
      _ttftMs = null;
      _tokensPerSecond = null;
      _evalCount = null;
      _promptTokenCount = nativeTokens;
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
      generationOptions: GenerationOptions(
        temperature: _temperature,
        topP: _topP,
        topK: _topK,
        maxTokens: _maxTokens,
        seed: _seed,
        repeatPenalty: _repeatPenalty,
      ),
    );

    _subscription = stream.listen(
      (chunk) {
        _onChunk(
          chunk,
          condition: condition,
          batteryLevelBefore: batteryLevelBefore,
          batteryTempBefore: batteryTempBefore,
          isHeatTest: isHeatTest,
          completer: completer,
        );
      },
      onError: (Object error, StackTrace _) {
        if (!mounted) {
          if (!completer.isCompleted) completer.complete(null);
          return;
        }
        setState(() {
          _running = false;
          _status = error.toString();
        });
        if (!completer.isCompleted) completer.complete(null);
      },
      onDone: () {
        unawaited(
          _onRunDone(
            condition: condition,
            batteryLevelBefore: batteryLevelBefore,
            batteryTempBefore: batteryTempBefore,
            isHeatTest: isHeatTest,
            completer: completer,
          ),
        );
      },
    );

    return completer.future;
  }

  void _onChunk(
    LLMChunk chunk, {
    required PromptCondition condition,
    required int? batteryLevelBefore,
    required double? batteryTempBefore,
    required bool isHeatTest,
    required Completer<RunRecord?> completer,
  }) {
    final text = chunk.message?.content;
    if (text != null && text.isNotEmpty) {
      final isFirst = _firstTokenAt == null;
      _firstTokenAt ??= _runWatch?.elapsed;
      _outputController.text += text;

      if (isFirst) {
        final currentRss = MemorySnapshot.capture().currentRssBytes;
        if (_peakRssBytes == null || currentRss > _peakRssBytes!) {
          _peakRssBytes = currentRss;
        }
      }

      // Early stop if any stop sequence leaked into the accumulated output
      for (final stop in stopSequences) {
        if (_outputController.text.contains(stop)) {
          unawaited(_subscription?.cancel());
          _subscription = null;
          final first = _firstTokenAt;
          final elapsed = _runWatch?.elapsed;
          final generation = first == null || elapsed == null
              ? Duration.zero
              : elapsed - first;

          final approxTokens =
              _evalCount ?? (_outputController.text.length / 3.8).round();
          final rate = tokensPerSecond(
            evalCount: approxTokens,
            generation: generation,
          );

          if (mounted) {
            setState(() {
              if (first != null) _ttftMs = first.inMilliseconds;
              _tokensPerSecond = rate;
              _evalCount = approxTokens;
              _generationDuration = generation;
            });
          }

          unawaited(
            _onRunDone(
              condition: condition,
              batteryLevelBefore: batteryLevelBefore,
              batteryTempBefore: batteryTempBefore,
              isHeatTest: isHeatTest,
              completer: completer,
            ),
          );
          return;
        }
      }
    }

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

      if (!mounted) return;
      setState(() {
        if (first != null) _ttftMs = first.inMilliseconds;
        _tokensPerSecond = rate;
        _evalCount = chunk.evalCount;
        _promptTokenCount ??= chunk.promptEvalCount;
        _generationDuration = generation;
      });
    }
  }

  Future<void> _onRunDone({
    required PromptCondition condition,
    required int? batteryLevelBefore,
    required double? batteryTempBefore,
    required bool isHeatTest,
    required Completer<RunRecord?> completer,
  }) async {
    if (_finalizingRun || completer.isCompleted) return;
    _finalizingRun = true;
    if (completer.isCompleted) return;
    if (!mounted) {
      completer.complete(null);
      return;
    }

    // Capture final memory before context cleanup
    final currentRssFinal = MemorySnapshot.capture().currentRssBytes;
    if (_peakRssBytes == null || currentRssFinal > _peakRssBytes!) {
      _peakRssBytes = currentRssFinal;
    }

    final batteryAfter = await readBattery();
    final batteryLevelAfter = batteryAfter.levelPercent;
    final batteryTempAfter = batteryAfter.tempCelsius;
    await _logSocThermal('after-run');
    _sessionRunCount++;
    _runIndexSinceLoad++;

    // Compute thermal throttle ONLY within the heat test relative to run 1 of that test
    double? throttlePercentage;
    if (isHeatTest && _tokensPerSecond != null) {
      _heatTestFirstRunTps ??= _tokensPerSecond;
      if (_heatTestFirstRunTps != null && _heatTestFirstRunTps! > 0) {
        final currentThrottle =
            ((_heatTestFirstRunTps! - _tokensPerSecond!) /
                _heatTestFirstRunTps!) *
            100.0;
        throttlePercentage = currentThrottle;
        _heatTestThrottlePercentage = currentThrottle;
        _heatTestMaxThrottlePercentage = math.max(
          _heatTestMaxThrottlePercentage ?? 0.0,
          math.max(0.0, currentThrottle),
        );
      }
    }

    // Clean any thinking blocks and trailing stop sequences from output
    final cleanedOutput = sanitizeModelOutput(_outputController.text);
    _outputController.text = cleanedOutput;

    final selected = _selected!;
    final catalog = _catalog!;
    final facts = selectFactsForCondition(
      condition: condition,
      prompt: selected,
      allFacts: catalog.facts,
      simRagConfig: catalog.simRagConfig,
    );

    final hitMax = _evalCount != null && _evalCount! >= _maxTokens;
    final coldForThisRun = isColdForThisRun(
      loadWasCold: _isColdLoad,
      runIndexSinceLoad: _runIndexSinceLoad,
    );
    // No thinking toggle is applied for any model.
    final bool? thinkingDisabledValue = thinkingDisabledForRun(
      toggleApplied: false,
    );

    final record = RunRecord(
      runIndex: _sessionRunCount,
      runIndexSinceLoad: _runIndexSinceLoad,
      isColdLoad: coldForThisRun,
      loadDurationMs: _loadDurationMs,
      timestamp: DateTime.now(),
      deviceModel: readDeviceModel(),
      osVersion:
          '${Platform.operatingSystem} ${Platform.operatingSystemVersion}',
      modelFileName: _modelFileName ?? 'unknown',
      architecture: _architecture,
      quantization: _quantization,
      backend: _backend,
      nGpuLayers: _nGpuLayers,
      condition: condition.shortCode,
      notesMode: condition.notesMode,
      k: condition.isSimulatedRag ? catalog.simRagConfig.k : 0,
      injectedNoteIds: facts.map((f) => f.id).toList(),
      promptId: selected.id,
      promptText: selected.text,
      decodingSettings: _currentDecodingSettings,
      systemPrompt: condition.isConditionC
          ? buildConditionCSystemPrompt(catalog.systemPrompt)
          : catalog.systemPrompt,
      messagesSent: _messagesController.text,
      output: cleanedOutput,
      hitMaxTokens: hitMax,
      ramBeforeLoadBytes: _ramBeforeLoadBytes,
      ramIdleBytes: _ramIdleBytes,
      ramPeakBytes: _peakRssBytes,
      promptTokenCount: _promptTokenCount ??
          _estimatePromptTokens(_messagesController.text),
      generatedTokenCount: _evalCount,
      ttftMs: _ttftMs,
      generationDurationMs: _generationDuration?.inMilliseconds,
      tokensPerSecond: _tokensPerSecond,
      batteryLevelBefore: batteryLevelBefore,
      batteryLevelAfter: batteryLevelAfter,
      batteryTempBeforeCelsius: batteryTempBefore,
      batteryTempAfterCelsius: batteryTempAfter,
      throttlePercentage: throttlePercentage,
      isHeatTest: isHeatTest,
      thinkingDisabled: thinkingDisabledValue,
    );

    _history.add(record);
    _comparisonResults[condition] = record;
    unawaited(RunPersistenceManager.appendRun(record));

    setState(() {
      _running = false;
      _subscription = null;
      _batteryTempCelsius = batteryTempAfter;
      if (_stopRequested && _tokensPerSecond == null) {
        _status = 'Stopped before token count reported.';
      }
    });

    completer.complete(record);
  }

  int _estimatePromptTokens(String text) =>
      text.isEmpty ? 0 : (text.length / 3.8).round();

  Future<void> _stop() async {
    _stopRequested = true;
    final firstTokenAt = _firstTokenAt;
    await _subscription?.cancel();
    _subscription = null;
    if (!mounted) return;
    setState(() {
      _running = false;
      if (firstTokenAt != null) {
        _ttftMs = firstTokenAt.inMilliseconds;
      }
      if (_tokensPerSecond == null) {
        _status = 'Stopped before token count reported.';
      }
    });
    _runCompleter?.complete(null);
  }

  /// Runs Condition A, B, and C (and sim-RAG if enabled) for current prompt.
  Future<void> _runCompareAll() async {
    if (!_canRun) return;

    _logKernel('compare start');
    _stopRequested = false;
    setState(() => _comparisonResults.clear());

    final conditions = _simulatedRagEnabled
        ? [
            PromptCondition.conditionA,
            PromptCondition.conditionB,
            PromptCondition.conditionBSimRag,
            PromptCondition.conditionC,
            PromptCondition.conditionCSimRag,
          ]
        : [
            PromptCondition.conditionA,
            PromptCondition.conditionB,
            PromptCondition.conditionC,
          ];

    for (final condition in conditions) {
      if (!mounted || _stopRequested) break;
      await _runCondition(condition);
      await Future<void>.delayed(const Duration(milliseconds: 300));
    }
  }

  /// Runs the full empirical protocol pass across all 10 prompts.
  /// Fixed prompt-grouped order: For each prompt: A, B, C (plus sim-RAG if enabled).
  Future<void> _runBatchProtocol() async {
    final catalog = _catalog;
    if (!_canRun || catalog == null || catalog.prompts.isEmpty) return;

    _logKernel('batch start');
    _stopRequested = false;
    final conditions = _simulatedRagEnabled
        ? [
            PromptCondition.conditionA,
            PromptCondition.conditionB,
            PromptCondition.conditionBSimRag,
            PromptCondition.conditionC,
            PromptCondition.conditionCSimRag,
          ]
        : [
            PromptCondition.conditionA,
            PromptCondition.conditionB,
            PromptCondition.conditionC,
          ];

    final totalRuns = catalog.prompts.length * conditions.length;
    var currentCount = 0;

    for (final prompt in catalog.prompts) {
      if (!mounted || _stopRequested) break;
      setState(() {
        _selected = prompt;
        _comparisonResults.clear();
      });

      for (final condition in conditions) {
        if (!mounted || _stopRequested) break;
        currentCount++;
        setState(() {
          _batchProgressStatus =
              'Batch: $currentCount/$totalRuns [${condition.shortCode}] ${prompt.id}';
        });

        await _runCondition(condition);
        await Future<void>.delayed(const Duration(milliseconds: 300));
      }
    }

    final wasStopped = _stopRequested;
    if (mounted) {
      setState(() => _batchProgressStatus = null);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            wasStopped
                ? 'Protocol Batch pass stopped after $currentCount runs.'
                : 'Protocol Batch pass completed ($currentCount runs)!',
          ),
        ),
      );
    }
  }

  /// 10-run heat test: The same prompt and condition (who-manages, Condition B)
  /// run 10 times back-to-back on a warm-loaded model to measure thermal throttle.
  Future<void> _runHeatTest() async {
    final catalog = _catalog;
    if (!_canRun || catalog == null) return;

    _logKernel('heat test start');
    final targetPrompt = catalog.prompts.firstWhere(
      (p) => p.id == 'who-manages',
      orElse: () => catalog.prompts.first,
    );

    _stopRequested = false;
    setState(() {
      _selected = targetPrompt;
      _heatTestFirstRunTps = null;
      _heatTestThrottlePercentage = null;
      _heatTestMaxThrottlePercentage = null;
    });

    const totalRuns = 10;
    for (var i = 1; i <= totalRuns; i++) {
      if (!mounted || _stopRequested) break;
      setState(() {
        _batchProgressStatus = 'Heat Test: Run $i/$totalRuns (who-manages, Cond B)';
      });

      await _runCondition(PromptCondition.conditionB, isHeatTest: true);
      await Future<void>.delayed(const Duration(milliseconds: 200));
    }

    final wasStopped = _stopRequested;
    if (mounted) {
      setState(() => _batchProgressStatus = null);
      final throttleText = _heatTestMaxThrottlePercentage != null
          ? '${_heatTestMaxThrottlePercentage!.toStringAsFixed(1)}%'
          : '--';
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            wasStopped
                ? 'Heat Test stopped. Peak Throttle so far: $throttleText'
                : '10-Run Heat Test complete! Max Thermal Throttle: $throttleText',
          ),
        ),
      );
    }
  }

  void _copyCsvToClipboard() {
    if (_history.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No runs to copy yet.')),
      );
      return;
    }
    Clipboard.setData(
      ClipboardData(text: BenchmarkExporter.exportCsv(_history)),
    );
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('CSV copied to clipboard (${_history.length} runs)!'),
      ),
    );
  }

  void _copyJsonToClipboard() {
    if (_history.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No runs to copy yet.')),
      );
      return;
    }
    Clipboard.setData(
      ClipboardData(text: BenchmarkExporter.exportJson(_history)),
    );
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('JSON copied to clipboard (${_history.length} runs)!'),
      ),
    );
  }

  Future<void> _exportFiles() async {
    if (_history.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No runs to export yet.')),
      );
      return;
    }

    final paths = await BenchmarkExporter.exportToFiles(_history);
    if (!mounted) return;

    showDialog<void>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Benchmark Files Exported'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Files successfully saved:'),
              const SizedBox(height: 8),
              SelectableText('CSV: ${paths['csv']}'),
              const SizedBox(height: 4),
              SelectableText('JSON: ${paths['json']}'),
            ],
          ),
          actions: [
            TextButton.icon(
              icon: const Icon(Icons.table_chart, size: 16),
              onPressed: () {
                _copyCsvToClipboard();
                Navigator.pop(context);
              },
              label: const Text('Copy CSV'),
            ),
            TextButton.icon(
              icon: const Icon(Icons.code, size: 16),
              onPressed: () {
                _copyJsonToClipboard();
                Navigator.pop(context);
              },
              label: const Text('Copy JSON'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Done'),
            ),
          ],
        );
      },
    );
  }

  void _clearHistory() {
    showDialog<void>(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          title: const Text('Clear Runs History?'),
          content: const Text(
            'This will delete all persisted run records from disk.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () async {
                Navigator.pop(dialogContext);
                await RunPersistenceManager.clearRuns();
                if (!mounted) return;
                setState(() {
                  _history.clear();
                  _comparisonResults.clear();
                  _sessionRunCount = 0;
                });
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('History cleared.')),
                );
              },
              child: const Text('Clear'),
            ),
          ],
        );
      },
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
            icon: const Icon(Icons.file_download),
            tooltip: 'Export CSV & JSON',
            onPressed: _history.isNotEmpty ? _exportFiles : null,
          ),
          PopupMenuButton<String>(
            onSelected: (val) {
              if (val == 'copyCsv') {
                _copyCsvToClipboard();
              } else if (val == 'copyJson') {
                _copyJsonToClipboard();
              } else if (val == 'clear') {
                _clearHistory();
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'copyCsv',
                child: Text('Copy CSV to Clipboard'),
              ),
              const PopupMenuItem(
                value: 'copyJson',
                child: Text('Copy JSON to Clipboard'),
              ),
              const PopupMenuDivider(),
              const PopupMenuItem(
                value: 'clear',
                child: Text('Clear History'),
              ),
            ],
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
              baselineRssBytes: _ramBeforeLoadBytes,
              idleModelRssBytes: _ramIdleBytes != null && _ramBeforeLoadBytes != null
                  ? (_ramIdleBytes! - _ramBeforeLoadBytes!)
                  : null,
              peakRssBytes: _peakRssBytes,
              currentRssBytes: _currentRssBytes,
              ttftMs: _ttftMs,
              tokensPerSecond: _tokensPerSecond,
              evalCount: _evalCount,
              generationDuration: _generationDuration,
              runCount: _sessionRunCount,
              throttlePercentage: _heatTestThrottlePercentage,
              batteryTempCelsius: _batteryTempCelsius,
            ),
            const SizedBox(height: 20),

            // Model Selection
            Text('Model', style: textTheme.titleMedium),
            const SizedBox(height: 6),
            Text(
              _modelFileName ?? 'No file selected',
              style: TextStyle(
                fontWeight:
                    _modelFileName != null ? FontWeight.w600 : FontWeight.normal,
              ),
            ),
            if (_loaded) ...[
              const SizedBox(height: 4),
              Text(
                'Arch: $_architecture | Quant: $_quantization | Backend: $_backend (${_nGpuLayers}L) | ${_isColdLoad ? "Cold load" : "Warm load"}: ${_loadDurationMs ?? "--"} ms',
                style: textTheme.bodySmall?.copyWith(
                  color: colorScheme.onSurfaceVariant,
                ),
              ),
              if (_kernelLine != null) ...[
                const SizedBox(height: 4),
                Text(_kernelLine!, style: textTheme.bodySmall),
              ],
              if (_status != null) ...[
                const SizedBox(height: 4),
                Text(
                  _status!,
                  style: textTheme.bodySmall?.copyWith(
                    color: _sanityOk
                        ? colorScheme.onSurfaceVariant
                        : colorScheme.error,
                  ),
                ),
              ],
            ],
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
                  child: Text(_loaded ? 'Reload Model' : 'Load'),
                ),
                if (_loading)
                  const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(strokeWidth: 2.5),
                  ),
              ],
            ),
            if (_loading && _status != null) ...[
              const SizedBox(height: 6),
              Text(_status!, style: textTheme.bodySmall),
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

            // Protocol Condition Selector
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
                  tooltip: 'Include local dataset notes',
                ),
                ButtonSegment(
                  value: PromptCondition.conditionC,
                  label: Text('C: Constraints'),
                  tooltip: 'Strict grounding, 1-3 sentences, and behavioural bans',
                ),
              ],
              selected: {
                _selectedCondition == PromptCondition.conditionBSimRag
                    ? PromptCondition.conditionB
                    : (_selectedCondition == PromptCondition.conditionCSimRag
                        ? PromptCondition.conditionC
                        : _selectedCondition)
              },
              onSelectionChanged: _running
                  ? null
                  : (set) => setState(() => _selectedCondition = set.first),
            ),
            const SizedBox(height: 6),
            Text(
              _effectiveCondition.shortDescription,
              style: textTheme.bodySmall?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),

            const SizedBox(height: 10),
            // Simulated RAG Toggle (off by default)
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Simulated RAG (fixed note selection)'),
              subtitle: Text(
                _simulatedRagEnabled
                    ? 'Active: Condition B and C use fixed top-k notes (k=2) with near misses'
                    : 'Off (Default): Condition B and C receive all 10 locked notes',
                style: textTheme.bodySmall,
              ),
              value: _simulatedRagEnabled,
              onChanged: _running
                  ? null
                  : (val) => setState(() => _simulatedRagEnabled = val),
            ),

            const SizedBox(height: 8),

            // Reproducible Decoding Defaults (Protocol Locked)
            ExpansionTile(
              title: const Text('Reproducible Decoding Settings (Locked)'),
              subtitle: Text(
                'Temp: $_temperature (Greedy) | Seed: $_seed | MaxTok: $_maxTokens | Ctx: $_contextSize',
                style: textTheme.bodySmall,
              ),
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Temperature: $_temperature (Greedy, deterministic)',
                        style: textTheme.bodyMedium,
                      ),
                      Text('Seed: $_seed', style: textTheme.bodyMedium),
                      Text('Top-K: $_topK | Top-P: $_topP', style: textTheme.bodyMedium),
                      Text('Repeat Penalty: $_repeatPenalty', style: textTheme.bodyMedium),
                      Text('Max Tokens: $_maxTokens', style: textTheme.bodyMedium),
                      Text('Context Size: $_contextSize', style: textTheme.bodyMedium),
                      Text(
                        'Thinking toggle: not applied (export column is N/A)',
                        style: textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            if (_batchProgressStatus != null) ...[
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: colorScheme.primaryContainer,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Row(
                  children: [
                    const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        _batchProgressStatus!,
                        style: TextStyle(
                          color: colorScheme.onPrimaryContainer,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
            ],

            // Action Run Buttons
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                FilledButton(
                  onPressed: _canRun ? () => _runCondition(_effectiveCondition) : null,
                  child: Text('Run (${_effectiveCondition.shortCode})'),
                ),
                FilledButton.tonal(
                  onPressed: _canRun ? _runCompareAll : null,
                  child: Text(
                    _simulatedRagEnabled
                        ? 'Compare All (A vs B vs B-sim vs C vs C-sim)'
                        : 'Compare All 3 (A vs B vs C)',
                  ),
                ),
                FilledButton.icon(
                  icon: const Icon(Icons.playlist_play, size: 18),
                  onPressed: _canRun ? _runBatchProtocol : null,
                  label: Text(
                    _simulatedRagEnabled
                        ? 'Run Batch Pass (50 runs)'
                        : 'Run Batch Pass (30 runs)',
                  ),
                ),
                OutlinedButton.icon(
                  icon: const Icon(Icons.local_fire_department, size: 18),
                  onPressed: _canRun ? _runHeatTest : null,
                  label: const Text('10-Run Heat Test'),
                ),
                OutlinedButton(
                  onPressed: _running ? _stop : null,
                  child: const Text('Stop'),
                ),
              ],
            ),

            const SizedBox(height: 12),

            // Export & Copy Buttons (Mobile Friendly)
            Wrap(
              spacing: 12,
              runSpacing: 8,
              children: [
                OutlinedButton.icon(
                  icon: const Icon(Icons.table_chart, size: 16),
                  onPressed: _history.isNotEmpty ? _copyCsvToClipboard : null,
                  label: const Text('Copy CSV'),
                ),
                OutlinedButton.icon(
                  icon: const Icon(Icons.code, size: 16),
                  onPressed: _history.isNotEmpty ? _copyJsonToClipboard : null,
                  label: const Text('Copy JSON'),
                ),
                OutlinedButton.icon(
                  icon: const Icon(Icons.file_download, size: 16),
                  onPressed: _history.isNotEmpty ? _exportFiles : null,
                  label: const Text('Export Files'),
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
                labelText: 'Live Output (${_effectiveCondition.code})',
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
                  if (_ttftMs != null)
                    Text(
                      'TTFT: $_ttftMs ms',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  if (_tokensPerSecond != null)
                    Text(
                      'TPS: ${_tokensPerSecond!.toStringAsFixed(2)} tok/s',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  if (_evalCount != null) Text('Tokens: $_evalCount'),
                  if (_status != null)
                    Text(_status!, style: TextStyle(color: colorScheme.error)),
                ],
              ),
            ],

            // Neutral Condition Comparison View
            if (_selected != null && _comparisonResults.isNotEmpty) ...[
              ComparisonView(
                prompt: _selected!,
                results: _comparisonResults,
                showSimRag: _simulatedRagEnabled,
              ),
            ],

            const SizedBox(height: 24),
            Row(
              children: [
                Text('Messages sent', style: textTheme.titleMedium),
                const SizedBox(width: 8),
                if (_messagesController.text.isNotEmpty)
                  Text(
                    '(~${_estimatePromptTokens(_messagesController.text)} tokens)',
                    style: textTheme.bodySmall?.copyWith(
                      color: colorScheme.onSurfaceVariant,
                    ),
                  ),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.copy, size: 16),
                  tooltip: 'Copy full prompt sent',
                  visualDensity: VisualDensity.compact,
                  onPressed: _messagesController.text.isNotEmpty
                      ? () {
                          Clipboard.setData(
                            ClipboardData(text: _messagesController.text),
                          );
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Prompt copied to clipboard'),
                            ),
                          );
                        }
                      : null,
                ),
              ],
            ),
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
