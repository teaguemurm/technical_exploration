import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:technical_exploration/build_variant.dart';
import 'package:technical_exploration/device_probe.dart';
import 'package:technical_exploration/kernel_diag.dart';
import 'package:technical_exploration/kleidi_sme.dart';
import 'package:technical_exploration/lora_match.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/model_integrity.dart';
import 'package:technical_exploration/prompt_tokens.dart';
import 'package:technical_exploration/public_exports.dart';
import 'package:technical_exploration/run_flags.dart';
import 'package:technical_exploration/sanity_check.dart';
import 'package:technical_exploration/system_metrics.dart';
import 'package:technical_exploration/widgets/config_card.dart';
import 'package:technical_exploration/widgets/diagnostics_card.dart';
import 'package:technical_exploration/widgets/messages_card.dart';
import 'package:technical_exploration/widgets/metrics_dashboard.dart';
import 'package:technical_exploration/widgets/report_surface.dart';
import 'package:technical_exploration/widgets/setup_card.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  applyKleidiAiSmeBeforeBackendInit();
  runApp(const TechnicalExplorationApp());
}

class TechnicalExplorationApp extends StatelessWidget {
  const TechnicalExplorationApp({super.key});

  @override
  Widget build(BuildContext context) {
    final scheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF3F6B4A),
      brightness: Brightness.light,
    );
    return MaterialApp(
      title: 'technical_exploration',
      theme: ThemeData(
        colorScheme: scheme,
        brightness: Brightness.light,
        useMaterial3: true,
        cardTheme: CardThemeData(
          color: scheme.surface,
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: BorderSide(color: scheme.outlineVariant),
          ),
        ),
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
  int? _modelSizeBytes;
  String? _verifiedPath;
  String? _adapterPath;
  String? _adapterFileName;
  String? _loadedAdapterPath;
  String? _loadedAdapterFileName;
  String? _loadedAdapterSha256;
  double? _loadedAdapterScale;
  int? _baseLoadMs;
  int? _adapterLoadMs;
  int? _parameterCount;
  String _deviceLabel = '';
  DateTime? _lastRunAt;
  PromptCondition? _shownCondition;
  List<LLMMessage> _sentMessages = const [];
  String _architecture = 'unknown';
  String _quantization = 'unknown';
  String _backend = 'CPU';
  int _nGpuLayers = 0;
  final bool _useMemoryMap = !noMmapBuild;

  /// Passed to [LlamaCppChatRepository]. Null means the package does not
  /// assign `n_threads`. The kernel line logs this same value.
  final int? _threads = threadsFromBuildDefine();
  bool _sanityOk = false;
  bool _sanityRan = false;
  bool _skipSanityRequested = false;
  String? _sanityFailure;
  String? _kernelLine;
  ModelHashStore? _hashStore;
  bool _finalizingRun = false;

  bool _hasLoadedEver = false;
  bool _isColdLoad = true;
  int? _loadDurationMs;
  int? _coldLoadMs;
  int? _ramBeforeLoadBytes;
  int? _ramIdleBytes;
  int? _peakRssBytes;

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
  String? _preflightPromptFormatter;
  String? _sentPromptFormatter;
  Duration? _generationDuration;
  double? _batteryTempCelsius;
  double? _heatTestThrottlePercentage;
  double? _heatTestMaxThrottlePercentage;
  double? _heatTestFirstRunTps;
  String? _status;

  final List<RunRecord> _history = [];
  List<String>? _lastPublicExportUris;
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
    _nGpuLayers =
        gpuLayersFromBuildDefine() ?? BackendDetector.getRecommendedGpuLayers();
    _backend = _nGpuLayers > 0 ? 'GPU' : 'CPU';
    _deviceLabel = readDeviceModel();
    unawaited(refreshDeviceIdentity().then((_) {
      if (!mounted) return;
      setState(() => _deviceLabel = readDeviceModel());
    }));
    unawaited(_initApp());
  }

  Future<void> _initApp() async {
    await _reloadCatalog();
    await _loadHashStore();
    final saved = await RunPersistenceManager.loadRuns();
    if (mounted && saved.isNotEmpty) {
      setState(() {
        _history.addAll(saved);
        _sessionRunCount = saved.length;
      });
    }
  }

  Future<void> _loadHashStore() async {
    try {
      _hashStore = await _ensureHashStore();
    } catch (error) {
      // ignore: avoid_print
      print('[technical_exploration] model hash cache unavailable: $error');
    }
  }

  Future<ModelHashStore> _ensureHashStore() async {
    final existing = _hashStore;
    if (existing != null) return existing;
    final exports = await BenchmarkExporter.getExportsDirectory();
    final store = ModelHashStore(
      File('${exports.path}/$modelHashCacheFileName'),
    );
    await store.load();
    _hashStore = store;
    return store;
  }

  /// Hashes the selected GGUF and stores path+size+mtime. Load does not.
  Future<void> _verifyModel() async {
    final path = _modelPath;
    if (path == null || _loading || _running) return;

    setState(() {
      _loading = true;
      _loadError = null;
      _status = 'Checking GGUF sha256...';
    });

    try {
      final store = await _ensureHashStore();
      final report = await inspectModelFile(
        path,
        cache: store.cache,
        force: true,
      );
      await store.save();
      // ignore: avoid_print
      print('[technical_exploration] verify ${report.logLine}');
      if (!mounted) return;
      if (report.hashMismatch) {
        _repo?.dispose();
        _repo = null;
        setState(() {
          _loading = false;
          _loaded = false;
          _sanityOk = false;
          _sanityRan = false;
          _verifiedPath = null;
          _status = report.mismatchMessage;
          _loadError = report.mismatchMessage;
        });
        return;
      }
      setState(() {
        _loading = false;
        _loadError = null;
        _verifiedPath = path;
        _status =
            'Verified ${report.fileName} sha256=${report.sha256} '
            'knownQwenMatch=${report.matchesKnownQwen}.';
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _loadError = 'Could not hash the GGUF: $error';
        _status = 'Could not hash the GGUF: $error';
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

  /// Mac-only. Android always runs the sanity prompts.
  bool get _skipSanityEffective => _skipSanityRequested && !Platform.isAndroid;

  bool get _adapterStale => _loaded && _adapterPath != _loadedAdapterPath;

  bool get _canRun =>
      _loaded &&
      _sanityOk &&
      _selected != null &&
      !_running &&
      !_loading &&
      _catalog != null &&
      !_adapterStale;

  String? get _runBlockedReason {
    if (_running || _canRun) return null;
    if (_loading) return 'The model is still loading';
    if (_adapterStale) return 'Load the model to apply the adapter';
    if (_modelPath == null || _modelPath!.isEmpty) return 'Pick a model first';
    if (!_loaded) return 'Load a model first';
    if (!_sanityOk) {
      return _sanityFailure ?? 'Sanity check failed';
    }
    if (_catalog == null || _selected == null) {
      return 'Choose a visitor question first';
    }
    return 'Load a model first';
  }

  String get _modelStatusLine {
    if (_loading) return 'Loading…';
    if (_loaded && _verifiedPath != null && _verifiedPath == _modelPath) {
      return 'Verified';
    }
    if (_loaded) return 'Loaded';
    return 'Not loaded';
  }

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
          final match = catalog.prompts.where(
            (prompt) => prompt.id == _selected!.id,
          );
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
      _modelSizeBytes = _fileSize(path);
      _pathInputController.text = path;
      _loaded = false;
      _loadError = null;
    });
  }

  Future<void> _pickAdapter() async {
    final file = await FilePicker.pickFile(
      dialogTitle: 'Choose a LoRA adapter GGUF',
      type: FileType.any,
    );
    if (!mounted || file == null) return;
    final path = file.path;
    if (path == null) {
      setState(() => _loadError = 'The picker did not return a local path.');
      return;
    }
    setState(() {
      _adapterPath = path;
      _adapterFileName = file.name;
      _loadError = null;
    });
  }

  void _clearAdapter() {
    setState(() {
      _adapterPath = null;
      _adapterFileName = null;
    });
  }

  int? _fileSize(String path) {
    try {
      return File(path).lengthSync();
    } catch (_) {
      return null;
    }
  }

  String? _readArchitecture(String path) {
    try {
      return GgufMetadata.fromFileSync(path).architecture;
    } catch (_) {
      return null;
    }
  }

  void _rememberLoad({
    required int baseMs,
    required int adapterMs,
    required int? idleBytes,
    required bool isCold,
    required ModelFileReport? adapterReport,
  }) {
    final total = baseMs + adapterMs;
    _baseLoadMs = baseMs;
    _adapterLoadMs = adapterMs;
    _loadDurationMs = total;
    if (isCold) _coldLoadMs = total;
    _ramIdleBytes = idleBytes;
    if (adapterReport == null) {
      _loadedAdapterPath = null;
      _loadedAdapterFileName = null;
      _loadedAdapterSha256 = null;
      _loadedAdapterScale = null;
    } else {
      _loadedAdapterPath = adapterReport.path;
      _loadedAdapterFileName = adapterReport.fileName;
      _loadedAdapterSha256 = adapterReport.sha256;
      _loadedAdapterScale = protocolLoraScale;
    }
    // ignore: avoid_print
    print(
      '[technical_exploration] load-time: base=${baseMs}ms '
      'adapter=${adapterMs}ms total=${total}ms',
    );
  }

  Future<void> _finishSanity() async {
    if (_skipSanityEffective) {
      _sanityRan = false;
      _sanityOk = true;
      _sanityFailure = null;
    } else {
      await _runSanityChecks();
    }
  }

  void _failLoad(String message) {
    if (!mounted) return;
    setState(() {
      _loading = false;
      _loaded = false;
      _sanityOk = false;
      _loadDurationMs = null;
      _parameterCount = null;
      _status = message;
      _loadError = message;
    });
  }

  Future<void> _loadModel() async {
    final path = _modelPath;
    if (path == null || _loading) return;
    final adapterPath = _adapterPath;

    setState(() {
      _loading = true;
      _loadError = null;
      _loaded = false;
      _sanityOk = false;
      _sanityRan = false;
      _sanityFailure = null;
      _kernelLine = null;
      _loadedAdapterPath = null;
      _loadedAdapterFileName = null;
      _loadedAdapterSha256 = null;
      _loadedAdapterScale = null;
      _hasRun = false;
      _ttftMs = null;
      _tokensPerSecond = null;
      _evalCount = null;
      _peakRssBytes = null;
      _generationDuration = null;
      _lastRunAt = null;
      _shownCondition = null;
      _sentMessages = const [];
      _outputController.text = '';
      _messagesController.text = '';
      _status = 'Checking cached model hash...';
    });

    // If another model is currently loaded, unload it first
    if (_repo != null) {
      _repo?.dispose();
      _repo = null;
      // Brief pause to allow memory release before measuring pre-load RAM
      await Future<void>.delayed(const Duration(milliseconds: 150));
    }

    GgufMetadata? ggufMeta;
    try {
      ggufMeta = GgufMetadata.fromFileSync(path);
    } catch (_) {
      ggufMeta = null;
    }

    ModelFileReport? adapterReport;
    if (adapterPath != null) {
      final refusal = adapterArchitectureRefusal(
        modelArchitecture: ggufMeta?.architecture,
        adapterArchitecture: _readArchitecture(adapterPath),
      );
      if (refusal != null) {
        _failLoad(refusal);
        return;
      }
      try {
        final store = await _ensureHashStore();
        adapterReport = await inspectModelFile(adapterPath, cache: store.cache);
        await store.save();
      } catch (error) {
        _failLoad('Could not hash the adapter: $error');
        return;
      }
    }

    final ModelFileReport fileReport;
    try {
      final store = await _ensureHashStore();
      fileReport = await lookupCachedModelFile(path, store.cache);
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _status = null;
        _loadError = 'Could not check the cached GGUF hash: $error';
      });
      return;
    }
    // ignore: avoid_print
    print('[technical_exploration] ${fileReport.logLine}');
    if (adapterReport == null) {
      // ignore: avoid_print
      print('[technical_exploration] adapter-file: none');
    } else {
      // ignore: avoid_print
      print(
        '[technical_exploration] adapter-file: name=${adapterReport.fileName} '
        'sha256=${adapterReport.sha256} scale=$protocolLoraScale',
      );
    }
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

    if (ggufMeta != null) {
      _architecture = ggufMeta.architecture ?? 'unknown';
      _quantization = ggufMeta.quantizationType ?? 'unknown';
    } else {
      // Unchanged fallback for a base file whose header cannot be read.
      _architecture = 'qwen2.5';
      _quantization = 'Q4_K_M';
    }
    // ignore: avoid_print
    print(
      '[technical_exploration] gguf arch=$_architecture quant=$_quantization '
      'tensors=${ggufMeta?.tensorCount ?? 'n/a'}',
    );

    // Capture baseline RAM before allocating model
    final preSnap = MemorySnapshot.capture();
    _ramBeforeLoadBytes = preSnap.currentRssBytes;

    final isCold = !_hasLoadedEver;
    _hasLoadedEver = true;
    _isColdLoad = isCold;

    final repo = LlamaCppChatRepository(
      contextSize: _contextSize,
      batchSize: _contextSize,
      threads: _threads,
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
      final baseMs = watch.elapsedMilliseconds;
      _runIndexSinceLoad = 0;
      _modelSizeBytes = _fileSize(path) ?? _modelSizeBytes;
      // ignore: deprecated_member_use
      final params = repo.model?.parameterCount;
      _parameterCount = params == null || params <= 0 ? null : params;

      if (adapterPath == null) {
        final postSnap = MemorySnapshot.capture();
        _rememberLoad(
          baseMs: baseMs,
          adapterMs: 0,
          idleBytes: postSnap.currentRssBytes,
          isCold: isCold,
          adapterReport: null,
        );
        if (!mounted) return;
        setState(() {
          _loaded = true;
          _status = _skipSanityEffective
              ? 'Sanity check skipped.'
              : 'Running sanity check...';
        });
        await _finishSanity();
      } else {
        if (!mounted) return;
        setState(() {
          _status = _skipSanityEffective
              ? 'Sanity check skipped.'
              : 'Running sanity check...';
        });
        await _finishSanity();
        if (!mounted) return;
        await Future<void>.delayed(const Duration(milliseconds: 150));
        final adapterWatch = Stopwatch()..start();
        try {
          repo.attachLora(adapterPath, scale: protocolLoraScale);
        } catch (error) {
          adapterWatch.stop();
          repo.dispose();
          _repo = null;
          _failLoad('Could not load this adapter. $error');
          return;
        }
        adapterWatch.stop();
        final postSnap = MemorySnapshot.capture();
        _rememberLoad(
          baseMs: baseMs,
          adapterMs: adapterWatch.elapsedMilliseconds,
          idleBytes: postSnap.currentRssBytes,
          isCold: isCold,
          adapterReport: adapterReport,
        );
      }

      final kernel = collectKernelReport(
        useMemoryMap: _useMemoryMap,
        nGpuLayers: _nGpuLayers,
        threads: _threads,
      );
      // ignore: avoid_print
      print('[technical_exploration] after load ${kernel.line}');
      // ignore: avoid_print
      print('[technical_exploration] after load ${kernel.detail}');
      if (!mounted) return;
      setState(() {
        _loading = false;
        _loaded = true;
        _kernelLine = kernel.line;
        _status = _skipSanityEffective
            ? 'Sanity check skipped. The first protocol run can be a cold load.'
            : (_sanityOk
                  ? 'Sanity check passed.'
                  : (_sanityFailure ??
                        'Sanity check failed. Batch not started.'));
      });
    } catch (error) {
      watch.stop();
      if (!mounted) return;
      setState(() {
        _loaded = false;
        _sanityOk = false;
        _loadDurationMs = null;
        _parameterCount = null;
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
      _sanityFailure =
          'Sanity check did not run because the model is not loaded.';
      return;
    }

    final transcript = StringBuffer();
    final failures = <String>[];
    _sanityRan = true;
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
          requireAnimalWord: prompt.requireAnimalWord,
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
    String? formatter;
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
      if (chunk.done == true) {
        evalCount = chunk.evalCount;
        formatter = chunk.providerMetadata['promptFormatter'] as String?;
      }
    }
    final cleaned = sanitizeModelOutput(
      buffer.toString(),
      keepThoughts: formatter == promptFormatterGemma4,
    );
    final hitMax = evalCount != null && evalCount >= sanityMaxTokens;
    return (text: cleaned, tokens: evalCount, hitMax: hitMax);
  }

  void _logKernel(String when) {
    final report = collectKernelReport(
      useMemoryMap: _useMemoryMap,
      nGpuLayers: _nGpuLayers,
      threads: _threads,
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
        _status =
            _sanityFailure ?? 'Sanity check has not passed. Run not started.';
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
    _preflightPromptFormatter = null;
    _sentPromptFormatter = null;
    final CountedPrompt? counted;
    try {
      counted = countPromptTokens(
        model: modelRef?.pointer,
        vocab: modelRef?.vocab,
        messages: messages,
      );
    } on ChatTemplateException catch (error) {
      setState(() {
        _status = error.message;
      });
      if (!completer.isCompleted) completer.complete(null);
      return null;
    }
    _preflightPromptFormatter = counted?.formatter;

    setState(() {
      _selectedCondition = switch (condition) {
        PromptCondition.conditionBSimRag => PromptCondition.conditionB,
        PromptCondition.conditionCSimRag => PromptCondition.conditionC,
        _ => condition,
      };
      _running = true;
      _hasRun = true;
      _stopRequested = false;
      _ttftMs = null;
      _tokensPerSecond = null;
      _evalCount = null;
      _promptTokenCount = counted?.tokenCount;
      _generationDuration = null;
      _status = null;
      _firstTokenAt = null;
      _peakRssBytes = initialSnap.currentRssBytes;
      _outputController.text = '';
      _messagesController.text = formatMessages(messages);
      _sentMessages = messages;
      _shownCondition = condition;
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

      // Early stop if any stop sequence leaked into the accumulated output.
      // Gemma also stops if `<eos>` leaked; other families leave that string
      // alone so a Qwen answer is unchanged.
      final leakedStops = _preflightPromptFormatter == promptFormatterGemma4
          ? const <String>[...stopSequences, '<eos>']
          : stopSequences;
      for (final stop in leakedStops) {
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
        final reported = chunk.providerMetadata['promptFormatter'];
        if (reported is String && reported.isNotEmpty) {
          _sentPromptFormatter = reported;
        }
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

    final sentFormatter = _sentPromptFormatter;
    final preflightFormatter = _preflightPromptFormatter;
    if (preflightFormatter != null &&
        sentFormatter != null &&
        preflightFormatter != sentFormatter) {
      final message = promptFormatterMismatchMessage(
        preflight: preflightFormatter,
        sent: sentFormatter,
      );
      setState(() {
        _running = false;
        _subscription = null;
        _status = message;
      });
      completer.complete(null);
      return;
    }
    final formatterForRecord = sentFormatter ?? preflightFormatter;

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

    // Gemma 4 keeps thought text. Other formatters still strip it, so a Qwen
    // answer stays what it was. Stop markers are cut either way.
    final rawOutput = _outputController.text;
    final thoughtEmitted = outputContainsThought(rawOutput);
    final cleanedOutput = sanitizeModelOutput(
      rawOutput,
      keepThoughts: formatterForRecord == promptFormatterGemma4,
    );
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
      sanityRan: _sanityRan,
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
      promptTokenCount:
          _promptTokenCount ?? _estimatePromptTokens(_messagesController.text),
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
      promptFormatter: formatterForRecord,
      thoughtEmitted: thoughtEmitted,
      adapterFileName: _loadedAdapterFileName ?? '',
      adapterSha256: _loadedAdapterSha256 ?? '',
      adapterScale: _loadedAdapterScale,
    );

    _history.add(record);
    _comparisonResults[condition] = record;
    unawaited(RunPersistenceManager.appendRun(record));

    setState(() {
      _running = false;
      _subscription = null;
      _batteryTempCelsius = batteryTempAfter;
      _lastRunAt = DateTime.now();
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

    var compareIndex = 0;
    for (final condition in conditions) {
      if (!mounted || _stopRequested) break;
      compareIndex++;
      setState(() {
        _batchProgressStatus = 'Compare $compareIndex/${conditions.length}';
      });
      await _runCondition(condition);
      await Future<void>.delayed(const Duration(milliseconds: 300));
    }
    if (mounted) setState(() => _batchProgressStatus = null);
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
          _batchProgressStatus = 'Batch $currentCount/$totalRuns';
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
        _batchProgressStatus = 'Heat $i/$totalRuns';
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
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('No runs to copy yet.')));
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
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('No runs to copy yet.')));
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

  Future<void> _exportFiles({bool openShareSheet = false}) async {
    if (_history.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('No runs to export yet.')));
      return;
    }

    final paths = await BenchmarkExporter.exportToFiles(_history);
    if (!mounted) return;

    if (Platform.isAndroid) {
      await _publishAndroidExports(paths, openShareSheet: openShareSheet);
      return;
    }

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

  Future<void> _publishAndroidExports(
    Map<String, String> paths, {
    required bool openShareSheet,
  }) async {
    final csvPath = paths['csv'];
    final jsonPath = paths['json'];
    if (csvPath == null || jsonPath == null) {
      _showExportSnackBar(
        publicExportFailureMessage(
          'the private export did not return both files',
        ),
      );
      return;
    }
    final cache = File('${File(csvPath).parent.path}/$modelHashCacheFileName');
    try {
      final published = await publishPublicExports(
        publicExportRequests(
          csvPath: csvPath,
          jsonPath: jsonPath,
          hashCachePath: cache.existsSync() ? cache.path : null,
        ),
      );
      _lastPublicExportUris = published.uris;
      if (!mounted) return;
      _showExportSnackBar(
        publicExportSavedMessage(published.directory, published.names),
        shareUris: published.uris,
      );
      if (openShareSheet) await _openShareSheet(published.uris);
    } on PlatformException catch (error) {
      if (!mounted) return;
      _showExportSnackBar(
        publicExportFailureMessage(error.message ?? error.code),
      );
    }
  }

  Future<void> _shareExports() async {
    final uris = _lastPublicExportUris;
    if (uris != null && uris.isNotEmpty) {
      await _openShareSheet(uris);
      return;
    }
    await _exportFiles(openShareSheet: true);
  }

  Future<void> _openShareSheet(List<String> uris) async {
    try {
      await sharePublicExports(uris);
    } on PlatformException catch (error) {
      if (!mounted) return;
      _showExportSnackBar(
        'Could not open the share sheet (${error.message ?? error.code}).',
      );
    }
  }

  void _showExportSnackBar(String message, {List<String>? shareUris}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: const Duration(seconds: 8),
        action: shareUris == null
            ? null
            : SnackBarAction(
                label: 'Share',
                onPressed: () => _openShareSheet(shareUris),
              ),
      ),
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

  String get _messagesHeader {
    final condition = _shownCondition ?? _effectiveCondition;
    final rag = condition.isSimulatedRag ? 'RAG on' : 'RAG off';
    if (_running) return '${condition.code} · $rag · Streaming';
    if (_hasRun && _runIndexSinceLoad > 0) {
      final duration = _generationDuration;
      final seconds = duration == null
          ? ''
          : ' · ${(duration.inMilliseconds / 1000).toStringAsFixed(1)} s';
      return '${condition.code} · $rag · Run $_runIndexSinceLoad$seconds';
    }
    return '${condition.code} · $rag';
  }

  String get _telemetryContext {
    final model = _loaded ? (_modelFileName ?? emDash) : emDash;
    final adapter = !_loaded ? emDash : (_loadedAdapterFileName ?? 'None');
    final condition = (_shownCondition ?? _effectiveCondition).shortCode;
    final backend = _loaded ? _backend : emDash;
    final device = _deviceLabel.isEmpty ? emDash : _deviceLabel;
    return '$model · $adapter · $condition · $backend · $device · '
        '${formatClock(_lastRunAt)}';
  }

  String get _overheadLabel {
    final idle = _ramIdleBytes;
    final before = _ramBeforeLoadBytes;
    if (idle == null || before == null) return emDash;
    return '${((idle - before) / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  String? get _loadSplitLabel {
    final base = _baseLoadMs;
    if (base == null) return null;
    final adapter = _adapterLoadMs ?? 0;
    final total = _loadDurationMs ?? base + adapter;
    if (_loadedAdapterPath == null) return 'Load $total ms';
    return 'Load $total ms (base $base ms, adapter $adapter ms)';
  }

  @override
  Widget build(BuildContext context) {
    final setup = SetupCard(
      modelName: _modelFileName ?? 'No file selected',
      modelSizeLabel: formatFileSizeLabel(_modelSizeBytes),
      statusLine: _modelStatusLine,
      adapterLabel: _adapterFileName ?? 'None',
      canPick: !_running && !_loading,
      canLoad: _canLoad,
      onPickModel: _pickModel,
      onLoad: _loadModel,
      onPickAdapter: _pickAdapter,
      onClearAdapter: _clearAdapter,
      adapterSelected: _adapterPath != null,
      condition: _selectedCondition,
      onCondition: (value) => setState(() => _selectedCondition = value),
      simulatedRag: _simulatedRagEnabled,
      onSimulatedRag: (value) => setState(() => _simulatedRagEnabled = value),
      prompts: _catalog?.prompts ?? const [],
      selectedPromptId: _selected?.id,
      onSelectPrompt: (prompt) {
        setState(() {
          _selected = prompt;
          _comparisonResults.clear();
        });
      },
      running: _running,
      catalogError: _catalogError,
      errorText: _loadError,
    );
    final actions = RunActionsBar(
      running: _running,
      canRun: _canRun,
      blockedReason: _runBlockedReason,
      progressLabel: _batchProgressStatus,
      onRun: () => _runCondition(_effectiveCondition),
      onStop: () => unawaited(_stop()),
      onBatch: () => unawaited(_runBatchProtocol()),
      onCompare: () => unawaited(_runCompareAll()),
      onHeat: () => unawaited(_runHeatTest()),
      compareLabel: _simulatedRagEnabled ? 'Compare all 5' : 'Compare all 3',
    );
    final telemetry = TelemetryCard(
      coldLoadMs: _coldLoadMs,
      idleRamBytes: _ramIdleBytes,
      peakRamBytes: _hasRun ? _peakRssBytes : null,
      ttftMs: _ttftMs,
      tokensPerSecond: _tokensPerSecond,
      tokensGenerated: _evalCount,
      hasRun: _hasRun,
      contextLine: _telemetryContext,
    );
    final config = ConfigCard(
      loaded: _loaded,
      modelFile: _modelFileName ?? emDash,
      architecture: _architecture,
      parameterCount: _parameterCount,
      quantization: _quantization,
      fileSizeLabel: formatFileSizeLabel(_modelSizeBytes),
      backend: _backend,
      threadsLabel: _threads == null ? emDash : '$_threads',
      contextSize: _contextSize,
      adapterLabel: _loadedAdapterFileName ?? 'None',
      temperature: _temperature,
      seed: _seed,
      topK: _topK,
      topP: _topP,
      repeatPenalty: _repeatPenalty,
      maxTokens: _maxTokens,
    );
    final history = HistoryCard(
      prompt: _selected,
      results: _comparisonResults,
      showSimRag: _simulatedRagEnabled,
    );
    final battery = _batteryTempCelsius == null
        ? emDash
        : '${_batteryTempCelsius!.toStringAsFixed(1)} °C';
    final throttle = _heatTestThrottlePercentage == null
        ? emDash
        : '${_heatTestThrottlePercentage!.toStringAsFixed(1)}%';
    final canShare =
        _history.isNotEmpty || (_lastPublicExportUris?.isNotEmpty ?? false);
    final diagnostics = DiagnosticsCard(
      canVerify: _canLoad,
      onVerify: () => unawaited(_verifyModel()),
      skipSanity: _skipSanityEffective,
      onSkipSanity: Platform.isAndroid || _running || _loading
          ? null
          : (value) => setState(() => _skipSanityRequested = value),
      kernelLine: _kernelLine,
      statusMessage: _status,
      overheadLabel: _overheadLabel,
      batteryLabel: battery,
      throttleLabel: throttle,
      loadSplitLabel: _loadSplitLabel,
      canExport: _history.isNotEmpty,
      onCopyCsv: _copyCsvToClipboard,
      onCopyJson: _copyJsonToClipboard,
      onExport: () => unawaited(_exportFiles()),
      onClearHistory: _clearHistory,
      showShare: Platform.isAndroid,
      onShare: canShare ? () => unawaited(_shareExports()) : null,
    );

    Widget messagesCard() {
      return ListenableBuilder(
        listenable: _outputController,
        builder: (context, _) {
          return MessagesCard(
            header: _messagesHeader,
            messages: _sentMessages,
            output: _outputController.text,
            streaming: _running,
            hasRun: _hasRun,
          );
        },
      );
    }

    const gap = SizedBox(height: reportGap);

    return Scaffold(
      appBar: AppBar(title: const Text('Visitor note & Benchmark')),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final width = constraints.maxWidth;
            final wide = width >= 840;
            final columnWidth = wide
                ? math.min(width, 1200).toDouble()
                : (width >= 600 ? math.min(width, 720).toDouble() : width);
            final children = wide
                ? <Widget>[
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          flex: 4,
                          child: Column(
                            children: [setup, gap, actions, gap, config],
                          ),
                        ),
                        const SizedBox(width: reportGap),
                        Expanded(
                          flex: 8,
                          child: Column(
                            children: [
                              telemetry,
                              gap,
                              messagesCard(),
                              gap,
                              history,
                            ],
                          ),
                        ),
                      ],
                    ),
                    gap,
                    diagnostics,
                  ]
                : <Widget>[
                    setup,
                    gap,
                    actions,
                    gap,
                    messagesCard(),
                    gap,
                    telemetry,
                    gap,
                    config,
                    gap,
                    history,
                    gap,
                    diagnostics,
                  ];
            return Align(
              alignment: Alignment.topCenter,
              child: SizedBox(
                width: columnWidth,
                child: ListView(
                  padding: const EdgeInsets.all(reportGap),
                  children: children,
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}
