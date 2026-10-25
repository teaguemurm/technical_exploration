import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:technical_exploration/device_probe.dart';

/// Captures a snapshot of current and peak process memory.
///
/// Documented measurement method:
/// - Process RSS: On Linux & Android, read from `/proc/self/status` `VmRSS`
///   (instantaneous physical memory) and `VmHWM` (High Water Mark / peak memory).
///   Falls back to `ProcessInfo.currentRss` and `ProcessInfo.maxRss`.
/// - On macOS and iOS, read directly via `ProcessInfo.currentRss` and `ProcessInfo.maxRss`.
class MemorySnapshot {
  const MemorySnapshot({
    required this.currentRssBytes,
    required this.maxRssBytes,
    this.systemTotalBytes,
    this.systemAvailableBytes,
  });

  final int currentRssBytes;
  final int maxRssBytes;
  final int? systemTotalBytes;
  final int? systemAvailableBytes;

  static MemorySnapshot capture() {
    var current = ProcessInfo.currentRss;
    var max = ProcessInfo.maxRss;
    int? sysTotal;
    int? sysAvail;

    if (Platform.isAndroid || Platform.isLinux) {
      try {
        final meminfo = _readProcFile('/proc/meminfo');
        if (meminfo != null) {
          final totalKb = _parseKb(meminfo, 'MemTotal');
          final availKb =
              _parseKb(meminfo, 'MemAvailable') ?? _parseKb(meminfo, 'MemFree');
          if (totalKb != null) sysTotal = totalKb * 1024;
          if (availKb != null) sysAvail = availKb * 1024;
        }

        final status = _readProcFile('/proc/self/status');
        if (status != null) {
          final vmRssKb = _parseKb(status, 'VmRSS');
          final vmHwmKb = _parseKb(status, 'VmHWM');
          if (vmRssKb != null && vmRssKb > 0) current = vmRssKb * 1024;
          if (vmHwmKb != null && vmHwmKb > 0) max = vmHwmKb * 1024;
        }
      } catch (_) {}
    }

    return MemorySnapshot(
      currentRssBytes: current,
      maxRssBytes: max,
      systemTotalBytes: sysTotal,
      systemAvailableBytes: sysAvail,
    );
  }

  static String? _readProcFile(String path) {
    try {
      final file = File(path);
      if (file.existsSync()) {
        return file.readAsStringSync();
      }
    } catch (_) {}
    return null;
  }

  static int? _parseKb(String content, String key) {
    final pattern = RegExp('^$key:\\s+(\\d+)', multiLine: true);
    final match = pattern.firstMatch(content);
    if (match != null) {
      return int.tryParse(match.group(1) ?? '');
    }
    return null;
  }
}

/// Reads the device model name (e.g. "Fairphone 5", "Mac14,6").
///
/// On Android this prefers the marketing name plus SKU from [Build] and
/// `ro.product.marketname`, cached by [refreshDeviceIdentity]. `getprop` is
/// only the fallback. `/system/build.prop` is not read: Treble can name a
/// reference device there.
String readDeviceModel() {
  if (Platform.isAndroid) {
    final cached = cachedAndroidDeviceLabel();
    if (cached != null && cached.isNotEmpty) return cached;
    try {
      final res = Process.runSync('getprop', ['ro.product.model']);
      if (res.exitCode == 0 && res.stdout.toString().trim().isNotEmpty) {
        final model = res.stdout.toString().trim();
        final brandRes = Process.runSync('getprop', ['ro.product.brand']);
        final brand = brandRes.stdout.toString().trim();
        return formatAndroidDeviceLabel(manufacturer: brand, model: model);
      }
    } catch (_) {}
    return 'Android Device';
  } else if (Platform.isMacOS) {
    try {
      final res = Process.runSync('sysctl', ['-n', 'hw.model']);
      if (res.exitCode == 0 && res.stdout.toString().trim().isNotEmpty) {
        return res.stdout.toString().trim();
      }
    } catch (_) {}
    return 'macOS (${Platform.localHostname})';
  } else if (Platform.isLinux) {
    return 'Linux Device';
  }
  return Platform.operatingSystem;
}

/// Formats a byte count into a human-readable string (e.g., "1.42 GB", "420 MB").
String formatBytes(int? bytes, {bool signed = false}) {
  if (bytes == null) {
    return '--';
  }
  final isNegative = bytes < 0;
  final abs = bytes.abs();
  final double value;
  final String unit;

  if (abs >= 1024 * 1024 * 1024) {
    value = abs / (1024.0 * 1024 * 1024);
    unit = 'GB';
  } else if (abs >= 1024 * 1024) {
    value = abs / (1024.0 * 1024);
    unit = 'MB';
  } else if (abs >= 1024) {
    value = abs / 1024.0;
    unit = 'KB';
  } else {
    value = abs.toDouble();
    unit = 'B';
  }

  final formatted = value.toStringAsFixed(2);
  if (signed) {
    final prefix = isNegative ? '-' : '+';
    return '$prefix$formatted $unit';
  }
  return '${isNegative ? '-' : ''}$formatted $unit';
}

/// Fixed decoding settings used across the reproducible protocol.
class DecodingSettings {
  const DecodingSettings({
    this.temperature = 0.0, // Protocol default: greedy decoding
    this.seed = 42, // Fixed reproducible seed
    this.topK = 40,
    this.topP = 0.9,
    this.repeatPenalty = 1.1,
    this.maxTokens = 256,
    this.contextSize = 2048,
  });

  final double temperature;
  final int? seed;
  final int topK;
  final double topP;
  final double? repeatPenalty;
  final int maxTokens;
  final int contextSize;

  Map<String, dynamic> toJson() => {
    'temperature': temperature,
    'seed': seed,
    'topK': topK,
    'topP': topP,
    'repeatPenalty': repeatPenalty,
    'maxTokens': maxTokens,
    'contextSize': contextSize,
  };

  factory DecodingSettings.fromJson(Map<String, dynamic> json) {
    return DecodingSettings(
      temperature: (json['temperature'] as num?)?.toDouble() ?? 0.0,
      seed: (json['seed'] as num?)?.toInt() ?? 42,
      topK: (json['topK'] as num?)?.toInt() ?? 40,
      topP: (json['topP'] as num?)?.toDouble() ?? 0.9,
      repeatPenalty: (json['repeatPenalty'] as num?)?.toDouble() ?? 1.1,
      maxTokens: (json['maxTokens'] as num?)?.toInt() ?? 256,
      contextSize: (json['contextSize'] as num?)?.toInt() ?? 2048,
    );
  }
}

/// A complete, persistent telemetry record of a single inference run.
class RunRecord {
  RunRecord({
    required this.runIndex,
    required this.runIndexSinceLoad,
    required this.isColdLoad,
    required this.timestamp,
    required this.deviceModel,
    required this.osVersion,
    required this.modelFileName,
    required this.architecture,
    required this.quantization,
    required this.backend,
    required this.nGpuLayers,
    required this.condition,
    required this.notesMode,
    required this.k,
    required this.injectedNoteIds,
    required this.promptId,
    required this.promptText,
    required this.decodingSettings,
    required this.systemPrompt,
    required this.messagesSent,
    required this.output,
    required this.hitMaxTokens,
    this.loadDurationMs,
    this.ramBeforeLoadBytes,
    this.ramIdleBytes,
    this.ramPeakBytes,
    this.promptTokenCount,
    this.generatedTokenCount,
    this.ttftMs,
    this.generationDurationMs,
    this.tokensPerSecond,
    this.batteryLevelBefore,
    this.batteryLevelAfter,
    this.batteryTempBeforeCelsius,
    this.batteryTempAfterCelsius,
    this.throttlePercentage,
    this.isHeatTest = false,
    this.thinkingDisabled,
    this.promptFormatter,
    this.thoughtEmitted = false,
    this.adapterFileName = '',
    this.adapterSha256 = '',
    this.adapterScale,
  });

  final int runIndex;
  final int runIndexSinceLoad;
  final bool isColdLoad;
  final DateTime timestamp;
  final String deviceModel;
  final String osVersion;
  final String modelFileName;
  final String architecture;
  final String quantization;
  final String backend;
  final int nGpuLayers;

  final String condition; // 'A', 'B', 'C', 'B-sim-RAG', 'C-sim-RAG'
  final String notesMode; // 'none', 'all', 'sim-rag'
  final int k; // 0 for A/B/C, 2 for sim-RAG
  final List<String> injectedNoteIds;

  final String promptId;
  final String promptText;
  final DecodingSettings decodingSettings;
  final String systemPrompt;
  final String messagesSent;
  final String output;
  final bool hitMaxTokens;

  final int? loadDurationMs;
  final int? ramBeforeLoadBytes;
  final int? ramIdleBytes;
  final int? ramPeakBytes;

  final int? promptTokenCount;
  final int? generatedTokenCount;
  final int? ttftMs;
  final int? generationDurationMs;
  final double? tokensPerSecond;

  final int? batteryLevelBefore;
  final int? batteryLevelAfter;
  final double? batteryTempBeforeCelsius;
  final double? batteryTempAfterCelsius;

  final double? throttlePercentage; // populated only during heat test
  final bool isHeatTest;
  final bool? thinkingDisabled;

  /// `native`, `gemma4`, or `chatml-fallback`. Null on runs recorded before
  /// the formatter was logged.
  final String? promptFormatter;

  /// True when the model output contained a thought block. Independent of
  /// [thinkingDisabled], which stays null because this app applies no toggle.
  final bool thoughtEmitted;

  /// Empty when the run used the base model. Trailing export fields.
  final String adapterFileName;
  final String adapterSha256;

  /// Null for a base run. `1.0` when an adapter was loaded.
  final double? adapterScale;

  String get loadType => isColdLoad ? 'Cold' : 'Warm';

  Map<String, dynamic> toJson() => {
    'runIndex': runIndex,
    'runIndexSinceLoad': runIndexSinceLoad,
    'isColdLoad': isColdLoad,
    'loadType': loadType,
    'loadDurationMs': loadDurationMs,
    'timestamp': timestamp.toIso8601String(),
    'deviceModel': deviceModel,
    'osVersion': osVersion,
    'modelFileName': modelFileName,
    'architecture': architecture,
    'quantization': quantization,
    'backend': backend,
    'nGpuLayers': nGpuLayers,
    'condition': condition,
    'notesMode': notesMode,
    'k': k,
    'injectedNoteIds': injectedNoteIds,
    'promptId': promptId,
    'promptText': promptText,
    'decodingSettings': decodingSettings.toJson(),
    'systemPrompt': systemPrompt,
    'messagesSent': messagesSent,
    'output': output,
    'hitMaxTokens': hitMaxTokens,
    'ramBeforeLoadBytes': ramBeforeLoadBytes,
    'ramIdleBytes': ramIdleBytes,
    'ramPeakBytes': ramPeakBytes,
    'promptTokenCount': promptTokenCount,
    'generatedTokenCount': generatedTokenCount,
    'ttftMs': ttftMs,
    'generationDurationMs': generationDurationMs,
    'tokensPerSecond': tokensPerSecond,
    'batteryLevelBefore': batteryLevelBefore,
    'batteryLevelAfter': batteryLevelAfter,
    'batteryTempBeforeCelsius': batteryTempBeforeCelsius,
    'batteryTempAfterCelsius': batteryTempAfterCelsius,
    'throttlePercentage': throttlePercentage,
    'isHeatTest': isHeatTest,
    'thinkingDisabled': thinkingDisabled,
    'promptFormatter': promptFormatter,
    'thoughtEmitted': thoughtEmitted,
    'adapterFileName': adapterFileName,
    'adapterSha256': adapterSha256,
    'adapterScale': adapterScale,
  };

  factory RunRecord.fromJson(Map<String, dynamic> json) {
    return RunRecord(
      runIndex: (json['runIndex'] as num?)?.toInt() ?? 0,
      runIndexSinceLoad: (json['runIndexSinceLoad'] as num?)?.toInt() ?? 1,
      isColdLoad:
          (json['loadType'] == 'Cold') ||
          (json['isColdLoad'] as bool? ?? false),
      loadDurationMs: (json['loadDurationMs'] as num?)?.toInt(),
      timestamp:
          DateTime.tryParse(json['timestamp']?.toString() ?? '') ??
          DateTime.now(),
      deviceModel: json['deviceModel']?.toString() ?? 'Unknown',
      osVersion: json['osVersion']?.toString() ?? 'Unknown',
      modelFileName: json['modelFileName']?.toString() ?? 'Unknown',
      architecture: json['architecture']?.toString() ?? 'unknown',
      quantization: json['quantization']?.toString() ?? 'unknown',
      backend: json['backend']?.toString() ?? 'CPU',
      nGpuLayers: (json['nGpuLayers'] as num?)?.toInt() ?? 0,
      condition: json['condition']?.toString() ?? 'A',
      notesMode: json['notesMode']?.toString() ?? 'none',
      k: (json['k'] as num?)?.toInt() ?? 0,
      injectedNoteIds:
          (json['injectedNoteIds'] as List?)
              ?.map((e) => e.toString())
              .toList() ??
          const [],
      promptId: json['promptId']?.toString() ?? '',
      promptText: json['promptText']?.toString() ?? '',
      decodingSettings: json['decodingSettings'] is Map<String, dynamic>
          ? DecodingSettings.fromJson(json['decodingSettings'])
          : const DecodingSettings(),
      systemPrompt: json['systemPrompt']?.toString() ?? '',
      messagesSent: json['messagesSent']?.toString() ?? '',
      output: json['output']?.toString() ?? '',
      hitMaxTokens: json['hitMaxTokens'] as bool? ?? false,
      ramBeforeLoadBytes: (json['ramBeforeLoadBytes'] as num?)?.toInt(),
      ramIdleBytes: (json['ramIdleBytes'] as num?)?.toInt(),
      ramPeakBytes: (json['ramPeakBytes'] as num?)?.toInt(),
      promptTokenCount: (json['promptTokenCount'] as num?)?.toInt(),
      generatedTokenCount: (json['generatedTokenCount'] as num?)?.toInt(),
      ttftMs: (json['ttftMs'] as num?)?.toInt(),
      generationDurationMs: (json['generationDurationMs'] as num?)?.toInt(),
      tokensPerSecond: (json['tokensPerSecond'] as num?)?.toDouble(),
      batteryLevelBefore: (json['batteryLevelBefore'] as num?)?.toInt(),
      batteryLevelAfter: (json['batteryLevelAfter'] as num?)?.toInt(),
      batteryTempBeforeCelsius: (json['batteryTempBeforeCelsius'] as num?)
          ?.toDouble(),
      batteryTempAfterCelsius: (json['batteryTempAfterCelsius'] as num?)
          ?.toDouble(),
      throttlePercentage: (json['throttlePercentage'] as num?)?.toDouble(),
      isHeatTest: json['isHeatTest'] as bool? ?? false,
      thinkingDisabled: json['thinkingDisabled'] as bool?,
      promptFormatter: json['promptFormatter'] as String?,
      thoughtEmitted: json['thoughtEmitted'] as bool? ?? false,
      adapterFileName: json['adapterFileName']?.toString() ?? '',
      adapterSha256: json['adapterSha256']?.toString() ?? '',
      adapterScale: (json['adapterScale'] as num?)?.toDouble(),
    );
  }
}

/// Manages persistence to disk across app sessions.
///
/// Runs are `runs_history.jsonl` in the application documents directory
/// (`context.getDir("flutter")` on Android). Installing a new APK over this
/// one keeps that directory when the package id and signing key stay the same
/// and the app is not uninstalled. Uninstall deletes it. Copies in
/// Download/technical-exploration are exports, not this history.
class RunPersistenceManager {
  static const _fileName = 'runs_history.jsonl';

  static Future<File> _getFile() async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/$_fileName');
  }

  /// Appends a new run to disk immediately.
  static Future<void> appendRun(RunRecord record) async {
    try {
      final file = await _getFile();
      final line = '${jsonEncode(record.toJson())}\n';
      await file.writeAsString(line, mode: FileMode.append, flush: true);
    } catch (e) {
      // ignore: avoid_print
      print('[RunPersistenceManager] Failed to persist run: $e');
    }
  }

  /// Loads all saved runs from disk, seamlessly merging any legacy sandboxed runs.
  static Future<List<RunRecord>> loadRuns() async {
    try {
      final file = await _getFile();
      final records = <RunRecord>[];
      if (await file.exists()) {
        final lines = await file.readAsLines();
        for (final line in lines) {
          if (line.trim().isEmpty) continue;
          try {
            final map = jsonDecode(line) as Map<String, dynamic>;
            records.add(RunRecord.fromJson(map));
          } catch (_) {}
        }
      }

      // Check legacy sandboxed container path on macOS
      if (Platform.isMacOS) {
        final home = Platform.environment['HOME'];
        if (home != null && home.isNotEmpty) {
          final sandboxFile = File(
            '$home/Library/Containers/com.example.technicalExploration/Data/Documents/$_fileName',
          );
          if (sandboxFile.path != file.path && await sandboxFile.exists()) {
            final existingKeys = {
              for (final r in records)
                '${r.timestamp.toIso8601String()}_${r.promptId}_${r.condition}',
            };
            var migratedCount = 0;
            final sandboxLines = await sandboxFile.readAsLines();
            for (final line in sandboxLines) {
              if (line.trim().isEmpty) continue;
              try {
                final map = jsonDecode(line) as Map<String, dynamic>;
                final rec = RunRecord.fromJson(map);
                final key =
                    '${rec.timestamp.toIso8601String()}_${rec.promptId}_${rec.condition}';
                if (!existingKeys.contains(key)) {
                  records.add(rec);
                  existingKeys.add(key);
                  migratedCount++;
                }
              } catch (_) {}
            }
            if (migratedCount > 0) {
              records.sort((a, b) => a.timestamp.compareTo(b.timestamp));
              // Rewrite primary file with merged history
              final buffer = StringBuffer();
              for (final r in records) {
                buffer.writeln(jsonEncode(r.toJson()));
              }
              await file.writeAsString(buffer.toString(), flush: true);
            }
          }
        }
      }

      return records;
    } catch (_) {
      return [];
    }
  }

  /// Clears persisted runs.
  static Future<void> clearRuns() async {
    try {
      final file = await _getFile();
      if (await file.exists()) {
        await file.delete();
      }
    } catch (_) {}
  }
}

/// Generates CSV and JSON benchmark exports matching the protocol sheets:
/// - "Load and idle" tab
/// - "Inference" tab
class BenchmarkExporter {
  /// Exports CSV aligning directly with the measurements sheet's tabs.
  static String exportCsv(List<RunRecord> runs) {
    final buffer = StringBuffer();

    // Section 1: Load and Idle Summary
    buffer.writeln('# Tab: Load and idle');
    buffer.writeln(
      'Device Model,OS Version,Model File,Architecture,Quantization,Backend,GPU Layers,Load Type,Load Time (ms),RAM Before Load (MB),RAM Idle Model (MB),RAM Overhead (MB),Timestamp,Adapter File,Adapter SHA-256,Adapter Scale',
    );
    final seenLoads = <String>{};
    for (final r in runs) {
      final key =
          '${r.modelFileName}_${r.adapterFileName}_${r.isColdLoad}_${r.loadDurationMs}';
      if (!seenLoads.contains(key) && r.loadDurationMs != null) {
        seenLoads.add(key);
        final beforeMb = r.ramBeforeLoadBytes != null
            ? (r.ramBeforeLoadBytes! / 1024 / 1024).toStringAsFixed(1)
            : '';
        final idleMb = r.ramIdleBytes != null
            ? (r.ramIdleBytes! / 1024 / 1024).toStringAsFixed(1)
            : '';
        final overheadMb =
            (r.ramIdleBytes != null && r.ramBeforeLoadBytes != null)
            ? ((r.ramIdleBytes! - r.ramBeforeLoadBytes!) / 1024 / 1024)
                  .toStringAsFixed(1)
            : '';
        buffer.writeln(
          '${_escape(r.deviceModel)},${_escape(r.osVersion)},${_escape(r.modelFileName)},${_escape(r.architecture)},${_escape(r.quantization)},${_escape(r.backend)},${r.nGpuLayers},${r.loadType},${r.loadDurationMs ?? ""},$beforeMb,$idleMb,$overheadMb,${r.timestamp.toIso8601String()},${_escape(r.adapterFileName)},${_escape(r.adapterSha256)},${_scaleCell(r.adapterScale)}',
        );
      }
    }

    buffer.writeln();

    // Section 2: Inference Runs
    buffer.writeln('# Tab: Inference');
    buffer.writeln(
      'Run Index,Run Since Load,Load Type,Timestamp,Device Model,OS Version,Model File,Architecture,Quantization,Backend,Condition,Notes Mode,k,Injected Notes,Prompt ID,Prompt Text,Temperature,Seed,Top-K,Top-P,Repeat Penalty,Max Tokens,Context Size,Prompt Tokens,Gen Tokens,Hit Max Tokens,TTFT (ms),Gen Duration (ms),TPS (tok/s),RAM Peak (MB),Battery Level Before (%),Battery Level After (%),Battery Temp Before (C),Battery Temp After (C),Throttle (%),Is Heat Test,Thinking Disabled,Prompt Formatter,Thought Emitted,Output,Adapter File,Adapter SHA-256,Adapter Scale',
    );

    for (final r in runs) {
      final peakMb = r.ramPeakBytes != null
          ? (r.ramPeakBytes! / 1024 / 1024).toStringAsFixed(1)
          : '';
      final tps = r.tokensPerSecond != null
          ? r.tokensPerSecond!.toStringAsFixed(2)
          : '';
      final throttle = r.throttlePercentage != null
          ? r.throttlePercentage!.toStringAsFixed(1)
          : '';
      final tempB = r.batteryTempBeforeCelsius != null
          ? r.batteryTempBeforeCelsius!.toStringAsFixed(1)
          : '';
      final tempA = r.batteryTempAfterCelsius != null
          ? r.batteryTempAfterCelsius!.toStringAsFixed(1)
          : '';

      buffer.writeln(
        [
          r.runIndex,
          r.runIndexSinceLoad,
          r.loadType,
          r.timestamp.toIso8601String(),
          _escape(r.deviceModel),
          _escape(r.osVersion),
          _escape(r.modelFileName),
          _escape(r.architecture),
          _escape(r.quantization),
          _escape(r.backend),
          _escape(r.condition),
          _escape(r.notesMode),
          r.k,
          _escape(r.injectedNoteIds.join('; ')),
          _escape(r.promptId),
          _escape(r.promptText),
          r.decodingSettings.temperature,
          r.decodingSettings.seed ?? '',
          r.decodingSettings.topK,
          r.decodingSettings.topP,
          r.decodingSettings.repeatPenalty ?? '',
          r.decodingSettings.maxTokens,
          r.decodingSettings.contextSize,
          r.promptTokenCount ?? '',
          r.generatedTokenCount ?? '',
          r.hitMaxTokens ? 'Yes' : 'No',
          r.ttftMs ?? '',
          r.generationDurationMs ?? '',
          tps,
          peakMb,
          r.batteryLevelBefore ?? '',
          r.batteryLevelAfter ?? '',
          tempB,
          tempA,
          throttle,
          r.isHeatTest ? 'Yes' : 'No',
          r.thinkingDisabled == null
              ? 'N/A'
              : (r.thinkingDisabled! ? 'Yes' : 'No'),
          _escape(r.promptFormatter ?? ''),
          r.thoughtEmitted ? 'Yes' : 'No',
          _escape(r.output),
          _escape(r.adapterFileName),
          _escape(r.adapterSha256),
          _scaleCell(r.adapterScale),
        ].join(','),
      );
    }

    return buffer.toString();
  }

  /// Exports complete JSON format.
  static String exportJson(List<RunRecord> runs) {
    const encoder = JsonEncoder.withIndent('  ');
    return encoder.convert(runs.map((r) => r.toJson()).toList());
  }

  /// Resolves the exports directory: saves locally to /exports in the project
  /// root or working directory, falling back to app documents /exports on mobile.
  static Future<Directory> getExportsDirectory() async {
    // 1. Locate project root by walking up from the running executable
    try {
      final execPath = Platform.resolvedExecutable;
      if (execPath.isNotEmpty) {
        var dir = File(execPath).parent;
        while (dir.parent.path != dir.path) {
          if (File('${dir.path}/pubspec.yaml').existsSync()) {
            final exportsDir = Directory('${dir.path}/exports');
            if (!exportsDir.existsSync()) {
              exportsDir.createSync(recursive: true);
            }
            return exportsDir;
          }
          dir = dir.parent;
        }
      }
    } catch (_) {}

    // 2. Locate project root by walking up from current working directory
    try {
      var dir = Directory.current;
      while (dir.parent.path != dir.path) {
        if (File('${dir.path}/pubspec.yaml').existsSync()) {
          final exportsDir = Directory('${dir.path}/exports');
          if (!exportsDir.existsSync()) {
            exportsDir.createSync(recursive: true);
          }
          return exportsDir;
        }
        dir = dir.parent;
      }
    } catch (_) {}

    // 3. Try relative 'exports' directory if writable
    try {
      final localDir = Directory('exports');
      if (!localDir.existsSync()) {
        localDir.createSync(recursive: true);
      }
      final testFile = File('${localDir.path}/.test_write');
      testFile.writeAsStringSync('test');
      testFile.deleteSync();
      return localDir;
    } catch (_) {}

    // 4. Fallback for sandboxed mobile devices (e.g. Android)
    final docDir = await getApplicationDocumentsDirectory();
    final fallbackDir = Directory('${docDir.path}/exports');
    if (!fallbackDir.existsSync()) {
      await fallbackDir.create(recursive: true);
    }
    return fallbackDir;
  }

  /// Writes exports to files in the /exports directory and returns file paths.
  static Future<Map<String, String>> exportToFiles(
    List<RunRecord> runs, {
    Directory? targetDir,
  }) async {
    final dir = targetDir ?? await getExportsDirectory();
    if (!dir.existsSync()) {
      await dir.create(recursive: true);
    }

    final now = DateTime.now();
    final timestamp =
        '${now.year.toString().padLeft(4, '0')}${now.month.toString().padLeft(2, '0')}${now.day.toString().padLeft(2, '0')}_'
        '${now.hour.toString().padLeft(2, '0')}${now.minute.toString().padLeft(2, '0')}${now.second.toString().padLeft(2, '0')}';
    final csvFile = File(
      '${dir.path}/technical_exploration_runs_$timestamp.csv',
    );
    final jsonFile = File(
      '${dir.path}/technical_exploration_runs_$timestamp.json',
    );

    await csvFile.writeAsString(exportCsv(runs), flush: true);
    await jsonFile.writeAsString(exportJson(runs), flush: true);

    return {'csv': csvFile.path, 'json': jsonFile.path};
  }

  static String _scaleCell(double? scale) =>
      scale == null ? '' : scale.toStringAsFixed(1);

  static String _escape(String value) {
    if (value.contains(',') ||
        value.contains('\n') ||
        value.contains('"') ||
        value.contains('\r')) {
      return '"${value.replaceAll('"', '""')}"';
    }
    return value;
  }
}
