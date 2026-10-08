import 'dart:convert';
import 'dart:io';

/// Captures a snapshot of current and peak process memory, as well as
/// system memory on Linux/Android.
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

  /// Takes an immediate snapshot of process and system memory.
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
          final availKb = _parseKb(meminfo, 'MemAvailable') ?? _parseKb(meminfo, 'MemFree');
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
      } catch (_) {
        // Fall back to dart:io ProcessInfo
      }
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

/// Attempts to read the device battery or SoC temperature on Android.
/// Returns temperature in degrees Celsius, or null if inaccessible.
double? readDeviceTemperatureCelsius() {
  if (!Platform.isAndroid && !Platform.isLinux) {
    return null;
  }
  const candidatePaths = [
    '/sys/class/power_supply/battery/temp', // often 1/10 degree C (e.g. 345 = 34.5 C)
    '/sys/class/thermal/thermal_zone0/temp', // often millidegree C (e.g. 34500 = 34.5 C)
    '/sys/devices/virtual/thermal/thermal_zone0/temp',
  ];
  for (final path in candidatePaths) {
    try {
      final file = File(path);
      if (file.existsSync()) {
        final text = file.readAsStringSync().trim();
        final raw = double.tryParse(text);
        if (raw != null && raw > 0) {
          if (raw > 1000) {
            return raw / 1000.0;
          } else if (raw > 100) {
            return raw / 10.0;
          }
          return raw;
        }
      }
    } catch (_) {}
  }
  return null;
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

/// A single recorded prompt run with full telemetry.
class RunRecord {
  RunRecord({
    required this.promptId,
    required this.promptText,
    required this.conditionLabel,
    required this.relevantOnly,
    required this.maxTokens,
    required this.output,
    required this.timestamp,
    this.coldLoadMs,
    this.ttftMs,
    this.tokensPerSecond,
    this.evalCount,
    this.generationDuration,
    this.peakRssBytes,
    this.batteryTempCelsius,
  });

  final String promptId;
  final String promptText;
  final String conditionLabel;
  final bool relevantOnly;
  final int maxTokens;
  final String output;
  final DateTime timestamp;
  final int? coldLoadMs;
  final int? ttftMs;
  final double? tokensPerSecond;
  final int? evalCount;
  final Duration? generationDuration;
  final int? peakRssBytes;
  final double? batteryTempCelsius;

  Map<String, dynamic> toJson() => {
    'promptId': promptId,
    'promptText': promptText,
    'condition': conditionLabel,
    'relevantOnly': relevantOnly,
    'maxTokens': maxTokens,
    'coldLoadMs': coldLoadMs,
    'ttftMs': ttftMs,
    'tokensPerSecond': tokensPerSecond,
    'evalCount': evalCount,
    'generationDurationMs': generationDuration?.inMilliseconds,
    'peakRssBytes': peakRssBytes,
    'batteryTempCelsius': batteryTempCelsius,
    'timestamp': timestamp.toIso8601String(),
    'output': output,
  };
}

/// Generates structured Markdown and JSON reports for benchmark export.
class BenchmarkExporter {
  static String exportMarkdownReport({
    required String? modelFileName,
    required int? coldLoadMs,
    required int? baselineRssBytes,
    required int? idleModelRssBytes,
    required List<RunRecord> runs,
  }) {
    final buffer = StringBuffer();
    buffer.writeln('# Technical Exploration Benchmark Report');
    buffer.writeln();
    buffer.writeln('**Device OS:** ${Platform.operatingSystem} (${Platform.operatingSystemVersion})');
    buffer.writeln('**Model:** ${modelFileName ?? "Unknown"}');
    buffer.writeln('**Cold Load Time:** ${coldLoadMs != null ? "$coldLoadMs ms" : "--"}');
    buffer.writeln('**Baseline App RAM:** ${formatBytes(baselineRssBytes)}');
    buffer.writeln('**Model Idle RAM Overhead:** ${formatBytes(idleModelRssBytes, signed: true)}');
    buffer.writeln();

    if (runs.isEmpty) {
      buffer.writeln('_No runs recorded yet._');
      return buffer.toString();
    }

    buffer.writeln('## Recorded Runs (${runs.length})');
    buffer.writeln();
    buffer.writeln('| # | Condition | Prompt ID | TTFT | TPS | Tokens | Gen Time | Peak RAM | Temp |');
    buffer.writeln('|---|-----------|-----------|------|-----|--------|----------|----------|------|');

    for (var i = 0; i < runs.length; i++) {
      final r = runs[i];
      final ttft = r.ttftMs != null ? '${r.ttftMs} ms' : '--';
      final tps = r.tokensPerSecond != null ? r.tokensPerSecond!.toStringAsFixed(2) : '--';
      final tokens = r.evalCount?.toString() ?? '--';
      final genTime = r.generationDuration != null ? '${r.generationDuration!.inMilliseconds} ms' : '--';
      final ram = formatBytes(r.peakRssBytes);
      final temp = r.batteryTempCelsius != null ? '${r.batteryTempCelsius!.toStringAsFixed(1)}°C' : '--';

      buffer.writeln('| ${i + 1} | ${r.conditionLabel} | ${r.promptId} | $ttft | $tps | $tokens | $genTime | $ram | $temp |');
    }

    buffer.writeln();
    buffer.writeln('### Detailed Outputs');
    buffer.writeln();
    for (var i = 0; i < runs.length; i++) {
      final r = runs[i];
      buffer.writeln('#### Run ${i + 1}: [${r.conditionLabel}] ${r.promptId}');
      buffer.writeln('> **Prompt:** ${r.promptText}');
      buffer.writeln();
      buffer.writeln('```');
      buffer.writeln(r.output.trim());
      buffer.writeln('```');
      buffer.writeln();
    }

    return buffer.toString();
  }

  static String exportJsonReport({
    required String? modelFileName,
    required int? coldLoadMs,
    required int? baselineRssBytes,
    required int? idleModelRssBytes,
    required List<RunRecord> runs,
  }) {
    final data = {
      'system': {
        'os': Platform.operatingSystem,
        'osVersion': Platform.operatingSystemVersion,
      },
      'model': {
        'name': modelFileName,
        'coldLoadMs': coldLoadMs,
        'baselineRssBytes': baselineRssBytes,
        'idleModelRssBytes': idleModelRssBytes,
      },
      'runs': runs.map((r) => r.toJson()).toList(),
    };
    return const JsonEncoder.withIndent('  ').convert(data);
  }
}
