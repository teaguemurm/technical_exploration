import 'package:flutter_test/flutter_test.dart';
import 'package:technical_exploration/device_probe.dart';
import 'package:technical_exploration/kernel_diag.dart';
import 'package:technical_exploration/model_integrity.dart';
import 'package:technical_exploration/run_flags.dart';

void main() {
  test('android label uses the marketing name and the SKU', () {
    expect(
      formatAndroidDeviceLabel(
        manufacturer: 'samsung',
        model: 'SM-F976B',
        marketName: 'Galaxy Z Fold8 Ultra',
      ),
      'Samsung Galaxy Z Fold8 Ultra (SM-F976B)',
    );
    expect(
      formatAndroidDeviceLabel(
        manufacturer: 'samsung',
        model: 'SM-F976B',
      ),
      'Samsung SM-F976B',
    );
  });

  test('battery tenths become celsius and the legacy zone stays out of that value', () {
    expect(celsiusFromBatteryTenths(352), 35.2);
    expect(celsiusFromBatteryTenths(null), isNull);
    expect(batteryPercentFromLevel(level: 70, scale: 100), 70);

    final thermal = thermalFromChannelMap({
      'legacy': {
        'path': '/sys/class/thermal/thermal_zone0/temp',
        'type': 'cpu-0-0-usr',
        'raw': '92300',
      },
      'zones': [
        {'name': 'thermal_zone0', 'type': 'cpu-0-0-usr', 'raw': '92300'},
        {'name': 'thermal_zone3', 'type': 'battery', 'raw': '35200'},
      ],
    });
    expect(thermal.legacyCelsius, closeTo(92.3, 0.01));
    expect(thermal.logLine, contains('thermal_zone0'));
    expect(thermal.logLine, contains('cpu-0-0-usr'));
    expect(thermal.logLine, contains('92.3'));
    expect(thermal.logLine, contains('hottest=thermal_zone0'));

    final battery = batteryFromChannelMap({
      'tempTenths': 352,
      'level': 80,
      'scale': 100,
    });
    expect(battery.tempCelsius, 35.2);
    expect(battery.levelPercent, 80);
  });

  test('cold load is only the first run, and thinking stays null', () {
    expect(
      isColdForThisRun(loadWasCold: true, runIndexSinceLoad: 1),
      isTrue,
    );
    expect(
      isColdForThisRun(loadWasCold: true, runIndexSinceLoad: 2),
      isFalse,
    );
    expect(
      isColdForThisRun(loadWasCold: true, runIndexSinceLoad: 50),
      isFalse,
    );
    expect(
      isColdForThisRun(loadWasCold: false, runIndexSinceLoad: 1),
      isFalse,
    );
    expect(thinkingDisabledForRun(toggleApplied: false), isNull);
    expect(thinkingDisabledForRun(toggleApplied: true), isTrue);
  });

  test('kernel line names the fields a phone batch must record', () {
    final line = formatKernelDiagnostic(
      cpuVariant: 'CPU (android_armv9.2_2)',
      kleidiAiSme: 'unset',
      threads: '8',
      kvType: 'F16/F16',
      flashAttention: 'AUTO',
      useMemoryMap: true,
      nGpuLayers: 0,
      vulkanBackend: 'registered',
    );
    expect(line, contains('variant=CPU (android_armv9.2_2)'));
    expect(line, contains('sme=unset'));
    expect(line, contains('threads=8'));
    expect(line, contains('kv=F16/F16'));
    expect(line, contains('flash=AUTO'));
    expect(line, contains('mmap=true'));
    expect(line, contains('nGpuLayers=0'));
    expect(line, contains('vulkan=registered'));
  });

  test('a different hash of the protocol qwen file is a mismatch', () {
    final report = ModelFileReport(
      path: '/tmp/qwen.gguf',
      fileName: knownQwenFileName,
      sizeBytes: 10,
      sha256: '0' * 64,
      matchesKnownQwen: false,
    );
    expect(report.hashMismatch, isTrue);
    expect(report.mismatchMessage, contains(knownQwenSha256));
  });
}
