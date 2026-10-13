import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:technical_exploration/build_variant.dart';
import 'package:technical_exploration/device_probe.dart';
import 'package:technical_exploration/kernel_diag.dart';
import 'package:technical_exploration/kleidi_sme.dart';
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
      formatAndroidDeviceLabel(manufacturer: 'samsung', model: 'SM-F976B'),
      'Samsung SM-F976B',
    );
  });

  test(
    'battery tenths become celsius and the legacy zone stays out of that value',
    () {
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
    },
  );

  test('cold load is only the first run when sanity did not run', () {
    expect(
      isColdForThisRun(
        loadWasCold: true,
        runIndexSinceLoad: 1,
        sanityRan: false,
      ),
      isTrue,
    );
    expect(
      isColdForThisRun(
        loadWasCold: true,
        runIndexSinceLoad: 2,
        sanityRan: false,
      ),
      isFalse,
    );
    expect(
      isColdForThisRun(
        loadWasCold: true,
        runIndexSinceLoad: 50,
        sanityRan: false,
      ),
      isFalse,
    );
    expect(
      isColdForThisRun(
        loadWasCold: false,
        runIndexSinceLoad: 1,
        sanityRan: false,
      ),
      isFalse,
    );
    expect(
      isColdForThisRun(
        loadWasCold: true,
        runIndexSinceLoad: 1,
        sanityRan: true,
      ),
      isFalse,
    );
    expect(
      isColdForThisRun(
        loadWasCold: true,
        runIndexSinceLoad: 2,
        sanityRan: true,
      ),
      isFalse,
    );
    expect(thinkingDisabledForRun(toggleApplied: false), isNull);
    expect(thinkingDisabledForRun(toggleApplied: true), isTrue);
  });

  test('repository context fields stay unset unless threads is passed', () {
    final unset = repositoryContextFields();
    expect(unset.threads, 'unset');
    expect(unset.kvType, 'unset');
    expect(unset.flashAttention, 'unset');
    expect(repositoryContextFields(threads: 8).threads, '8');
    expect(kleidiAiSmeForLog(), 'unset');
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

  test('registered library names are basenames', () {
    expect(
      formatRegisteredLibraries([
        '/data/app/lib/arm64/libggml-cpu-android_armv8.2_1.so',
        r'C:\libs\libggml-cpu-android_armv9.2_2.so',
      ]),
      'libggml-cpu-android_armv8.2_1.so;libggml-cpu-android_armv9.2_2.so',
    );
    expect(formatRegisteredLibraries(const <String>[]), 'none');
    expect(
      libraryBasename('libggml-cpu-android_armv8.0_1.so'),
      'libggml-cpu-android_armv8.0_1.so',
    );
    expect(threadsFromBuildDefine(), isNull);
    expect(gpuLayersFromBuildDefine(), isNull);
    expect(noMmapBuild, isFalse);
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

  test('a cache hit does not hash, and the cache reloads from disk', () async {
    final dir = await Directory.systemTemp.createTemp('gguf-hash');
    addTearDown(() async {
      if (dir.existsSync()) await dir.delete(recursive: true);
    });
    final file = File('${dir.path}/$knownQwenFileName');
    await file.writeAsString('not a real gguf');
    var hashes = 0;
    Future<String> hasher(File _) async {
      hashes++;
      return 'abc';
    }

    final cache = ModelHashCache();
    final first = await inspectModelFile(
      file.path,
      cache: cache,
      hasher: hasher,
    );
    expect(hashes, 1);
    expect(first.verified, isTrue);
    expect(first.hashMismatch, isTrue);

    final second = await inspectModelFile(
      file.path,
      cache: cache,
      hasher: hasher,
    );
    expect(hashes, 1);
    expect(second.sha256, 'abc');

    final lookedUp = await lookupCachedModelFile(file.path, cache);
    expect(lookedUp.verified, isTrue);
    expect(lookedUp.sha256, 'abc');
    expect(hashes, 1);

    final other = File('${dir.path}/other.gguf');
    await other.writeAsString('x');
    final miss = await lookupCachedModelFile(other.path, cache);
    expect(miss.verified, isFalse);
    expect(miss.logLine, contains('sha256=unverified'));
    expect(miss.hashMismatch, isFalse);
    expect(hashes, 1);

    await file.setLastModified(DateTime.utc(2030, 1, 2));
    final changed = await inspectModelFile(
      file.path,
      cache: cache,
      hasher: hasher,
    );
    expect(hashes, 2);
    expect(changed.sha256, 'abc');

    final storeFile = File('${dir.path}/$modelHashCacheFileName');
    final store = ModelHashStore(storeFile, cache: cache);
    await store.save();
    final reloaded = ModelHashStore(storeFile);
    await reloaded.load();
    final fromDisk = await lookupCachedModelFile(file.path, reloaded.cache);
    expect(fromDisk.sha256, 'abc');
    expect(fromDisk.hashMismatch, isTrue);
    expect(hashes, 2);
  });
}
