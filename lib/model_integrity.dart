import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';

/// SHA-256 of the Mac protocol copy of `qwen2.5-1.5b-instruct-q4_k_m.gguf`.
///
/// A phone file with the same name and a different hash is a different file.
/// Kernel experiments on that copy would not explain the Fold 8 batch.
const knownQwenFileName = 'qwen2.5-1.5b-instruct-q4_k_m.gguf';
const knownQwenSha256 =
    '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e';

/// JSON file written beside the CSV and JSON exports.
const modelHashCacheFileName = 'model_sha256_cache.json';

/// Reads a GGUF and returns its SHA-256. Tests replace this to count reads.
typedef ModelHasher = Future<String> Function(File file);

class ModelFileReport {
  const ModelFileReport({
    required this.path,
    required this.fileName,
    required this.sizeBytes,
    required this.sha256,
    required this.matchesKnownQwen,
    this.modifiedMs = 0,
    this.verified = true,
  });

  final String path;
  final String fileName;
  final int sizeBytes;
  final String sha256;

  /// Milliseconds since epoch of the mtime this report was keyed on.
  final int modifiedMs;

  /// False when the load path reused no cache entry and did not read the file.
  final bool verified;

  /// True when this is the known Qwen file and the hash matches.
  /// False when the name matches and the hash does not.
  /// Null for any other file name, and for an unverified file.
  final bool? matchesKnownQwen;

  bool get hashMismatch => matchesKnownQwen == false;

  String get logLine =>
      'model-file: name=$fileName bytes=$sizeBytes '
      'sha256=${verified ? sha256 : 'unverified'} '
      'knownQwenMatch=$matchesKnownQwen';

  String get mismatchMessage =>
      'GGUF sha256 does not match the Mac protocol file $knownQwenFileName. '
      'Expected $knownQwenSha256, got $sha256. Re-copy the file. '
      'The batch was not started.';

  Map<String, dynamic> toJson() => {
    'path': path,
    'fileName': fileName,
    'sizeBytes': sizeBytes,
    'modifiedMs': modifiedMs,
    'sha256': sha256,
    'matchesKnownQwen': matchesKnownQwen,
  };

  static ModelFileReport? fromJson(Map<String, dynamic> json) {
    final path = json['path'];
    final fileName = json['fileName'];
    final sizeBytes = json['sizeBytes'];
    final modifiedMs = json['modifiedMs'];
    final sha256Hash = json['sha256'];
    if (path is! String ||
        fileName is! String ||
        sizeBytes is! int ||
        modifiedMs is! int ||
        sha256Hash is! String) {
      return null;
    }
    final match = json['matchesKnownQwen'];
    return ModelFileReport(
      path: path,
      fileName: fileName,
      sizeBytes: sizeBytes,
      sha256: sha256Hash,
      modifiedMs: modifiedMs,
      matchesKnownQwen: match is bool ? match : null,
    );
  }
}

/// In-memory SHA-256 cache keyed by path, size, and mtime.
class ModelHashCache {
  final Map<String, ModelFileReport> _byKey = {};

  ModelFileReport? lookup(String path, int sizeBytes, DateTime modified) {
    return _byKey[_key(path, sizeBytes, modified)];
  }

  void put(ModelFileReport report) {
    _byKey[_key(
          report.path,
          report.sizeBytes,
          DateTime.fromMillisecondsSinceEpoch(report.modifiedMs),
        )] =
        report;
  }

  Map<String, dynamic> toJson() => {
    'version': 1,
    'entries': [for (final report in _byKey.values) report.toJson()],
  };

  void replaceFromJson(Map<String, dynamic> json) {
    _byKey.clear();
    final entries = json['entries'];
    if (entries is! List) return;
    for (final entry in entries) {
      if (entry is! Map) continue;
      final report = ModelFileReport.fromJson(Map<String, dynamic>.from(entry));
      if (report == null || !report.verified) continue;
      put(report);
    }
  }

  static String _key(String path, int sizeBytes, DateTime modified) =>
      '$path|$sizeBytes|${modified.millisecondsSinceEpoch}';
}

/// `model_sha256_cache.json` in the exports directory.
class ModelHashStore {
  ModelHashStore(this.file, {ModelHashCache? cache})
    : cache = cache ?? ModelHashCache();

  final File file;
  final ModelHashCache cache;

  Future<void> load() async {
    if (!await file.exists()) return;
    try {
      final decoded = jsonDecode(await file.readAsString());
      if (decoded is Map<String, dynamic>) {
        cache.replaceFromJson(decoded);
      } else if (decoded is Map) {
        cache.replaceFromJson(Map<String, dynamic>.from(decoded));
      }
    } catch (_) {
      // A corrupt cache must not block a load. The next verify rewrites it.
    }
  }

  Future<void> save() async {
    final parent = file.parent;
    if (!await parent.exists()) {
      await parent.create(recursive: true);
    }
    const encoder = JsonEncoder.withIndent('  ');
    await file.writeAsString(
      '${encoder.convert(cache.toJson())}\n',
      flush: true,
    );
  }
}

/// Cache lookup only. Does not read the GGUF.
///
/// A miss is `verified: false` and `sha256` logged as unverified. A hit returns
/// the stored report, including a known-Qwen mismatch.
Future<ModelFileReport> lookupCachedModelFile(
  String path,
  ModelHashCache cache,
) async {
  final file = File(path);
  final stat = await file.stat();
  if (stat.type == FileSystemEntityType.notFound) {
    throw FileSystemException('GGUF not found', path);
  }
  final hit = cache.lookup(path, stat.size, stat.modified);
  if (hit != null) return hit;
  return ModelFileReport(
    path: path,
    fileName: _fileName(path),
    sizeBytes: stat.size,
    modifiedMs: stat.modified.millisecondsSinceEpoch,
    sha256: 'unverified',
    matchesKnownQwen: null,
    verified: false,
  );
}

/// Hashes [path] and stores the result.
///
/// A cache hit is returned without reading the file unless [force] is true.
/// Verify passes `force: true`. Load does not call this.
Future<ModelFileReport> inspectModelFile(
  String path, {
  ModelHashCache? cache,
  bool force = false,
  ModelHasher? hasher,
}) async {
  final file = File(path);
  final stat = await file.stat();
  if (stat.type == FileSystemEntityType.notFound) {
    throw FileSystemException('GGUF not found', path);
  }
  if (!force && cache != null) {
    final hit = cache.lookup(path, stat.size, stat.modified);
    if (hit != null) return hit;
  }
  final hash = await (hasher ?? _sha256OfFile)(file);
  final fileName = _fileName(path);
  final bool? matches = fileName == knownQwenFileName
      ? hash == knownQwenSha256
      : null;
  final report = ModelFileReport(
    path: path,
    fileName: fileName,
    sizeBytes: stat.size,
    modifiedMs: stat.modified.millisecondsSinceEpoch,
    sha256: hash,
    matchesKnownQwen: matches,
  );
  cache?.put(report);
  return report;
}

Future<String> _sha256OfFile(File file) async {
  final digest = await sha256.bind(file.openRead()).first;
  return digest.toString();
}

String _fileName(String path) => path.split(Platform.pathSeparator).last;
