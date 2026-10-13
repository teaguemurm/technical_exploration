import 'dart:io';

import 'package:crypto/crypto.dart';

/// SHA-256 of the Mac protocol copy of `qwen2.5-1.5b-instruct-q4_k_m.gguf`.
///
/// A phone file with the same name and a different hash is a different file.
/// Kernel experiments on that copy would not explain the Fold 8 batch.
const knownQwenFileName = 'qwen2.5-1.5b-instruct-q4_k_m.gguf';
const knownQwenSha256 =
    '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e';

class ModelFileReport {
  const ModelFileReport({
    required this.path,
    required this.fileName,
    required this.sizeBytes,
    required this.sha256,
    required this.matchesKnownQwen,
  });

  final String path;
  final String fileName;
  final int sizeBytes;
  final String sha256;

  /// True when this is the known Qwen file and the hash matches.
  /// False when the name matches and the hash does not.
  /// Null for any other file name (logged, not compared).
  final bool? matchesKnownQwen;

  bool get hashMismatch => matchesKnownQwen == false;

  String get logLine =>
      'model-file: name=$fileName bytes=$sizeBytes sha256=$sha256 '
      'knownQwenMatch=$matchesKnownQwen';

  String get mismatchMessage =>
      'GGUF sha256 does not match the Mac protocol file $knownQwenFileName. '
      'Expected $knownQwenSha256, got $sha256. Re-copy the file. '
      'The batch was not started.';
}

Future<ModelFileReport> inspectModelFile(String path) async {
  final file = File(path);
  final sizeBytes = await file.length();
  final digest = await sha256.bind(file.openRead()).first;
  final hash = digest.toString();
  final fileName = path.split(Platform.pathSeparator).last;
  final bool? matches = fileName == knownQwenFileName
      ? hash == knownQwenSha256
      : null;
  return ModelFileReport(
    path: path,
    fileName: fileName,
    sizeBytes: sizeBytes,
    sha256: hash,
    matchesKnownQwen: matches,
  );
}
