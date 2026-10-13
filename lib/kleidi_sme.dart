import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';

/// Compile-time value from `--dart-define=GGML_KLEIDIAI_SME=0`.
///
/// Empty unless that build asked for it. KleidiAI reads the process
/// environment when the CPU backend is loaded, so this is applied with
/// `setenv` before any backend init.
const kleidiAiSmeDefine = String.fromEnvironment('GGML_KLEIDIAI_SME');

/// The value this process actually passed to `setenv`, or null.
String? appliedKleidiAiSme;

/// `sme=` on the kernel line. This is not [Platform.environment]: a later
/// `setenv` does not show up there.
String kleidiAiSmeForLog() => appliedKleidiAiSme ?? 'unset';

/// Sets `GGML_KLEIDIAI_SME` on Android before llama/ggml backend init.
///
/// Other platforms and builds without the define leave the variable alone.
void applyKleidiAiSmeBeforeBackendInit() {
  if (kleidiAiSmeDefine.isEmpty || !Platform.isAndroid) return;
  final libc = ffi.DynamicLibrary.open('libc.so');
  final setenv = libc
      .lookupFunction<
        ffi.Int32 Function(ffi.Pointer<Utf8>, ffi.Pointer<Utf8>, ffi.Int32),
        int Function(ffi.Pointer<Utf8>, ffi.Pointer<Utf8>, int)
      >('setenv');
  final namePtr = 'GGML_KLEIDIAI_SME'.toNativeUtf8();
  final valuePtr = kleidiAiSmeDefine.toNativeUtf8();
  try {
    final result = setenv(namePtr, valuePtr, 1);
    if (result != 0) {
      throw StateError(
        'setenv(GGML_KLEIDIAI_SME, $kleidiAiSmeDefine) failed ($result)',
      );
    }
  } finally {
    calloc.free(namePtr);
    calloc.free(valuePtr);
  }
  appliedKleidiAiSme = kleidiAiSmeDefine;
  // ignore: avoid_print
  print(
    '[technical_exploration] GGML_KLEIDIAI_SME=$kleidiAiSmeDefine '
    '(set before backend init)',
  );
}
