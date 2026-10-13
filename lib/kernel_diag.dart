// ignore_for_file: implementation_imports

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:llm_llamacpp/src/loader/loader.dart';
import 'package:technical_exploration/kleidi_sme.dart';
import 'package:technical_exploration/prompt_tokens.dart';

/// One-line record of the CPU configuration that will run a batch.
///
/// No export column. The app prints this at batch start and keeps it on screen.
String formatKernelDiagnostic({
  required String cpuVariant,
  required String kleidiAiSme,
  required String threads,
  required String kvType,
  required String flashAttention,
  required bool useMemoryMap,
  required int nGpuLayers,
  required String vulkanBackend,
}) {
  return 'kernel: variant=$cpuVariant sme=$kleidiAiSme threads=$threads '
      'kv=$kvType flash=$flashAttention mmap=$useMemoryMap '
      'nGpuLayers=$nGpuLayers vulkan=$vulkanBackend';
}

class KernelReport {
  const KernelReport({required this.line, required this.detail});

  final String line;
  final String detail;
}

/// Threads, KV type, and flash attention as this app's repository passes them.
///
/// `llm_llamacpp` writes `n_threads` only when [threads] is non-null. It never
/// writes the KV cache type or flash attention. Those stay `unset` here.
/// This does not call `llama_context_default_params`.
({String threads, String kvType, String flashAttention})
repositoryContextFields({int? threads}) {
  return (
    threads: threads == null ? 'unset' : '$threads',
    kvType: 'unset',
    flashAttention: 'unset',
  );
}

/// Reads the backends ggml actually registered, plus the context fields the
/// repository passes. See [repositoryContextFields].
KernelReport collectKernelReport({
  required bool useMemoryMap,
  required int nGpuLayers,
  int? threads,
}) {
  final bindings = resolveLlamaBindings();
  final context = repositoryContextFields(threads: threads);
  var variant = 'unknown';
  var vulkan = 'unknown';
  var detail = 'kernel-detail: unavailable';

  if (bindings != null) {
    try {
      final lib = loadLlamaLibrary();
      final devices = _deviceLines(lib);
      variant = devices.variant;
      vulkan = devices.vulkan;
    } catch (error) {
      variant = 'lookup-failed';
      vulkan = 'lookup-failed';
    }

    try {
      final infoPtr = bindings.llama_print_system_info();
      final info = infoPtr.address == 0
          ? ''
          : infoPtr.cast<Utf8>().toDartString().replaceAll('\n', ' | ');
      final features = _cpuFeatures();
      detail =
          'kernel-detail: system=${info.trim()} features=${features ?? 'n/a'}';
    } catch (error) {
      detail = 'kernel-detail: system-info-failed ($error)';
    }
  }

  return KernelReport(
    line: formatKernelDiagnostic(
      cpuVariant: variant,
      kleidiAiSme: kleidiAiSmeForLog(),
      threads: context.threads,
      kvType: context.kvType,
      flashAttention: context.flashAttention,
      useMemoryMap: useMemoryMap,
      nGpuLayers: nGpuLayers,
      vulkanBackend: vulkan,
    ),
    detail: detail,
  );
}

class _Devices {
  const _Devices({required this.variant, required this.vulkan});

  final String variant;
  final String vulkan;
}

_Devices _deviceLines(ffi.DynamicLibrary lib) {
  final count = lib.lookupFunction<ffi.Size Function(), int Function()>(
    'ggml_backend_dev_count',
  );
  final get = lib
      .lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Size),
        ffi.Pointer<ffi.Void> Function(int)
      >('ggml_backend_dev_get');
  final nameOf = lib
      .lookupFunction<
        ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Void>),
        ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Void>)
      >('ggml_backend_dev_name');
  final descOf = lib
      .lookupFunction<
        ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Void>),
        ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Void>)
      >('ggml_backend_dev_description');

  final parts = <String>[];
  var vulkan = 'absent';
  final n = count();
  for (var i = 0; i < n; i++) {
    final dev = get(i);
    if (dev.address == 0) continue;
    final name = _cstr(nameOf(dev));
    final desc = _cstr(descOf(dev));
    parts.add(desc.isEmpty ? name : '$name ($desc)');
    if (name.toLowerCase().contains('vulkan') ||
        desc.toLowerCase().contains('vulkan')) {
      vulkan = 'registered';
    }
  }
  return _Devices(
    variant: parts.isEmpty ? 'none' : parts.join('; '),
    vulkan: vulkan,
  );
}

String _cstr(ffi.Pointer<ffi.Char> ptr) {
  if (ptr.address == 0) return '';
  return ptr.cast<Utf8>().toDartString();
}

String? _cpuFeatures() {
  if (!Platform.isAndroid && !Platform.isLinux) return null;
  try {
    final text = File('/proc/cpuinfo').readAsStringSync();
    final match = RegExp(
      r'^Features\s*:\s*(.+)$',
      multiLine: true,
    ).firstMatch(text);
    return match?.group(1)?.trim();
  } catch (_) {
    return null;
  }
}
