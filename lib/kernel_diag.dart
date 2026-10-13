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

/// Basenames of the libraries that registered, joined for `variant=`.
///
/// `ggml_backend_dev_name` is `CPU` for every KleidiAI variant. Callers pass
/// the paths `dladdr` returned for each device.
String formatRegisteredLibraries(Iterable<String> paths) {
  final names = <String>[
    for (final path in paths)
      if (libraryBasename(path).isNotEmpty) libraryBasename(path),
  ];
  if (names.isEmpty) return 'none';
  return names.join(';');
}

/// Last path segment, accepting both `/` and `\`.
String libraryBasename(String path) {
  final normalized = path.replaceAll('\\', '/');
  final slash = normalized.lastIndexOf('/');
  if (slash < 0 || slash == normalized.length - 1) {
    return slash < 0 ? normalized : '';
  }
  return normalized.substring(slash + 1);
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
  var ggmlFeatures = '';
  var detail = 'kernel-detail: unavailable';

  if (bindings != null) {
    try {
      final lib = loadLlamaLibrary();
      final devices = _deviceLines(lib);
      variant = devices.variant;
      vulkan = devices.vulkan;
      ggmlFeatures = devices.ggmlFeatures;
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
          'kernel-detail: system=${info.trim()} features=${features ?? 'n/a'}'
          '${ggmlFeatures.isEmpty ? '' : ' ggml-features=$ggmlFeatures'}';
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
  const _Devices({
    required this.variant,
    required this.vulkan,
    required this.ggmlFeatures,
  });

  final String variant;
  final String vulkan;
  final String ggmlFeatures;
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
  final regOf = lib
      .lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>),
        ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>)
      >('ggml_backend_dev_backend_reg');
  final procOf = lib
      .lookupFunction<
        ffi.Pointer<ffi.Void> Function(
          ffi.Pointer<ffi.Void>,
          ffi.Pointer<ffi.Char>,
        ),
        ffi.Pointer<ffi.Void> Function(
          ffi.Pointer<ffi.Void>,
          ffi.Pointer<ffi.Char>,
        )
      >('ggml_backend_reg_get_proc_address');
  final dladdr = _lookupDladdr();

  final labels = <String>[];
  final featureParts = <String>[];
  var vulkan = 'absent';
  final n = count();
  for (var i = 0; i < n; i++) {
    final dev = get(i);
    if (dev.address == 0) continue;
    final name = _cstr(nameOf(dev));
    final desc = _cstr(descOf(dev));
    final reg = regOf(dev);
    final library = reg.address == 0
        ? ''
        : _libraryBasenameForReg(procOf, dladdr, reg);
    final label = library.isNotEmpty
        ? library
        : (desc.isEmpty ? name : '$name ($desc)');
    labels.add(label);
    if (reg.address != 0) {
      final features = _backendFeatures(procOf, reg);
      if (features.isNotEmpty) featureParts.add('$label[$features]');
    }
    final blob = '$name $desc $library'.toLowerCase();
    if (blob.contains('vulkan')) vulkan = 'registered';
  }
  return _Devices(
    variant: formatRegisteredLibraries(labels),
    vulkan: vulkan,
    ggmlFeatures: featureParts.join(';'),
  );
}

final class _DlInfo extends ffi.Struct {
  external ffi.Pointer<ffi.Char> fname;

  external ffi.Pointer<ffi.Void> fbase;

  external ffi.Pointer<ffi.Char> sname;

  external ffi.Pointer<ffi.Void> saddr;
}

final class _GgmlFeature extends ffi.Struct {
  external ffi.Pointer<ffi.Char> name;

  external ffi.Pointer<ffi.Char> value;
}

typedef _Dladdr = int Function(ffi.Pointer<ffi.Void>, ffi.Pointer<_DlInfo>);
typedef _ProcOf =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Char>,
    );

_Dladdr? _lookupDladdr() {
  final candidates = <ffi.DynamicLibrary>[];
  if (Platform.isMacOS) {
    candidates.add(ffi.DynamicLibrary.process());
  } else {
    for (final name in ['libdl.so', 'libdl.so.2']) {
      try {
        candidates.add(ffi.DynamicLibrary.open(name));
      } catch (_) {}
    }
    try {
      candidates.add(ffi.DynamicLibrary.process());
    } catch (_) {}
  }
  for (final lib in candidates) {
    try {
      return lib.lookupFunction<
        ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<_DlInfo>),
        _Dladdr
      >('dladdr');
    } catch (_) {}
  }
  return null;
}

String _libraryBasenameForReg(
  _ProcOf procOf,
  _Dladdr? dladdr,
  ffi.Pointer<ffi.Void> reg,
) {
  if (dladdr == null) return '';
  final symbol = 'ggml_backend_set_n_threads'.toNativeUtf8();
  final info = calloc<_DlInfo>();
  try {
    final proc = procOf(reg, symbol.cast());
    if (proc.address == 0) return '';
    if (dladdr(proc, info) == 0 || info.ref.fname.address == 0) return '';
    return libraryBasename(_cstr(info.ref.fname));
  } catch (_) {
    return '';
  } finally {
    calloc.free(symbol);
    calloc.free(info);
  }
}

String _backendFeatures(_ProcOf procOf, ffi.Pointer<ffi.Void> reg) {
  final symbol = 'ggml_backend_get_features'.toNativeUtf8();
  try {
    final proc = procOf(reg, symbol.cast());
    if (proc.address == 0) return '';
    final getFeatures = proc
        .cast<
          ffi.NativeFunction<
            ffi.Pointer<_GgmlFeature> Function(ffi.Pointer<ffi.Void>)
          >
        >()
        .asFunction<
          ffi.Pointer<_GgmlFeature> Function(ffi.Pointer<ffi.Void>)
        >();
    final first = getFeatures(reg);
    if (first.address == 0) return '';
    final parts = <String>[];
    for (var i = 0; i < 64; i++) {
      final feature = first[i];
      if (feature.name.address == 0) break;
      final name = _cstr(feature.name);
      if (name.isEmpty) break;
      final value = feature.value.address == 0 ? '' : _cstr(feature.value);
      parts.add(value.isEmpty ? name : '$name=$value');
    }
    return parts.join(',');
  } catch (_) {
    return '';
  } finally {
    calloc.free(symbol);
  }
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
