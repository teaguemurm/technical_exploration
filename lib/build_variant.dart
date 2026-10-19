// Compile-time switches for the one-change Fold APKs.
//
// Stock builds leave every value at the protocol default: mmap on, threads
// unset, and GPU layers chosen by BackendDetector (0 on Android).

/// `--dart-define=NO_MMAP=true` on the nommap APK only.
const noMmapBuild = bool.fromEnvironment('NO_MMAP');

/// `--dart-define=N_THREADS=4` on the armv8 APK only. Empty leaves threads unset.
const threadBuildDefine = String.fromEnvironment('N_THREADS');

/// `--dart-define=N_GPU_LAYERS=99` on the Vulkan diagnostic APK only.
const gpuLayersBuildDefine = String.fromEnvironment('N_GPU_LAYERS');

/// Null when the build does not pass a thread count.
int? threadsFromBuildDefine() {
  if (threadBuildDefine.isEmpty) return null;
  return int.parse(threadBuildDefine);
}

/// Null when the build does not override GPU layers.
int? gpuLayersFromBuildDefine() {
  if (gpuLayersBuildDefine.isEmpty) return null;
  return int.parse(gpuLayersBuildDefine);
}
