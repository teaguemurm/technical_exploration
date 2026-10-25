/// Scale used for every protocol run that has an adapter loaded.
const protocolLoraScale = 1.0;

/// Why an adapter must not be loaded, or null when the architectures match.
///
/// [modelArchitecture] and [adapterArchitecture] are `general.architecture`
/// from the GGUF headers. Null means the header could not be read. This does
/// not use a fallback architecture.
String? adapterArchitectureRefusal({
  required String? modelArchitecture,
  required String? adapterArchitecture,
}) {
  if (modelArchitecture == null || modelArchitecture.isEmpty) {
    return 'Could not read the model architecture, so the adapter was not loaded.';
  }
  if (adapterArchitecture == null || adapterArchitecture.isEmpty) {
    return "Could not read this adapter's architecture, so it was not loaded.";
  }
  if (modelArchitecture != adapterArchitecture) {
    return 'This adapter is for $adapterArchitecture, but the model is '
        '$modelArchitecture. Pick an adapter that matches the model.';
  }
  return null;
}
