import 'dart:async';
import 'dart:convert';
import 'dart:isolate';

import 'package:llm_core/llm_core.dart'
    show
        LLMApiException,
        LLMChatRepository,
        LLMChunk,
        LLMChunkMessage,
        LLMEmbedding,
        LLMFinishReason,
        LLMLogger,
        DefaultLLMLogger,
        LLMLogLevel,
        LLMMessage,
        LLMRole,
        LLMTool,
        LLMChatOptions,
        ModelLoadException,
        StreamChatOptionsMerger,
        Validation,
        VisionNotSupportedException;
import 'package:llm_llamacpp/src/backend_initializer.dart';
import 'package:llm_llamacpp/src/bindings/llama_bindings.dart';
import 'package:llm_llamacpp/src/embedding_isolate.dart';
import 'package:llm_llamacpp/src/error_translator.dart';
import 'package:llm_llamacpp/src/generation_options.dart';
import 'package:llm_llamacpp/src/persistent_inference_isolate.dart';
import 'package:llm_llamacpp/src/isolate_messages.dart';
import 'package:llm_llamacpp/src/llamacpp_model.dart';
import 'package:llm_llamacpp/src/llamacpp_repository.dart';
import 'package:llm_llamacpp/src/loader/loader.dart';

import 'package:llm_llamacpp/src/response_format_injector.dart';
import 'package:llm_llamacpp/src/tool_call_stream_handler.dart';
import 'package:llm_llamacpp/src/tool_executor.dart';

part 'llamacpp_chat_repository_impl.dart';
part 'llamacpp_chat_repository_embedding.dart';
part 'llamacpp_chat_repository_deprecated.dart';

/// Repository for chatting with llama.cpp models locally.
///
/// Implements [LLMChatRepository] for chat operations. Model management
/// should be handled by [LlamaCppRepository].
class LlamaCppChatRepository extends LLMChatRepository {
  /// Logger instance for this package.
  static final LLMLogger _log = DefaultLLMLogger('llm_llamacpp');

  LlamaCppChatRepository({
    this.contextSize = 4096,
    this.batchSize = 512,
    this.threads,
    this.nGpuLayers = 0,
    this.maxToolAttempts = 90,
    this.stopTokens = const [],
    this._loraPath,
    this._loraScale = 1.0,
  }) : _ownsModel = true;

  LlamaCppChatRepository.withModel(
    LlamaCppModel model,
    LlamaBindings bindings, {
    this.contextSize = 4096,
    this.batchSize = 512,
    this.threads,
    this.nGpuLayers = 0,
    this.maxToolAttempts = 90,
    this.stopTokens = const [],
    this._loraPath,
    this._loraScale = 1.0,
  }) : _model = model,
       _bindings = bindings,
       _backendInitialized = true,
       _ownsModel = false;

  LlamaCppChatRepository.withModelPath(
    String modelPath, {
    this.contextSize = 4096,
    this.batchSize = 512,
    this.threads,
    this.nGpuLayers = 0,
    this.maxToolAttempts = 90,
    this.stopTokens = const [],
    this._loraPath,
    this._loraScale = 1.0,
  }) : _modelPath = modelPath,
       _ownsModel = false;

  /// The context size (number of tokens).
  final int contextSize;

  /// The batch size for processing.
  final int batchSize;

  /// Number of threads to use (null = auto-detect).
  final int? threads;

  /// Number of layers to offload to GPU.
  final int nGpuLayers;

  /// Maximum number of tool calling attempts.
  final int maxToolAttempts;

  /// Extra text-level stop strings for models whose turn-end marker the GGUF
  /// does not flag as end-of-generation.
  ///
  /// These are *additional*. The markers implied by the model's own chat
  /// template are detected from the rendered prompt and appended to this list,
  /// so ChatML (`<|im_end|>`) and Llama-3 (`<|eot_id|>`) already stop correctly
  /// with the default empty list -- set this only for a template that is not
  /// detected:
  ///
  /// ```dart
  /// // Gemma
  /// LlamaCppChatRepository(stopTokens: ['<end_of_turn>']);
  /// // Phi-3
  /// LlamaCppChatRepository(stopTokens: ['<|end|>']);
  /// ```
  ///
  /// Naming a marker that would have been detected anyway is a no-op. Note that
  /// these match against generated *text*, so a string the model can emit as
  /// ordinary prose will truncate the response.
  final List<String> stopTokens;

  /// Whether this repository owns and should dispose the model.
  final bool _ownsModel;

  LlamaBindings? _bindings;
  LlamaCppModel? _model;
  bool _backendInitialized = false;

  /// Model path for lazy loading (model is loaded in inference isolate, not main isolate)
  /// This is used when the repository is created with withModelPath constructor.
  String? _modelPath;

  // LoRA configuration. [_loraPath] is what the next chat request applies.
  // [_attachedLora] is the copy kept on this process's loaded model so a
  // load-time memory reading includes the adapter weights.
  String? _loraPath;
  double _loraScale;
  LoraManager? _loraManager;
  LlamaLoraAdapter? _attachedLora;

  void _releaseAttachedLora() {
    final attached = _attachedLora;
    _attachedLora = null;
    if (attached != null) {
      _loraManager?.unloadLora(attached, force: true);
    }
  }

  /// The currently loaded model, if any.
  @Deprecated('Use LlamaCppRepository for model management')
  LlamaCppModel? get model => _model;

  /// Whether a model is currently loaded.
  bool get isModelLoaded => _model != null;

  String? get loraPath => _loraPath;
  double get loraScale => _loraScale;
  bool get hasLora => _loraPath != null;

  void setLora(String path, {double scale = 1.0}) {
    if (_attachedLora != null && _attachedLora!.path != path) {
      _releaseAttachedLora();
    }
    _loraPath = path;
    _loraScale = scale;
  }

  void setLoraConfig(LoraConfig config) {
    setLora(config.path, scale: config.scale);
  }

  void clearLora() {
    _loraPath = null;
    _loraScale = 1.0;
    _releaseAttachedLora();
  }

  /// Keeps [path] resident on the loaded model and uses it for later chats.
  ///
  /// The model must already be loaded in this process. A second call with the
  /// same path only updates [scale]. Failure leaves no adapter path set.
  void attachLora(String path, {double scale = 1.0}) {
    final model = _model;
    final bindings = _bindings;
    if (model == null || bindings == null) {
      throw StateError('Load a model before attaching a LoRA adapter.');
    }
    if (_attachedLora?.path == path) {
      _loraPath = path;
      _loraScale = scale;
      return;
    }
    if (_attachedLora != null) {
      _releaseAttachedLora();
    }
    _loraManager ??= LoraManager(bindings);
    try {
      _attachedLora = _loraManager!.loadLora(path, model);
      _loraPath = path;
      _loraScale = scale;
    } catch (error) {
      _attachedLora = null;
      _loraPath = null;
      _loraScale = 1.0;
      rethrow;
    }
  }

  @Deprecated('Use LlamaCppRepository for backend management')
  void initializeBackend() {
    _initializeBackendImpl(this);
  }

  @Deprecated(
    'Use LlamaCppRepository.loadModel() instead for proper separation of concerns',
  )
  Future<void> loadModel(
    String modelPath, {
    ModelLoadOptions options = const ModelLoadOptions(),
  }) async {
    await _loadModelImpl(this, modelPath, options: options);
  }

  @Deprecated('Use LlamaCppRepository.unloadModel() instead')
  void unloadModel() {
    _unloadModelImpl(this);
  }

  @override
  Stream<LLMChunk> streamChat(
    String model, {
    required List<LLMMessage> messages,
    bool think = false,
    List<LLMTool> tools = const [],
    dynamic extra,
    LLMChatOptions? options,
  }) async* {
    yield* streamChatWithGenerationOptions(
      model,
      messages: messages,
      think: think,
      tools: tools,
      extra: extra,
      options: options,
      generationOptions: const GenerationOptions(),
    );
  }

  Stream<LLMChunk> streamChatWithGenerationOptions(
    String model, {
    required List<LLMMessage> messages,
    bool think = false,
    List<LLMTool> tools = const [],
    dynamic extra,
    LLMChatOptions? options,
    GenerationOptions? generationOptions,
  }) async* {
    yield* _streamChatImpl(
      this,
      model,
      messages,
      think,
      tools,
      extra,
      options,
      generationOptions,
    );
  }

  @override
  Future<List<LLMEmbedding>> embed({
    required String model,
    required List<String> messages,
    Map<String, dynamic> options = const {},
  }) async {
    return _embedImpl(this, model, messages, options);
  }

  @override
  Future<List<LLMEmbedding>> batchEmbed({
    required String model,
    required List<String> messages,
    Map<String, dynamic> options = const {},
  }) async {
    return embed(model: model, messages: messages, options: options);
  }

  void dispose() {
    _releaseAttachedLora();
    _loraManager?.dispose();
    _loraManager = null;
    if (_ownsModel) {
      // ignore: deprecated_member_use_from_same_package
      unloadModel();
      if (_backendInitialized && _bindings != null) {
        _bindings!.llama_backend_free();
        _backendInitialized = false;
      }
    }
    // Clear references but don't dispose if we don't own the model
    _model = null;
    _bindings = null;
  }
}
