import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:technical_exploration/messages.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const TechnicalExplorationApp());
}

class TechnicalExplorationApp extends StatelessWidget {
  const TechnicalExplorationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'technical_exploration',
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF3F6B4A),
        useMaterial3: true,
      ),
      home: const NoteScreen(),
    );
  }
}

class NoteScreen extends StatefulWidget {
  const NoteScreen({super.key});

  @override
  State<NoteScreen> createState() => _NoteScreenState();
}

class _NoteScreenState extends State<NoteScreen> {
  final _outputController = TextEditingController();
  final _messagesController = TextEditingController();

  LlamaCppChatRepository? _repo;
  CatalogData? _catalog;
  PromptSpec? _selected;

  String? _modelPath;
  String? _modelFileName;
  int? _coldLoadMs;
  String? _loadError;
  String? _catalogError;
  bool _loading = false;
  bool _loaded = false;
  bool _useDataset = false;
  bool _running = false;
  bool _hasRun = false;
  bool _stopRequested = false;

  int? _ttftMs;
  double? _tokensPerSecond;
  String? _memoryLine;
  String? _status;

  StreamSubscription<LLMChunk>? _subscription;
  Stopwatch? _runWatch;
  Duration? _firstTokenAt;

  @override
  void initState() {
    super.initState();
    unawaited(_reloadCatalog());
  }

  @override
  void dispose() {
    unawaited(_subscription?.cancel());
    _repo?.dispose();
    _outputController.dispose();
    _messagesController.dispose();
    super.dispose();
  }

  bool get _canLoad =>
      _modelPath != null && !_loading && !_running;

  bool get _canRun =>
      _loaded && _selected != null && !_running && !_loading && _catalog != null;

  Future<void> _reloadCatalog() async {
    try {
      final catalog = parseCatalog(
        systemPrompt: await rootBundle.loadString('assets/system_prompt.txt'),
        promptsJson: await rootBundle.loadString('assets/prompts.json'),
        datasetJson: await rootBundle.loadString('assets/dataset.json'),
      );
      if (!mounted) {
        return;
      }
      setState(() {
        _catalog = catalog;
        _catalogError = null;
        if (_selected != null) {
          final match = catalog.prompts.where((prompt) => prompt.id == _selected!.id);
          _selected = match.isEmpty ? null : match.first;
        }
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() => _catalogError = error.toString());
    }
  }

  Future<void> _pickModel() async {
    final file = await FilePicker.pickFile(
      dialogTitle: 'Choose a GGUF model',
      type: FileType.custom,
      allowedExtensions: const ['gguf'],
    );
    if (!mounted || file == null) {
      return;
    }
    final path = file.path;
    if (path == null) {
      setState(() {
        _loadError = 'The picker did not return a local path.';
      });
      return;
    }
    setState(() {
      if (_modelPath != path) {
        _loaded = false;
        _coldLoadMs = null;
        _loadError = null;
      }
      _modelPath = path;
      _modelFileName = file.name;
    });
  }

  Future<void> _loadModel() async {
    final path = _modelPath;
    if (path == null || _loading) {
      return;
    }
    setState(() {
      _loading = true;
      _loadError = null;
      _loaded = false;
      _coldLoadMs = null;
    });
    final gpuLayers = BackendDetector.getRecommendedGpuLayers();
    final repo = _repo ??= LlamaCppChatRepository(
      contextSize: 2048,
      nGpuLayers: gpuLayers,
    );
    final watch = Stopwatch()..start();
    try {
      // 0.7.0's basic-chat example loads here. The repository method is the
      // newer split, and this call is what the README still shows.
      // ignore: deprecated_member_use
      await repo.loadModel(
        path,
        options: ModelLoadOptions(
          nGpuLayers: gpuLayers,
          useMemoryMap: true,
        ),
      );
      watch.stop();
      if (!mounted) {
        return;
      }
      setState(() {
        _loaded = true;
        _coldLoadMs = watch.elapsedMilliseconds;
        _loading = false;
      });
    } catch (error) {
      watch.stop();
      if (!mounted) {
        return;
      }
      setState(() {
        _loaded = false;
        _coldLoadMs = null;
        _loadError = error.toString();
        _loading = false;
      });
    }
  }

  Future<void> _choosePrompt() async {
    final catalog = _catalog;
    if (catalog == null) {
      return;
    }
    final picked = await showDialog<PromptSpec>(
      context: context,
      builder: (context) {
        return SimpleDialog(
          title: const Text('Prompts'),
          children: [
            for (final prompt in catalog.prompts)
              SimpleDialogOption(
                onPressed: () => Navigator.pop(context, prompt),
                child: Text(prompt.label),
              ),
          ],
        );
      },
    );
    if (picked != null && mounted) {
      setState(() => _selected = picked);
    }
  }

  Future<void> _run() async {
    final repo = _repo;
    final selected = _selected;
    final fileName = _modelFileName;
    if (repo == null || selected == null || fileName == null || _running) {
      return;
    }
    late final CatalogData catalog;
    try {
      catalog = parseCatalog(
        systemPrompt: await rootBundle.loadString('assets/system_prompt.txt'),
        promptsJson: await rootBundle.loadString('assets/prompts.json'),
        datasetJson: await rootBundle.loadString('assets/dataset.json'),
      );
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() => _catalogError = error.toString());
      return;
    }
    final prompt = catalog.prompts.where((item) => item.id == selected.id);
    if (prompt.isEmpty) {
      setState(() {
        _catalog = catalog;
        _selected = null;
        _catalogError = 'The selected prompt is no longer in prompts.json.';
      });
      return;
    }
    final messages = buildMessages(
      systemPrompt: catalog.systemPrompt,
      promptText: prompt.first.text,
      useDataset: _useDataset,
      datasetLines: [for (final fact in catalog.facts) fact.text],
    );
    if (!mounted) {
      return;
    }
    setState(() {
      _catalog = catalog;
      _selected = prompt.first;
      _catalogError = null;
      _running = true;
      _hasRun = true;
      _stopRequested = false;
      _ttftMs = null;
      _tokensPerSecond = null;
      _memoryLine = null;
      _status = null;
      _firstTokenAt = null;
      _outputController.text = '';
      _messagesController.text = formatMessages(messages);
    });

    final watch = Stopwatch()..start();
    _runWatch = watch;
    final stream = repo.streamChatWithGenerationOptions(
      fileName,
      messages: messages,
      generationOptions: const GenerationOptions(maxTokens: 256),
    );
    _subscription = stream.listen(
      _onChunk,
      onError: (Object error, StackTrace _) {
        if (!mounted) {
          return;
        }
        setState(() {
          _running = false;
          _status = error.toString();
        });
      },
      onDone: () {
        if (!mounted) {
          return;
        }
        setState(() {
          _running = false;
          _subscription = null;
          if (_stopRequested && _tokensPerSecond == null) {
            _status = 'Stopped before the library reported a token count.';
          }
        });
      },
    );
  }

  void _onChunk(LLMChunk chunk) {
    final text = chunk.message?.content;
    if (text != null && text.isNotEmpty) {
      _firstTokenAt ??= _runWatch?.elapsed;
      _outputController.text += text;
    }
    if (chunk.done == true) {
      final first = _firstTokenAt;
      final elapsed = _runWatch?.elapsed;
      final generation = first == null || elapsed == null
          ? Duration.zero
          : elapsed - first;
      final rate = tokensPerSecond(
        evalCount: chunk.evalCount,
        generation: generation,
      );
      final memory = formatMemoryReport(
        providerMetadata: chunk.providerMetadata,
        backends: [
          for (final backend in BackendDetector.getAvailableBackends())
            BackendMemory(
              name: backend.name,
              total: backend.memoryTotal,
              free: backend.memoryFree,
            ),
        ],
      );
      if (!mounted) {
        return;
      }
      setState(() {
        if (first != null) {
          _ttftMs = first.inMilliseconds;
        }
        _tokensPerSecond = rate;
        _memoryLine = memory;
      });
    }
  }

  Future<void> _stop() async {
    _stopRequested = true;
    final firstTokenAt = _firstTokenAt;
    await _subscription?.cancel();
    _subscription = null;
    if (!mounted) {
      return;
    }
    setState(() {
      _running = false;
      if (firstTokenAt != null) {
        _ttftMs = firstTokenAt.inMilliseconds;
      }
      if (_tokensPerSecond == null) {
        _status = 'Stopped before the library reported a token count.';
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('Visitor note')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            Text('Model', style: textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(_modelFileName ?? 'No file selected'),
            const SizedBox(height: 12),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                FilledButton.tonal(
                  onPressed: _running || _loading ? null : _pickModel,
                  child: const Text('Pick GGUF'),
                ),
                FilledButton(
                  onPressed: _canLoad ? _loadModel : null,
                  child: const Text('Load'),
                ),
                if (_loading) const CircularProgressIndicator(),
              ],
            ),
            if (_coldLoadMs != null) ...[
              const SizedBox(height: 8),
              Text('Cold load: $_coldLoadMs ms'),
            ],
            if (_loadError != null) ...[
              const SizedBox(height: 8),
              Text(_loadError!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ],
            const SizedBox(height: 24),
            Align(
              alignment: Alignment.centerLeft,
              child: FilledButton.tonal(
                onPressed: _catalog == null || _running ? null : _choosePrompt,
                child: Text(_selected?.label ?? 'Choose prompt'),
              ),
            ),
            if (_catalogError != null) ...[
              const SizedBox(height: 8),
              Text(_catalogError!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ],
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Use dataset'),
              value: _useDataset,
              onChanged: _running
                  ? null
                  : (value) => setState(() => _useDataset = value),
            ),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                FilledButton(
                  onPressed: _canRun ? _run : null,
                  child: const Text('Run'),
                ),
                OutlinedButton(
                  onPressed: _running ? _stop : null,
                  child: const Text('Stop'),
                ),
              ],
            ),
            const SizedBox(height: 24),
            TextField(
              controller: _outputController,
              readOnly: true,
              minLines: 8,
              maxLines: 16,
              decoration: const InputDecoration(
                labelText: 'Output',
                alignLabelWithHint: true,
                border: OutlineInputBorder(),
              ),
            ),
            if (_hasRun && !_running) ...[
              const SizedBox(height: 12),
              if (_ttftMs != null) Text('Time to first token: $_ttftMs ms'),
              if (_tokensPerSecond != null)
                Text(
                  'Tokens per second: ${_tokensPerSecond!.toStringAsFixed(2)}',
                ),
              if (_memoryLine != null) Text(_memoryLine!),
              if (_status != null) Text(_status!),
            ],
            const SizedBox(height: 24),
            Text('Messages sent', style: textTheme.titleMedium),
            const SizedBox(height: 8),
            TextField(
              controller: _messagesController,
              readOnly: true,
              minLines: 8,
              maxLines: 18,
              decoration: const InputDecoration(
                alignLabelWithHint: true,
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
