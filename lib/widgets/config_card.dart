import 'package:flutter/material.dart';
import 'package:technical_exploration/widgets/report_surface.dart';

class ConfigCard extends StatelessWidget {
  const ConfigCard({
    super.key,
    required this.loaded,
    required this.modelFile,
    required this.architecture,
    required this.parameterCount,
    required this.quantization,
    required this.fileSizeLabel,
    required this.backend,
    required this.threadsLabel,
    required this.contextSize,
    required this.adapterLabel,
    required this.temperature,
    required this.seed,
    required this.topK,
    required this.topP,
    required this.repeatPenalty,
    required this.maxTokens,
  });

  final bool loaded;
  final String modelFile;
  final String architecture;
  final int? parameterCount;
  final String quantization;
  final String fileSizeLabel;
  final String backend;
  final String threadsLabel;
  final int contextSize;
  final String adapterLabel;
  final double temperature;
  final int seed;
  final int topK;
  final double topP;
  final double repeatPenalty;
  final int maxTokens;

  @override
  Widget build(BuildContext context) {
    final modelValue = loaded ? null : emDash;
    return ReportCard(
      title: 'LLM configuration',
      saveName: 'config',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _GroupTitle('Model'),
          _Kv('File', modelValue ?? modelFile),
          _Kv('Architecture', modelValue ?? _orUnknown(architecture)),
          _Kv('Parameters', modelValue ?? formatParameterCount(parameterCount)),
          _Kv('Quantisation', modelValue ?? _orUnknown(quantization)),
          _Kv('File size', modelValue ?? fileSizeLabel),
          const SizedBox(height: 16),
          const _GroupTitle('Runtime'),
          _Kv('Backend', loaded ? backend : emDash),
          _Kv('Threads', loaded ? threadsLabel : emDash),
          _Kv('Context size', loaded ? '$contextSize' : emDash),
          _Kv('Adapter', loaded ? adapterLabel : emDash),
          const SizedBox(height: 16),
          const _GroupTitle('Decoding', locked: true),
          _Kv('Temperature', temperature.toStringAsFixed(1)),
          _Kv('Seed', '$seed'),
          _Kv('Top-k', '$topK'),
          _Kv('Top-p', topP.toStringAsFixed(1)),
          _Kv('Repeat penalty', repeatPenalty.toStringAsFixed(1)),
          _Kv('Max tokens', '$maxTokens'),
        ],
      ),
    );
  }

  static String _orUnknown(String value) {
    if (value.isEmpty) return 'unknown';
    return value;
  }
}

class _GroupTitle extends StatelessWidget {
  const _GroupTitle(this.text, {this.locked = false});

  final String text;
  final bool locked;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Text(text, style: theme.textTheme.titleSmall),
          if (locked) ...[
            const SizedBox(width: 8),
            Text(
              'Locked',
              style: theme.textTheme.labelSmall?.copyWith(
                color: theme.colorScheme.primary,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _Kv extends StatelessWidget {
  const _Kv(this.label, this.value);

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 120,
            child: Text(
              label,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: theme.textTheme.bodyMedium?.copyWith(
                fontFeatures: tabularFigures,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
