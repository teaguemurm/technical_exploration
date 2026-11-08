import 'package:flutter/material.dart';
import 'package:llm_llamacpp/llm_llamacpp.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/system_metrics.dart';
import 'package:technical_exploration/widgets/comparison_view.dart';
import 'package:technical_exploration/widgets/report_surface.dart';

class MessagesCard extends StatelessWidget {
  const MessagesCard({
    super.key,
    required this.header,
    required this.messages,
    required this.output,
    required this.streaming,
    required this.hasRun,
  });

  final String header;
  final List<LLMMessage> messages;
  final String output;
  final bool streaming;
  final bool hasRun;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final generatedLabel = streaming
        ? 'Streaming'
        : (hasRun ? 'Final' : '');
    return ReportCard(
      title: 'Messages & output',
      saveName: 'messages',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            header,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 16),
          Text('Sent', style: theme.textTheme.titleSmall),
          const SizedBox(height: 8),
          if (messages.isEmpty)
            Text(
              'No messages yet.',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            )
          else
            for (final message in messages) ...[
              _MessageBlock(
                label: _labelFor(message),
                text: message.content ?? '',
                injected: _isInjected(message),
              ),
              const SizedBox(height: 8),
            ],
          const SizedBox(height: 8),
          Row(
            children: [
              Text('Generated note', style: theme.textTheme.titleSmall),
              if (generatedLabel.isNotEmpty) ...[
                const SizedBox(width: 8),
                Text(
                  generatedLabel,
                  style: theme.textTheme.labelMedium?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
              const Spacer(),
              CopyIcon(text: output, label: 'generated note'),
            ],
          ),
          const SizedBox(height: 8),
          ConstrainedBox(
            constraints: const BoxConstraints(minHeight: 120),
            child: Align(
              alignment: Alignment.topLeft,
              child: Text(
                output.isEmpty ? emDash : output,
                style: theme.textTheme.bodyLarge?.copyWith(height: 1.4),
              ),
            ),
          ),
        ],
      ),
    );
  }

  static bool _isInjected(LLMMessage message) {
    final text = message.content ?? '';
    return message.role == LLMRole.user && text.startsWith(localNotesTitle);
  }

  static String _labelFor(LLMMessage message) {
    if (message.role == LLMRole.system) return 'System';
    if (_isInjected(message)) return 'Injected notes';
    if (message.role == LLMRole.user) return 'User';
    return message.role.name;
  }
}

class HistoryCard extends StatefulWidget {
  const HistoryCard({
    super.key,
    required this.prompt,
    required this.results,
    required this.showSimRag,
  });

  final PromptSpec? prompt;
  final Map<PromptCondition, RunRecord?> results;
  final bool showSimRag;

  @override
  State<HistoryCard> createState() => _HistoryCardState();
}

class _HistoryCardState extends State<HistoryCard> {
  final _boundaryKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hasResults = widget.prompt != null && widget.results.isNotEmpty;
    return RepaintBoundary(
      key: _boundaryKey,
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: Theme(
          data: theme.copyWith(dividerColor: Colors.transparent),
          child: ExpansionTile(
            title: GestureDetector(
              onLongPress: () => saveCardPng(context, _boundaryKey, 'history'),
              child: Text('History', style: theme.textTheme.titleMedium),
            ),
            childrenPadding: const EdgeInsets.fromLTRB(8, 0, 8, 8),
            children: [
              if (hasResults)
                ComparisonView(
                  prompt: widget.prompt!,
                  results: widget.results,
                  showSimRag: widget.showSimRag,
                )
              else
                Padding(
                  padding: const EdgeInsets.fromLTRB(8, 0, 8, 12),
                  child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'No comparison yet.',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _MessageBlock extends StatelessWidget {
  const _MessageBlock({
    required this.label,
    required this.text,
    required this.injected,
  });

  final String label;
  final String text;
  final bool injected;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return DecoratedBox(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 8, 4, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text(
                  label,
                  style: theme.textTheme.labelMedium?.copyWith(
                    color: injected ? scheme.primary : scheme.onSurfaceVariant,
                  ),
                ),
                const Spacer(),
                CopyIcon(text: text, label: label.toLowerCase()),
              ],
            ),
            const SizedBox(height: 4),
            Text(text.isEmpty ? emDash : text, style: theme.textTheme.bodyLarge),
          ],
        ),
      ),
    );
  }
}
