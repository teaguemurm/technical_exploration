import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/system_metrics.dart';

/// Renders Condition outputs neutrally for direct comparative analysis without automated judging.
class ComparisonView extends StatefulWidget {
  const ComparisonView({
    super.key,
    required this.prompt,
    required this.results,
    this.showSimRag = false,
  });

  final PromptSpec prompt;
  final Map<PromptCondition, RunRecord?> results;
  final bool showSimRag;

  @override
  State<ComparisonView> createState() => _ComparisonViewState();
}

class _ComparisonViewState extends State<ComparisonView> {
  int _selectedTabIndex = 0;

  List<PromptCondition> get _activeConditions {
    if (widget.showSimRag) {
      return PromptCondition.values;
    }
    return [
      PromptCondition.conditionA,
      PromptCondition.conditionB,
      PromptCondition.conditionC,
    ];
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final conditions = _activeConditions;
    final allIndex = conditions.length;

    if (_selectedTabIndex > allIndex) {
      _selectedTabIndex = 0;
    }

    return Card(
      margin: const EdgeInsets.only(top: 16),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: colorScheme.outlineVariant.withValues(alpha: 0.6),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.compare_arrows, color: colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  widget.showSimRag
                      ? 'Condition Comparison (A vs B vs B-sim vs C vs C-sim)'
                      : 'Condition Comparison (A vs B vs C)',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              widget.prompt.text,
              style: theme.textTheme.bodySmall?.copyWith(
                color: colorScheme.onSurfaceVariant,
                fontStyle: FontStyle.italic,
              ),
            ),
            const SizedBox(height: 12),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: SegmentedButton<int>(
                segments: [
                  for (var i = 0; i < conditions.length; i++)
                    ButtonSegment(
                      value: i,
                      label: Text(conditions[i].shortCode),
                      tooltip: conditions[i].label,
                    ),
                  ButtonSegment(
                    value: allIndex,
                    label: Text(widget.showSimRag ? 'All 5' : 'All 3'),
                  ),
                ],
                selected: {_selectedTabIndex},
                onSelectionChanged: (set) {
                  setState(() => _selectedTabIndex = set.first);
                },
              ),
            ),
            const SizedBox(height: 16),
            if (_selectedTabIndex < allIndex)
              _buildSingleConditionCard(conditions[_selectedTabIndex])
            else
              Column(
                children: [
                  for (final condition in conditions) ...[
                    _buildSingleConditionCard(condition),
                    const SizedBox(height: 12),
                  ],
                ],
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildSingleConditionCard(PromptCondition condition) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final run = widget.results[condition];

    return Container(
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerLow,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: colorScheme.outlineVariant.withValues(alpha: 0.5),
        ),
      ),
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                condition.label,
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: colorScheme.primary,
                ),
              ),
              const Spacer(),
              if (run != null)
                IconButton(
                  icon: const Icon(Icons.copy, size: 16),
                  visualDensity: VisualDensity.compact,
                  tooltip: 'Copy output',
                  onPressed: () {
                    Clipboard.setData(ClipboardData(text: run.output));
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('Copied ${condition.code} output')),
                    );
                  },
                ),
            ],
          ),
          Text(
            condition.shortDescription,
            style: theme.textTheme.bodySmall?.copyWith(
              color: colorScheme.onSurfaceVariant,
              fontSize: 11,
            ),
          ),
          const SizedBox(height: 8),
          if (run == null)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16),
              child: Center(
                child: Text(
                  'No run recorded for this condition yet.',
                  style: TextStyle(fontStyle: FontStyle.italic),
                ),
              ),
            )
          else ...[
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: colorScheme.surface,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _stat('TTFT', '${run.ttftMs ?? "--"} ms'),
                  _stat(
                    'TPS',
                    run.tokensPerSecond != null
                        ? '${run.tokensPerSecond!.toStringAsFixed(1)} t/s'
                        : '--',
                  ),
                  _stat('Tokens', '${run.generatedTokenCount ?? "--"}'),
                  _stat('RAM Peak', formatBytes(run.ramPeakBytes)),
                ],
              ),
            ),
            const SizedBox(height: 8),
            SelectableText(
              run.output.trim().isEmpty ? '(Empty output)' : run.output.trim(),
              style: theme.textTheme.bodyMedium,
            ),
          ],
        ],
      ),
    );
  }

  Widget _stat(String label, String value) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w500),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }
}
