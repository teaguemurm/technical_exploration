import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/system_metrics.dart';

/// Renders Condition A, B, and C outputs for direct comparative analysis.
class ComparisonView extends StatefulWidget {
  const ComparisonView({
    super.key,
    required this.prompt,
    required this.results,
  });

  final PromptSpec prompt;
  final Map<PromptCondition, RunRecord?> results;

  @override
  State<ComparisonView> createState() => _ComparisonViewState();
}

class _ComparisonViewState extends State<ComparisonView> {
  int _selectedTabIndex = 0;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final conditions = PromptCondition.values;

    return Card(
      margin: const EdgeInsets.only(top: 16),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: colorScheme.outlineVariant.withValues(alpha: 0.6)),
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
                  'Condition Comparison (A vs B vs C)',
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
            SegmentedButton<int>(
              segments: [
                for (var i = 0; i < conditions.length; i++)
                  ButtonSegment(
                    value: i,
                    label: Text(conditions[i].code),
                  ),
                const ButtonSegment(
                  value: 3,
                  label: Text('All 3'),
                ),
              ],
              selected: {_selectedTabIndex},
              onSelectionChanged: (set) {
                setState(() => _selectedTabIndex = set.first);
              },
            ),
            const SizedBox(height: 16),
            if (_selectedTabIndex < 3)
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

    final isConditionC = condition == PromptCondition.conditionC;
    final refusalTriggered = run != null && isRefusalOutput(run.output);

    return Container(
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerLow,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: colorScheme.outlineVariant.withValues(alpha: 0.5)),
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
              if (run != null) ...[
                if (isConditionC)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    margin: const EdgeInsets.only(right: 8),
                    decoration: BoxDecoration(
                      color: refusalTriggered
                          ? colorScheme.primaryContainer
                          : colorScheme.tertiaryContainer,
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(
                      refusalTriggered ? 'Refusal Enforced' : 'Grounded Answer',
                      style: theme.textTheme.labelSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: refusalTriggered
                            ? colorScheme.onPrimaryContainer
                            : colorScheme.onTertiaryContainer,
                      ),
                    ),
                  ),
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
                  _stat('TPS', run.tokensPerSecond != null ? '${run.tokensPerSecond!.toStringAsFixed(1)} t/s' : '--'),
                  _stat('Tokens', '${run.evalCount ?? "--"}'),
                  _stat('RAM', formatBytes(run.peakRssBytes)),
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
          style: const TextStyle(fontSize: 10, color: Colors.grey),
        ),
        Text(
          value,
          style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }
}
