import 'package:flutter/material.dart';
import 'package:technical_exploration/messages.dart';
import 'package:technical_exploration/widgets/report_surface.dart';

class SetupCard extends StatelessWidget {
  const SetupCard({
    super.key,
    required this.modelName,
    required this.modelSizeLabel,
    required this.statusLine,
    required this.adapterLabel,
    required this.canPick,
    required this.canLoad,
    required this.onPickModel,
    required this.onLoad,
    required this.onPickAdapter,
    required this.onClearAdapter,
    required this.adapterSelected,
    required this.condition,
    required this.onCondition,
    required this.simulatedRag,
    required this.onSimulatedRag,
    required this.prompts,
    required this.selectedPromptId,
    required this.onSelectPrompt,
    required this.running,
    this.catalogError,
    this.errorText,
  });

  final String modelName;
  final String modelSizeLabel;
  final String statusLine;
  final String adapterLabel;
  final bool canPick;
  final bool canLoad;
  final VoidCallback onPickModel;
  final VoidCallback onLoad;
  final VoidCallback onPickAdapter;
  final VoidCallback onClearAdapter;
  final bool adapterSelected;
  final PromptCondition condition;
  final ValueChanged<PromptCondition> onCondition;
  final bool simulatedRag;
  final ValueChanged<bool> onSimulatedRag;
  final List<PromptSpec> prompts;
  final String? selectedPromptId;
  final ValueChanged<PromptSpec> onSelectPrompt;
  final bool running;
  final String? catalogError;
  final String? errorText;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final selected = _selectedPrompt();
    return ReportCard(
      title: 'Setup',
      saveName: 'setup',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      modelName,
                      style: theme.textTheme.bodyLarge?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      modelSizeLabel == emDash
                          ? statusLine
                          : '$modelSizeLabel · $statusLine',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              TextButton(
                onPressed: canPick ? onPickModel : null,
                child: const Text('Pick GGUF'),
              ),
              FilledButton(
                onPressed: canLoad ? onLoad : null,
                child: const Text('Load'),
              ),
            ],
          ),
          const SizedBox(height: reportGap),
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'LoRA adapter',
                      style: theme.textTheme.labelMedium?.copyWith(
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(adapterLabel, style: theme.textTheme.bodyLarge),
                  ],
                ),
              ),
              TextButton(
                onPressed: canPick ? onPickAdapter : null,
                child: const Text('Pick'),
              ),
              if (adapterSelected)
                TextButton(
                  onPressed: canPick ? onClearAdapter : null,
                  child: const Text('Clear'),
                ),
            ],
          ),
          const SizedBox(height: reportGap),
          Row(
            children: [
              Text('Condition', style: theme.textTheme.labelLarge),
              HelpIcon(title: 'Conditions', body: _conditionHelp),
            ],
          ),
          const SizedBox(height: 8),
          SegmentedButton<PromptCondition>(
            showSelectedIcon: false,
            style: const ButtonStyle(
              visualDensity: VisualDensity.compact,
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              padding: WidgetStatePropertyAll(
                EdgeInsets.symmetric(horizontal: 8, vertical: 10),
              ),
              textStyle: WidgetStatePropertyAll(TextStyle(fontSize: 13)),
            ),
            segments: const [
              ButtonSegment(
                value: PromptCondition.conditionA,
                label: Text('A · Base'),
              ),
              ButtonSegment(
                value: PromptCondition.conditionB,
                label: Text('B · Dataset'),
              ),
              ButtonSegment(
                value: PromptCondition.conditionC,
                label: Text('C · Constraints'),
              ),
            ],
            selected: {condition},
            onSelectionChanged: running
                ? null
                : (next) => onCondition(next.first),
          ),
          const SizedBox(height: reportGap),
          Row(
            children: [
              Switch(
                value: simulatedRag,
                onChanged: running ? null : onSimulatedRag,
              ),
              const SizedBox(width: 4),
              const Flexible(child: Text('Simulated RAG')),
              HelpIcon(title: 'Simulated RAG', body: _simRagHelp),
            ],
          ),
          const SizedBox(height: reportGap),
          Text('Visitor question', style: theme.textTheme.labelLarge),
          const SizedBox(height: 8),
          _QuestionField(
            text: selected?.text,
            enabled: prompts.isNotEmpty && !running,
            onTap: () => _openQuestions(context),
          ),
          if (catalogError != null) ...[
            const SizedBox(height: 8),
            Text(catalogError!, style: TextStyle(color: scheme.error)),
          ],
          if (errorText != null) ...[
            const SizedBox(height: 8),
            Text(errorText!, style: TextStyle(color: scheme.error)),
          ],
        ],
      ),
    );
  }

  PromptSpec? _selectedPrompt() {
    for (final prompt in prompts) {
      if (prompt.id == selectedPromptId) return prompt;
    }
    return null;
  }

  Future<void> _openQuestions(BuildContext context) async {
    final picked = await showModalBottomSheet<PromptSpec>(
      context: context,
      showDragHandle: true,
      isScrollControlled: true,
      builder: (context) {
        return SafeArea(
          child: ListView(
            shrinkWrap: true,
            children: [
              for (final prompt in prompts)
                ListTile(
                  title: Text(prompt.text),
                  selected: prompt.id == selectedPromptId,
                  onTap: () => Navigator.pop(context, prompt),
                ),
            ],
          ),
        );
      },
    );
    if (picked != null) onSelectPrompt(picked);
  }
}

class _QuestionField extends StatelessWidget {
  const _QuestionField({
    required this.text,
    required this.enabled,
    required this.onTap,
  });

  final String? text;
  final bool enabled;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: enabled ? onTap : null,
        borderRadius: BorderRadius.circular(8),
        child: InputDecorator(
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
            isDense: true,
            contentPadding: EdgeInsets.fromLTRB(12, 12, 8, 12),
            suffixIcon: Icon(Icons.arrow_drop_down),
          ),
          child: Text(
            text ?? emDash,
            style: theme.textTheme.bodyLarge,
          ),
        ),
      ),
    );
  }
}

String get _conditionHelp {
  final a = PromptCondition.conditionA.shortDescription;
  final b = PromptCondition.conditionB.shortDescription;
  final c = PromptCondition.conditionC.shortDescription;
  final bSim = PromptCondition.conditionBSimRag.shortDescription;
  final cSim = PromptCondition.conditionCSimRag.shortDescription;
  return 'A · Base. $a\n\n'
      'B · Dataset. $b\n\n'
      'C · Constraints. $c\n\n'
      'With simulated RAG on, B uses: $bSim\n\n'
      'With simulated RAG on, C uses: $cSim';
}

const _simRagHelp =
    'Off (default): Condition B and C receive all 10 locked notes.\n\n'
    'On: Condition B and C use fixed top-k notes (k=2) with near misses.';

class RunActionsBar extends StatelessWidget {
  const RunActionsBar({
    super.key,
    required this.running,
    required this.canRun,
    required this.blockedReason,
    required this.progressLabel,
    required this.onRun,
    required this.onStop,
    required this.onBatch,
    required this.onCompare,
    required this.onHeat,
    required this.compareLabel,
  });

  final bool running;
  final bool canRun;
  final String? blockedReason;
  final String? progressLabel;
  final VoidCallback onRun;
  final VoidCallback onStop;
  final VoidCallback onBatch;
  final VoidCallback onCompare;
  final VoidCallback onHeat;
  final String compareLabel;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            running
                ? FilledButton(onPressed: onStop, child: const Text('Stop'))
                : FilledButton(
                    onPressed: canRun ? onRun : null,
                    child: const Text('Run'),
                  ),
            const SizedBox(width: 8),
            FilledButton.tonal(
              onPressed: canRun ? onBatch : null,
              child: const Text('Run batch'),
            ),
            if (running && progressLabel != null) ...[
              const SizedBox(width: 12),
              Flexible(
                child: Text(
                  progressLabel!,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontFeatures: tabularFigures,
                  ),
                ),
              ),
            ],
            const Spacer(),
            PopupMenuButton<String>(
              onSelected: canRun
                  ? (value) {
                      if (value == 'compare') onCompare();
                      if (value == 'heat') onHeat();
                    }
                  : null,
              itemBuilder: (context) => [
                PopupMenuItem(
                  value: 'compare',
                  enabled: canRun,
                  child: Text(compareLabel),
                ),
                PopupMenuItem(
                  value: 'heat',
                  enabled: canRun,
                  child: const Text('10-run heat test'),
                ),
              ],
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                child: Text(
                  'More',
                  style: theme.textTheme.labelLarge?.copyWith(
                    color: canRun
                        ? theme.colorScheme.primary
                        : theme.colorScheme.onSurface.withValues(alpha: 0.38),
                  ),
                ),
              ),
            ),
          ],
        ),
        if (!running && blockedReason != null) ...[
          const SizedBox(height: 4),
          Text(
            blockedReason!,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ],
    );
  }
}
