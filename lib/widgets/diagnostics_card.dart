import 'dart:io';

import 'package:flutter/material.dart';
import 'package:technical_exploration/widgets/report_surface.dart';

class DiagnosticsCard extends StatefulWidget {
  const DiagnosticsCard({
    super.key,
    required this.canVerify,
    required this.onVerify,
    required this.skipSanity,
    required this.onSkipSanity,
    required this.kernelLine,
    this.statusMessage,
    required this.overheadLabel,
    required this.batteryLabel,
    required this.throttleLabel,
    required this.loadSplitLabel,
    required this.canExport,
    required this.onCopyCsv,
    required this.onCopyJson,
    required this.onExport,
    required this.onClearHistory,
    this.showShare = false,
    this.onShare,
  });

  final bool canVerify;
  final VoidCallback onVerify;
  final bool skipSanity;
  final ValueChanged<bool>? onSkipSanity;
  final String? kernelLine;
  final String? statusMessage;
  final String overheadLabel;
  final String batteryLabel;
  final String throttleLabel;
  final String? loadSplitLabel;
  final bool canExport;
  final VoidCallback onCopyCsv;
  final VoidCallback onCopyJson;
  final VoidCallback onExport;
  final VoidCallback onClearHistory;
  final bool showShare;
  final VoidCallback? onShare;

  @override
  State<DiagnosticsCard> createState() => _DiagnosticsCardState();
}

class _DiagnosticsCardState extends State<DiagnosticsCard> {
  final _boundaryKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final skipExplanation = Platform.isAndroid
        ? 'Ignored on Android. Sanity always runs before a protocol batch.'
        : 'Off by default. On: sanity does not run, so the first protocol run after a cold load stays cold.';
    return RepaintBoundary(
      key: _boundaryKey,
      child: Card(
        child: Theme(
          data: theme.copyWith(dividerColor: Colors.transparent),
          child: ExpansionTile(
            title: GestureDetector(
              onLongPress: () =>
                  saveCardPng(context, _boundaryKey, 'diagnostics'),
              child: Text('Diagnostics', style: theme.textTheme.titleMedium),
            ),
            childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            children: [
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton(
                  onPressed: widget.canVerify ? widget.onVerify : null,
                  child: const Text('Verify model'),
                ),
              ),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Skip sanity check'),
                subtitle: Text(skipExplanation),
                value: widget.skipSanity,
                onChanged: widget.onSkipSanity,
              ),
              const SizedBox(height: 8),
              _Line('Overhead', widget.overheadLabel),
              _Line('Battery', widget.batteryLabel),
              _Line('Throttle', widget.throttleLabel),
              if (widget.loadSplitLabel != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    widget.loadSplitLabel!,
                    style: theme.textTheme.bodySmall,
                  ),
                ),
              const SizedBox(height: 8),
              if (widget.statusMessage != null) ...[
                SelectableText(
                  widget.statusMessage!,
                  style: theme.textTheme.bodySmall,
                ),
                const SizedBox(height: 8),
              ],
              SelectableText(
                widget.kernelLine ?? 'Kernel line appears after load.',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                  fontFeatures: tabularFigures,
                ),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  TextButton(
                    onPressed: widget.canExport ? widget.onCopyCsv : null,
                    child: const Text('Copy CSV'),
                  ),
                  TextButton(
                    onPressed: widget.canExport ? widget.onCopyJson : null,
                    child: const Text('Copy JSON'),
                  ),
                  TextButton(
                    onPressed: widget.canExport ? widget.onExport : null,
                    child: const Text('Export files'),
                  ),
                  if (widget.showShare)
                    TextButton(
                      onPressed: widget.onShare,
                      child: const Text('Share'),
                    ),
                  TextButton(
                    onPressed: widget.onClearHistory,
                    child: const Text('Clear history'),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                'Long-press a card title to save a PNG.',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Line extends StatelessWidget {
  const _Line(this.label, this.value);

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(
            width: 96,
            child: Text(
              label,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: theme.textTheme.bodySmall?.copyWith(
                fontFeatures: tabularFigures,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
