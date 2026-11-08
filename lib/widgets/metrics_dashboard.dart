import 'package:flutter/material.dart';
import 'package:technical_exploration/widgets/report_surface.dart';

/// Six fixed tiles. The grid stays the same before and after a run.
class TelemetryCard extends StatelessWidget {
  const TelemetryCard({
    super.key,
    required this.coldLoadMs,
    required this.idleRamBytes,
    required this.peakRamBytes,
    required this.ttftMs,
    required this.tokensPerSecond,
    required this.tokensGenerated,
    required this.hasRun,
    required this.contextLine,
  });

  final int? coldLoadMs;
  final int? idleRamBytes;
  final int? peakRamBytes;
  final int? ttftMs;
  final double? tokensPerSecond;
  final int? tokensGenerated;
  final bool hasRun;
  final String contextLine;

  @override
  Widget build(BuildContext context) {
    final tiles = <_MetricTile>[
      _MetricTile(
        label: 'Cold load',
        value: coldLoadMs?.toString() ?? emDash,
        unit: 'ms',
      ),
      _MetricTile(
        label: 'Idle RAM',
        value: formatMbValue(idleRamBytes),
        unit: 'MB',
      ),
      _MetricTile(
        label: 'Peak RAM',
        value: formatMbValue(peakRamBytes),
        unit: 'MB',
      ),
      _MetricTile(label: 'TTFT', value: ttftMs?.toString() ?? emDash, unit: 'ms'),
      _MetricTile(
        label: 'Tokens/s',
        value: tokensPerSecond?.toStringAsFixed(1) ?? emDash,
        unit: 'tok/s',
      ),
      _MetricTile(
        label: 'Tokens generated',
        value: tokensGenerated?.toString() ?? emDash,
        unit: 'tok',
      ),
    ];

    return ReportCard(
      title: 'Telemetry',
      saveName: 'telemetry',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          LayoutBuilder(
            builder: (context, constraints) {
              const minTile = 112.0;
              const gap = 8.0;
              final columns = constraints.maxWidth >= minTile * 6 + gap * 5
                  ? 6
                  : 3;
              return _MetricGrid(columns: columns, tiles: tiles);
            },
          ),
          const SizedBox(height: 8),
          Visibility(
            visible: !hasRun,
            maintainSize: true,
            maintainAnimation: true,
            maintainState: true,
            child: Text(
              'Run to measure',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            contextLine,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricGrid extends StatelessWidget {
  const _MetricGrid({required this.columns, required this.tiles});

  final int columns;
  final List<_MetricTile> tiles;

  @override
  Widget build(BuildContext context) {
    final rows = <Widget>[];
    for (var i = 0; i < tiles.length; i += columns) {
      final end = i + columns > tiles.length ? tiles.length : i + columns;
      final slice = tiles.sublist(i, end);
      rows.add(
        Padding(
          padding: EdgeInsets.only(top: i == 0 ? 0 : 8),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              for (var c = 0; c < columns; c++)
                Expanded(
                  child: Padding(
                    padding: EdgeInsets.only(left: c == 0 ? 0 : 8),
                    child: c < slice.length ? slice[c] : const SizedBox.shrink(),
                  ),
                ),
            ],
          ),
        ),
      );
    }
    return Column(children: rows);
  }
}

class _MetricTile extends StatelessWidget {
  const _MetricTile({
    required this.label,
    required this.value,
    required this.unit,
  });

  final String label;
  final String value;
  final String unit;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final missing = value == emDash;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: theme.textTheme.labelMedium?.copyWith(
            color: scheme.onSurfaceVariant,
          ),
        ),
        const SizedBox(height: 4),
        Text.rich(
          TextSpan(
            children: [
              TextSpan(
                text: value,
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontFeatures: tabularFigures,
                  fontWeight: FontWeight.w600,
                  color: missing ? scheme.onSurfaceVariant : scheme.onSurface,
                ),
              ),
              TextSpan(
                text: ' $unit',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: scheme.onSurfaceVariant,
                  fontWeight: FontWeight.w400,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
