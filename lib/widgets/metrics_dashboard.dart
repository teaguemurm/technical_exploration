import 'package:flutter/material.dart';
import 'package:technical_exploration/system_metrics.dart';

/// A prominent, easy-to-read dashboard displaying hardware and generation metrics.
class MetricsDashboard extends StatelessWidget {
  const MetricsDashboard({
    super.key,
    required this.coldLoadMs,
    required this.baselineRssBytes,
    required this.idleModelRssBytes,
    required this.peakRssBytes,
    required this.currentRssBytes,
    required this.ttftMs,
    required this.tokensPerSecond,
    required this.evalCount,
    required this.generationDuration,
    required this.runCount,
    required this.throttlePercentage,
    required this.batteryTempCelsius,
  });

  final int? coldLoadMs;
  final int? baselineRssBytes;
  final int? idleModelRssBytes;
  final int? peakRssBytes;
  final int? currentRssBytes;
  final int? ttftMs;
  final double? tokensPerSecond;
  final int? evalCount;
  final Duration? generationDuration;
  final int runCount;
  final double? throttlePercentage;
  final double? batteryTempCelsius;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: colorScheme.outlineVariant.withValues(alpha: 0.5)),
      ),
      color: colorScheme.surfaceContainerHighest.withValues(alpha: 0.4),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.speed, size: 20, color: colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  'Hardware & Inference Telemetry',
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: colorScheme.primary,
                  ),
                ),
                const Spacer(),
                if (batteryTempCelsius != null)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: colorScheme.errorContainer.withValues(alpha: 0.6),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.thermostat, size: 14, color: colorScheme.onErrorContainer),
                        const SizedBox(width: 4),
                        Text(
                          '${batteryTempCelsius!.toStringAsFixed(1)}°C',
                          style: theme.textTheme.labelSmall?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: colorScheme.onErrorContainer,
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _MetricChip(
                  icon: Icons.hourglass_top,
                  label: 'Cold Load',
                  value: coldLoadMs != null ? '$coldLoadMs ms' : '--',
                  highlight: coldLoadMs != null,
                ),
                _MetricChip(
                  icon: Icons.memory,
                  label: 'Model Idle RAM',
                  value: idleModelRssBytes != null
                      ? formatBytes(idleModelRssBytes, signed: true)
                      : (currentRssBytes != null ? formatBytes(currentRssBytes) : '--'),
                  highlight: idleModelRssBytes != null,
                ),
                _MetricChip(
                  icon: Icons.compress,
                  label: 'Peak RAM',
                  value: formatBytes(peakRssBytes),
                ),
                _MetricChip(
                  icon: Icons.flash_on,
                  label: 'TTFT',
                  value: ttftMs != null ? '$ttftMs ms' : '--',
                  highlight: ttftMs != null,
                ),
                _MetricChip(
                  icon: Icons.trending_up,
                  label: 'Tokens / Sec',
                  value: tokensPerSecond != null
                      ? '${tokensPerSecond!.toStringAsFixed(2)} tok/s'
                      : '--',
                  highlight: tokensPerSecond != null,
                ),
                _MetricChip(
                  icon: Icons.tag,
                  label: 'Tokens Generated',
                  value: evalCount != null
                      ? '$evalCount tok (${(generationDuration?.inMilliseconds ?? 0) / 1000.0}s)'
                      : '--',
                ),
                if (runCount > 1 && throttlePercentage != null)
                  _MetricChip(
                    icon: Icons.local_fire_department,
                    label: 'Throttle (Run #$runCount)',
                    value: '${throttlePercentage! >= 0 ? '-' : '+'}${throttlePercentage!.abs().toStringAsFixed(1)}%',
                    isAlert: throttlePercentage! > 10.0,
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _MetricChip extends StatelessWidget {
  const _MetricChip({
    required this.icon,
    required this.label,
    required this.value,
    this.highlight = false,
    this.isAlert = false,
  });

  final IconData icon;
  final String label;
  final String value;
  final bool highlight;
  final bool isAlert;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    final bgColor = isAlert
        ? colorScheme.errorContainer.withValues(alpha: 0.8)
        : (highlight
            ? colorScheme.primaryContainer.withValues(alpha: 0.5)
            : colorScheme.surface);

    final textColor = isAlert
        ? colorScheme.onErrorContainer
        : (highlight ? colorScheme.onPrimaryContainer : colorScheme.onSurface);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: isAlert
              ? colorScheme.error
              : (highlight ? colorScheme.primary.withValues(alpha: 0.3) : colorScheme.outlineVariant.withValues(alpha: 0.4)),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: textColor),
          const SizedBox(width: 6),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                label,
                style: theme.textTheme.labelSmall?.copyWith(
                  color: textColor.withValues(alpha: 0.7),
                  fontSize: 10,
                ),
              ),
              Text(
                value,
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: textColor,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
