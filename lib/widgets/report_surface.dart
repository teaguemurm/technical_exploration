import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:technical_exploration/system_metrics.dart';

const reportGap = 16.0;
const reportPad = 16.0;

const emDash = '—';

const tabularFigures = [FontFeature.tabularFigures()];

String formatGroupedInt(int value) {
  final negative = value < 0;
  final digits = value.abs().toString();
  final buffer = StringBuffer();
  for (var i = 0; i < digits.length; i++) {
    if (i > 0 && (digits.length - i) % 3 == 0) buffer.write(',');
    buffer.write(digits[i]);
  }
  return negative ? '-$buffer' : buffer.toString();
}

String formatParameterCount(int? count) {
  if (count == null || count <= 0) return emDash;
  return formatGroupedInt(count);
}

String formatFileSizeLabel(int? bytes) {
  if (bytes == null || bytes < 0) return emDash;
  if (bytes >= 1024 * 1024 * 1024) {
    return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB';
  }
  if (bytes >= 1024 * 1024) {
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
  if (bytes >= 1024) {
    return '${(bytes / 1024).toStringAsFixed(0)} KB';
  }
  return '$bytes B';
}

/// One decimal megabyte value, or an em dash. The unit is drawn separately.
String formatMbValue(int? bytes) {
  if (bytes == null) return emDash;
  return (bytes / (1024 * 1024)).toStringAsFixed(1);
}

String formatClock(DateTime? time) {
  if (time == null) return emDash;
  final local = time.toLocal();
  final hour = local.hour.toString().padLeft(2, '0');
  final minute = local.minute.toString().padLeft(2, '0');
  return '$hour:$minute';
}

Future<void> showHelpSheet(
  BuildContext context, {
  required String title,
  required String body,
}) {
  return showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    builder: (context) {
      final theme = Theme.of(context);
      return Padding(
        padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: theme.textTheme.titleMedium),
            const SizedBox(height: 12),
            Text(body, style: theme.textTheme.bodyMedium),
          ],
        ),
      );
    },
  );
}

class HelpIcon extends StatelessWidget {
  const HelpIcon({super.key, required this.title, required this.body});

  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      visualDensity: VisualDensity.compact,
      padding: EdgeInsets.zero,
      constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
      icon: const Icon(Icons.info_outline, size: 20),
      onPressed: () => showHelpSheet(context, title: title, body: body),
    );
  }
}

class ReportCard extends StatefulWidget {
  const ReportCard({
    super.key,
    required this.title,
    required this.saveName,
    required this.child,
    this.trailing,
  });

  final String title;
  final String saveName;
  final Widget child;
  final Widget? trailing;

  @override
  State<ReportCard> createState() => _ReportCardState();
}

class _ReportCardState extends State<ReportCard> {
  final _boundaryKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return RepaintBoundary(
      key: _boundaryKey,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(reportPad),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: GestureDetector(
                      onLongPress: () =>
                          saveCardPng(context, _boundaryKey, widget.saveName),
                      child: Text(
                        widget.title,
                        style: theme.textTheme.titleMedium,
                      ),
                    ),
                  ),
                  if (widget.trailing != null) widget.trailing!,
                ],
              ),
              const SizedBox(height: reportGap),
              widget.child,
            ],
          ),
        ),
      ),
    );
  }
}

Future<void> saveCardPng(
  BuildContext context,
  GlobalKey boundaryKey,
  String saveName,
) async {
  final messenger = ScaffoldMessenger.of(context);
  final boundary =
      boundaryKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
  if (boundary == null) {
    messenger.showSnackBar(
      const SnackBar(content: Text('Could not capture this card.')),
    );
    return;
  }
  try {
    final image = await boundary.toImage(pixelRatio: 3);
    final data = await image.toByteData(format: ui.ImageByteFormat.png);
    if (data == null) {
      messenger.showSnackBar(
        const SnackBar(content: Text('Could not encode this card.')),
      );
      return;
    }
    final dir = await BenchmarkExporter.getExportsDirectory();
    final now = DateTime.now();
    final stamp =
        '${now.year.toString().padLeft(4, '0')}'
        '${now.month.toString().padLeft(2, '0')}'
        '${now.day.toString().padLeft(2, '0')}_'
        '${now.hour.toString().padLeft(2, '0')}'
        '${now.minute.toString().padLeft(2, '0')}'
        '${now.second.toString().padLeft(2, '0')}';
    final file = File('${dir.path}/card_${saveName}_$stamp.png');
    await file.writeAsBytes(data.buffer.asUint8List(), flush: true);
    messenger.showSnackBar(SnackBar(content: Text('Saved ${file.path}')));
  } catch (error) {
    messenger.showSnackBar(
      SnackBar(content: Text('Could not save this card: $error')),
    );
  }
}

class CopyIcon extends StatelessWidget {
  const CopyIcon({super.key, required this.text, required this.label});

  final String text;
  final String label;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      visualDensity: VisualDensity.compact,
      padding: EdgeInsets.zero,
      constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
      icon: const Icon(Icons.copy_outlined, size: 18),
      onPressed: text.isEmpty
          ? null
          : () async {
              await Clipboard.setData(ClipboardData(text: text));
              if (!context.mounted) return;
              ScaffoldMessenger.of(
                context,
              ).showSnackBar(SnackBar(content: Text('Copied $label')));
            },
    );
  }
}
