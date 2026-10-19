import 'package:flutter/services.dart';
import 'package:technical_exploration/device_probe.dart';
import 'package:technical_exploration/model_integrity.dart';

/// Folder the Files app and Windows Explorer show for the public copies.
const publicExportDirectoryLabel = 'Download/technical-exploration';

/// One private file to copy into [publicExportDirectoryLabel].
class PublicExportRequest {
  const PublicExportRequest({
    required this.path,
    required this.name,
    required this.mime,
    required this.replace,
  });

  final String path;
  final String name;
  final String mime;

  /// True replaces an existing public file of the same name. False keeps the
  /// existing file and writes a suffixed name instead.
  final bool replace;

  Map<String, Object> toChannelMap() {
    return {'path': path, 'name': name, 'mime': mime, 'replace': replace};
  }
}

/// Names MediaStore actually wrote, which may include a suffix.
class PublishedExport {
  const PublishedExport({required this.name, required this.uri});

  final String name;
  final String uri;
}

class PublicExportBatch {
  const PublicExportBatch({required this.directory, required this.files});

  final String directory;
  final List<PublishedExport> files;

  List<String> get uris => [for (final file in files) file.uri];

  List<String> get names => [for (final file in files) file.name];
}

String exportFileName(String path) {
  final normalized = path.replaceAll('\\', '/');
  final slash = normalized.lastIndexOf('/');
  if (slash < 0 || slash == normalized.length - 1) {
    return slash < 0 ? normalized : '';
  }
  return normalized.substring(slash + 1);
}

/// CSV and JSON always. The hash cache only when its private file exists.
List<PublicExportRequest> publicExportRequests({
  required String csvPath,
  required String jsonPath,
  String? hashCachePath,
}) {
  return [
    PublicExportRequest(
      path: csvPath,
      name: exportFileName(csvPath),
      mime: 'text/csv',
      replace: false,
    ),
    PublicExportRequest(
      path: jsonPath,
      name: exportFileName(jsonPath),
      mime: 'application/json',
      replace: false,
    ),
    if (hashCachePath != null)
      PublicExportRequest(
        path: hashCachePath,
        name: modelHashCacheFileName,
        mime: 'application/json',
        replace: true,
      ),
  ];
}

String publicExportSavedMessage(String directory, List<String> names) {
  return 'Saved to $directory/\n${names.join('\n')}';
}

String publicExportFailureMessage(String detail) {
  return 'Could not copy exports to $publicExportDirectoryLabel ($detail). '
      'The app-private copies were still written.';
}

/// Copies finished private files into Download via MediaStore.
Future<PublicExportBatch> publishPublicExports(
  List<PublicExportRequest> files,
) async {
  final raw = await deviceChannel.invokeMethod<Map<Object?, Object?>>(
    'publishDownloads',
    {
      'files': [for (final file in files) file.toChannelMap()],
    },
  );
  if (raw == null) {
    throw PlatformException(
      code: 'publish',
      message: 'Download publish returned nothing',
    );
  }
  final directory = raw['directory']?.toString() ?? publicExportDirectoryLabel;
  final listed = raw['files'];
  if (listed is! List || listed.isEmpty) {
    throw PlatformException(
      code: 'publish',
      message: 'Download publish returned no files',
    );
  }
  final published = <PublishedExport>[];
  for (final entry in listed) {
    if (entry is! Map) {
      throw PlatformException(
        code: 'publish',
        message: 'Download publish returned a bad file entry',
      );
    }
    final name = entry['name']?.toString() ?? '';
    final uri = entry['uri']?.toString() ?? '';
    if (name.isEmpty || uri.isEmpty) {
      throw PlatformException(
        code: 'publish',
        message: 'Download publish omitted a file name or uri',
      );
    }
    published.add(PublishedExport(name: name, uri: uri));
  }
  return PublicExportBatch(directory: directory, files: published);
}

/// Opens the Android share sheet for the MediaStore URIs from the last publish.
Future<void> sharePublicExports(List<String> uris) {
  return deviceChannel.invokeMethod<void>('shareDownloads', {'uris': uris});
}
