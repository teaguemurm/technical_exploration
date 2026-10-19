import 'package:flutter_test/flutter_test.dart';
import 'package:technical_exploration/model_integrity.dart';
import 'package:technical_exploration/public_exports.dart';

void main() {
  test(
    'public export requests keep csv and json and replace only the cache',
    () {
      final requests = publicExportRequests(
        csvPath: '/data/exports/technical_exploration_runs_20261013_120000.csv',
        jsonPath: r'C:\exports\technical_exploration_runs_20261013_120000.json',
        hashCachePath: '/data/exports/$modelHashCacheFileName',
      );

      expect(requests, hasLength(3));
      expect(
        requests[0].name,
        'technical_exploration_runs_20261013_120000.csv',
      );
      expect(requests[0].mime, 'text/csv');
      expect(requests[0].replace, isFalse);
      expect(
        requests[1].name,
        'technical_exploration_runs_20261013_120000.json',
      );
      expect(requests[1].replace, isFalse);
      expect(requests[2].name, modelHashCacheFileName);
      expect(requests[2].replace, isTrue);
      expect(
        publicExportSavedMessage(publicExportDirectoryLabel, [
          requests[0].name,
          requests[2].name,
        ]),
        'Saved to Download/technical-exploration/\n'
        'technical_exploration_runs_20261013_120000.csv\n'
        '$modelHashCacheFileName',
      );
    },
  );

  test('a missing hash cache is left out of the public copy', () {
    final requests = publicExportRequests(
      csvPath: '/data/exports/runs.csv',
      jsonPath: '/data/exports/runs.json',
    );
    expect(requests.map((r) => r.name), ['runs.csv', 'runs.json']);
    expect(
      publicExportFailureMessage('short write'),
      contains('Download/technical-exploration'),
    );
    expect(
      publicExportFailureMessage('short write'),
      contains('app-private copies were still written'),
    );
  });
}
