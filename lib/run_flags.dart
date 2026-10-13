/// True only for the first run after a cold load.
///
/// [loadWasCold] is the load-level flag. It is false for every load after the
/// first one in this process. [runIndexSinceLoad] is 1-based and already
/// counts the run being recorded.
bool isColdForThisRun({
  required bool loadWasCold,
  required int runIndexSinceLoad,
}) =>
    loadWasCold && runIndexSinceLoad == 1;

/// Null unless a thinking toggle was actually applied for this run.
///
/// This app does not inject `/no_think` or pass a chat-template thinking
/// flag, so callers pass `toggleApplied: false` and the export stays `N/A`.
bool? thinkingDisabledForRun({required bool toggleApplied}) =>
    toggleApplied ? true : null;
