/// True only for the first protocol run after a cold load when sanity did not run.
///
/// [loadWasCold] is the load-level flag. It is false for every load after the
/// first one in this process. [runIndexSinceLoad] is 1-based and already
/// counts the run being recorded. [sanityRan] is true when the fixed sanity
/// prompts ran after that load: they exercise the loaded weights, so no
/// protocol row from that load is a cold inference. Skipping sanity (Mac only)
/// leaves [sanityRan] false and restores the run-1 rule.
bool isColdForThisRun({
  required bool loadWasCold,
  required int runIndexSinceLoad,
  required bool sanityRan,
}) => loadWasCold && runIndexSinceLoad == 1 && !sanityRan;

/// Null unless a thinking toggle was actually applied for this run.
///
/// This app does not inject `/no_think` or pass a chat-template thinking
/// flag, so callers pass `toggleApplied: false` and the export stays `N/A`.
bool? thinkingDisabledForRun({required bool toggleApplied}) =>
    toggleApplied ? true : null;
