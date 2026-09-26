# Workflow Analytics (Secondary Event Log)

Dataset: 31588 incidents, 242901 events. Not linked to the
primary NLP dataset (no shared identifiers) — standalone process analysis.

## Resolution duration
- Cases with both created & closed events: 31588 of 31588
- Mean: 14.73h, Median: 12.53h, Max: 56.32h

## Escalation funnel
- Escalated L1->L2: 18216 (57.7%)
- Escalated L2->L3: 1979 (6.3%)
- Reopened by customer: 1188 (3.8%)
- Rejected at L1: 934 (3.0%)

## Time spent per support level (hours, avg to next event)
- WIP - level 1 support: 4.24h (n=32223)
- WIP - level 2 support: 3.94h (n=21955)
- WIP - level 3 support: 1.05h (n=3240)

## Charts
- `08_resolution_duration_dist.png`
- `09_escalation_funnel.png`
- `10_stage_bottleneck.png`
## Interpretation

- **High escalation rate**: 57.7% of all cases escalate from L1 to L2 — the
  majority of incidents are not resolved at first-line support. Only 6.3%
  reach L3, so L2 is where most complex resolution actually happens.
- **Stage duration decreases with escalation level** (L1: 4.24h -> L2: 3.94h
  -> L3: 1.05h to next event). This is a real, measured pattern, not an
  assumption — it could mean L3 (specialist/senior) staff resolve issues
  faster once truly escalated, or that L3 handles a narrower, more
  well-defined problem set (only 3,240 of 32,223 total WIP events ever
  reach L3). Both are plausible; the data alone doesn't distinguish them —
  stated as an open question for the report rather than a firm conclusion.
- **Reopened rate (3.8%) and L1 rejection rate (3.0%)** are both low,
  suggesting most closed tickets stay closed and most tickets are accepted
  at intake rather than bounced back.
- Mean total resolution duration is 14.73h (median 12.53h) — the gap between
  mean and median indicates a right-skewed distribution (a minority of
  long-running cases, up to 56.32h max, pull the mean above the median).

## Note on the initial bug found during this analysis
An earlier version of the stage-duration calculation produced impossible
negative durations for WIP - level 3 support, caused by a merge-based
approach mismatching event pairs on cases with repeated stage occurrences.
Fixed using a groupby+shift approach that pairs each event only with the
immediately next event in the same case, with an explicit assertion
guarding against negative durations going forward.
