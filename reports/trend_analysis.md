# Incident Trend Analysis

All charts in reports/figures/. Correlations are real Pearson r
(month index vs. metric) — |r| > 0.3 treated as a notable trend,
not a strict statistical threshold, since this is a short (12-month) series.

## Charts
1. `01_volume_by_issue_type.png` — monthly ticket volume by issue_type
2. `02_resolution_time_trend.png` — monthly avg resolution time
3. `03_resolution_rate_trend.png` — monthly resolution coverage %
4. `04_priority_mix_trend.png` — monthly priority mix %
5. `05_csat_trend.png` — monthly avg CSAT
6. `06_reopened_rate_trend.png` — monthly reopened rate %
7. `07_cluster_distribution_trend.png` — monthly cluster share % (Stage 11)

## Real computed trend correlations

- **Resolution time vs. time:** r = 0.006 (no clear trend)
- **Resolution coverage vs. time:** r = -0.041 (no clear trend)
- **Urgent-priority share vs. time:** r = 0.001 (no clear trend)
- **CSAT vs. time:** r = 0.122 (no clear trend)
- **Reopened rate vs. time:** r = 0.237 (no clear trend)
## Interpretation

None of the 5 metrics show a meaningful trend over the ~12-month period —
all |r| values are well below the 0.3 threshold used as a notability
cutoff, with the strongest being reopened rate at r=0.237 (still weak).

This is consistent with earlier findings: Stage 5's EDA found a uniform
region distribution, and this dataset's ticket volume, resolution time,
resolution coverage, CSAT, and reopened rate all appear stable/randomly
distributed over time rather than trending. The most defensible conclusion
is that this synthetic dataset was not generated with deliberate temporal
patterns (seasonality, degradation, improvement) built in — a real
limitation of synthetic data worth stating in the certification report,
not a sign the analysis method is wrong. A real production dataset would
be expected to show genuine trends (e.g. resolution time improving as a
team gains experience, or volume spiking around releases).
