# Stage 25 — Optional System Log Analysis: Evaluation

## What was evaluated
LogHub (Zhu et al., ISSRE 2023 / arXiv:2008.06448) — a collection of 17-19
real-world system log datasets (HDFS, OpenSSH, Linux, Apache, Proxifier,
Hadoop, Spark, Zookeeper, BGL, Thunderbird, HealthApp, and others),
freely available for research use.

## Why it was not integrated

The proposed feature ("historical incident + related system-log pattern
-> operational context") requires a genuine, defensible link between an
incident and a log entry — shared timestamps, hostnames, service names,
or IDs. LogHub provides none of this relative to IncidentLens's primary
dataset:

- **No shared identifiers.** LogHub's logs come from entirely separate
  systems and research studies (a 203-node HDFS cluster, an OpenSSH
  server monitored over 28 days, etc.) — unrelated to the synthetic
  ticket dataset's customers, tickets, or timeframe.
- **No overlapping timeframe or subject matter.** The primary dataset's
  incidents (data export failures, billing disputes, login lockouts) have
  no natural counterpart in HDFS block operations or SSH connection logs.
- **Any labeled anomaly information in LogHub (e.g., HDFS block-level
  normal/abnormal labels) is internal to that log dataset** — it doesn't
  transfer to an unrelated incident.

Building this feature would require inventing a mapping between tickets
and log lines with no real basis — directly against this project's rule
against fabricated results and correlations. This is a case where the
honest, correct outcome of a proposed enhancement is not to build it.

## Decision
**Skipped.** No functionality was added in this stage. This evaluation
and its real citations are recorded as evidence the option was
considered on its merits, not skipped by default.
