"""
Stage 13 — process/workflow analytics on the secondary event log.
Real, measured statistics only. Not merged with the primary dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DATA_RAW_DIR, PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "workflow_analytics.md"

sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(DATA_RAW_DIR / "Incident_Management_CSV.csv", sep=";", low_memory=False)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M")
    df = df.sort_values(["Case ID", "Timestamp"])

    # --- Resolution duration per case (created -> closed) ---
    created = df[df["Event"] == "Ticket created"].groupby("Case ID")["Timestamp"].min()
    closed = df[df["Event"] == "Ticket closed"].groupby("Case ID")["Timestamp"].min()
    duration = (closed - created).dropna()
    duration_hours = duration.dt.total_seconds() / 3600

    print(f"Cases with both created & closed events: {len(duration_hours)} of {df['Case ID'].nunique()}")
    print(f"Resolution duration (hours): mean={duration_hours.mean():.2f}, "
          f"median={duration_hours.median():.2f}, max={duration_hours.max():.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(duration_hours, bins=50, ax=ax)
    ax.set_xlabel("Resolution duration (hours)")
    ax.set_title("Distribution of case resolution duration")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_resolution_duration_dist.png", dpi=100)
    plt.close()

    # --- Escalation funnel ---
    cases = df["Case ID"].nunique()
    reaches_l2 = df[df["Event"] == "Level 1 escalates to level 2 support"]["Case ID"].nunique()
    reaches_l3 = df[df["Event"] == "Level 2 escalates to level 3 support"]["Case ID"].nunique()
    reopened = df[df["Event"] == "Ticket reopened by customer"]["Case ID"].nunique()
    rejected_l1 = df[df["Event"] == "Ticket rejected by level 1 support"]["Case ID"].nunique()

    print(f"\nEscalation funnel (of {cases} total cases):")
    print(f"  Escalated L1->L2: {reaches_l2} ({reaches_l2/cases*100:.1f}%)")
    print(f"  Escalated L2->L3: {reaches_l3} ({reaches_l3/cases*100:.1f}%)")
    print(f"  Reopened by customer: {reopened} ({reopened/cases*100:.1f}%)")
    print(f"  Rejected at L1: {rejected_l1} ({rejected_l1/cases*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 4))
    funnel = pd.Series({
        "All cases": cases,
        "Escalated to L2": reaches_l2,
        "Escalated to L3": reaches_l3,
    })
    funnel.plot(kind="bar", ax=ax, color=["steelblue", "orange", "crimson"])
    ax.set_ylabel("Number of cases")
    ax.set_title("Escalation funnel")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_escalation_funnel.png", dpi=100)
    plt.close()

    # --- Time spent per WIP stage (bottleneck identification) ---
    # df is already sorted by Case ID, Timestamp. For each row, the time to
    # the NEXT event in the same case is next_timestamp - timestamp. This
    # avoids the earlier merge-based approach, which broke on cases with
    # repeated stage occurrences (duplicate Case ID index caused incorrect
    # cross-pairing and produced impossible negative durations).
    df["next_timestamp"] = df.groupby("Case ID")["Timestamp"].shift(-1)
    df["time_to_next_hours"] = (df["next_timestamp"] - df["Timestamp"]).dt.total_seconds() / 3600

    wip_events = ["WIP - level 1 support", "WIP - level 2 support", "WIP - level 3 support"]
    stage_durations = {}
    for stage in wip_events:
        durations = df.loc[df["Event"] == stage, "time_to_next_hours"].dropna()
        assert (durations >= 0).all(), f"Negative duration found for {stage} — investigate before proceeding"
        stage_durations[stage] = durations

    print("\nAvg time spent per WIP stage (hours, to next event):")
    for stage, durations in stage_durations.items():
        print(f"  {stage}: mean={durations.mean():.2f}h (n={len(durations)})")

    fig, ax = plt.subplots(figsize=(8, 4))
    means = pd.Series({k: v.mean() for k, v in stage_durations.items()})
    means.plot(kind="barh", ax=ax, color="teal")
    ax.set_xlabel("Avg hours to next event")
    ax.set_title("Time spent per support level (bottleneck view)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_stage_bottleneck.png", dpi=100)
    plt.close()

    report_lines = [
        "# Workflow Analytics (Secondary Event Log)",
        "",
        f"Dataset: {cases} incidents, {len(df)} events. Not linked to the",
        "primary NLP dataset (no shared identifiers) — standalone process analysis.",
        "",
        "## Resolution duration",
        f"- Cases with both created & closed events: {len(duration_hours)} of {cases}",
        f"- Mean: {duration_hours.mean():.2f}h, Median: {duration_hours.median():.2f}h, Max: {duration_hours.max():.2f}h",
        "",
        "## Escalation funnel",
        f"- Escalated L1->L2: {reaches_l2} ({reaches_l2/cases*100:.1f}%)",
        f"- Escalated L2->L3: {reaches_l3} ({reaches_l3/cases*100:.1f}%)",
        f"- Reopened by customer: {reopened} ({reopened/cases*100:.1f}%)",
        f"- Rejected at L1: {rejected_l1} ({rejected_l1/cases*100:.1f}%)",
        "",
        "## Time spent per support level (hours, avg to next event)",
    ]
    for stage, durations in stage_durations.items():
        report_lines.append(f"- {stage}: {durations.mean():.2f}h (n={len(durations)})")

    report_lines += [
        "",
        "## Charts",
        "- `08_resolution_duration_dist.png`",
        "- `09_escalation_funnel.png`",
        "- `10_stage_bottleneck.png`",
    ]

    REPORT_PATH.write_text("\n".join(report_lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
