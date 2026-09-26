"""
Stage 12 — incident trend analysis.

Every chart answers a specific question. Alongside each chart, a real
Pearson correlation (month index vs. metric) is computed and printed —
an objective trend direction, not a subjective read of chart shape.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "trend_analysis.md"

sns.set_theme(style="whitegrid")


def load_data():
    clustered_path = DATA_PROCESSED_DIR / "tickets_clustered.csv"
    nlp_path = DATA_PROCESSED_DIR / "tickets_nlp.csv"
    if clustered_path.exists():
        print("Loading clustered dataset (includes cluster column).")
        df = pd.read_csv(clustered_path)
        has_cluster = True
    else:
        print("tickets_clustered.csv not found — loading tickets_nlp.csv (no cluster trend).")
        df = pd.read_csv(nlp_path)
        has_cluster = False
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    return df.dropna(subset=["created_at"]), has_cluster


def trend_correlation(monthly_series):
    """Pearson r between month order (0, 1, 2, ...) and the metric's
    values, using numpy directly to avoid pandas index-alignment
    issues (the two inputs have different index types)."""
    import numpy as np
    clean = monthly_series.dropna()
    if len(clean) < 3:
        return float("nan")
    x = np.arange(len(clean))
    y = clean.values
    return float(np.corrcoef(x, y)[0, 1])


def main():
    df, has_cluster = load_data()
    findings = []

    # 1. Monthly volume by issue_type
    monthly_issue = df.groupby([pd.Grouper(key="created_at", freq="MS"), "issue_type"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_issue.plot(kind="area", stacked=True, ax=ax, alpha=0.8)
    ax.set_title("Monthly ticket volume by issue_type")
    ax.set_ylabel("Tickets")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_volume_by_issue_type.png", dpi=100)
    plt.close()

    # 2. Monthly average resolution time
    monthly_res_time = df[df["has_resolution"]].groupby(pd.Grouper(key="created_at", freq="MS"))["resolution_time_hours"].mean()
    corr_res_time = trend_correlation(monthly_res_time)
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_res_time.plot(ax=ax, marker="o", color="darkorange")
    ax.set_title(f"Monthly avg resolution time (hours) — trend correlation: {corr_res_time:.3f}")
    ax.set_ylabel("Hours")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_resolution_time_trend.png", dpi=100)
    plt.close()
    findings.append(("Resolution time vs. time", corr_res_time))

    # 3. Monthly resolution rate
    monthly_res_rate = df.groupby(pd.Grouper(key="created_at", freq="MS"))["has_resolution"].mean() * 100
    corr_res_rate = trend_correlation(monthly_res_rate)
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_res_rate.plot(ax=ax, marker="o", color="seagreen")
    ax.set_title(f"Monthly resolution coverage % — trend correlation: {corr_res_rate:.3f}")
    ax.set_ylabel("% with recorded resolution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_resolution_rate_trend.png", dpi=100)
    plt.close()
    findings.append(("Resolution coverage vs. time", corr_res_rate))

    # 4. Monthly priority mix (%)
    monthly_priority = df.groupby([pd.Grouper(key="created_at", freq="MS"), "priority"]).size().unstack(fill_value=0)
    monthly_priority_pct = monthly_priority.div(monthly_priority.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_priority_pct.plot(kind="bar", stacked=True, ax=ax, width=0.9)
    ax.set_title("Monthly priority mix (%)")
    ax.set_ylabel("% of tickets")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_priority_mix_trend.png", dpi=100)
    plt.close()

    urgent_pct_corr = None
    if "urgent" in monthly_priority_pct.columns:
        urgent_pct_corr = trend_correlation(monthly_priority_pct["urgent"])
        findings.append(("Urgent-priority share vs. time", urgent_pct_corr))

    # 5. Monthly avg CSAT
    monthly_csat = df.groupby(pd.Grouper(key="created_at", freq="MS"))["csat_score"].mean()
    corr_csat = trend_correlation(monthly_csat)
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_csat.plot(ax=ax, marker="o", color="purple")
    ax.set_title(f"Monthly avg CSAT score — trend correlation: {corr_csat:.3f}")
    ax.set_ylabel("CSAT (1-5)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_csat_trend.png", dpi=100)
    plt.close()
    findings.append(("CSAT vs. time", corr_csat))

    # 6. Monthly reopened rate
    monthly_reopen = df.groupby(pd.Grouper(key="created_at", freq="MS"))["reopened"].mean() * 100
    corr_reopen = trend_correlation(monthly_reopen)
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_reopen.plot(ax=ax, marker="o", color="crimson")
    ax.set_title(f"Monthly reopened rate % — trend correlation: {corr_reopen:.3f}")
    ax.set_ylabel("% reopened")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_reopened_rate_trend.png", dpi=100)
    plt.close()
    findings.append(("Reopened rate vs. time", corr_reopen))

    # 7. Monthly cluster distribution (if available)
    if has_cluster:
        monthly_cluster = df.groupby([pd.Grouper(key="created_at", freq="MS"), "cluster"]).size().unstack(fill_value=0)
        monthly_cluster_pct = monthly_cluster.div(monthly_cluster.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        monthly_cluster_pct.plot(kind="area", stacked=True, ax=ax, alpha=0.8)
        ax.set_title("Monthly cluster share (%) — from Stage 11 clustering")
        ax.set_ylabel("% of tickets")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "07_cluster_distribution_trend.png", dpi=100)
        plt.close()

    print("\nComputed trend correlations (Pearson, month index vs. metric):")
    for name, corr in findings:
        direction = "increasing" if corr > 0.3 else "decreasing" if corr < -0.3 else "no clear trend"
        print(f"  {name}: r={corr:.3f} ({direction})")

    report_lines = [
        "# Incident Trend Analysis",
        "",
        "All charts in reports/figures/. Correlations are real Pearson r",
        "(month index vs. metric) — |r| > 0.3 treated as a notable trend,",
        "not a strict statistical threshold, since this is a short (12-month) series.",
        "",
        "## Charts",
        "1. `01_volume_by_issue_type.png` — monthly ticket volume by issue_type",
        "2. `02_resolution_time_trend.png` — monthly avg resolution time",
        "3. `03_resolution_rate_trend.png` — monthly resolution coverage %",
        "4. `04_priority_mix_trend.png` — monthly priority mix %",
        "5. `05_csat_trend.png` — monthly avg CSAT",
        "6. `06_reopened_rate_trend.png` — monthly reopened rate %",
    ]
    if has_cluster:
        report_lines.append("7. `07_cluster_distribution_trend.png` — monthly cluster share % (Stage 11)")

    report_lines += ["", "## Real computed trend correlations", ""]
    for name, corr in findings:
        direction = "increasing" if corr > 0.3 else "decreasing" if corr < -0.3 else "no clear trend"
        report_lines.append(f"- **{name}:** r = {corr:.3f} ({direction})")

    REPORT_PATH.write_text("\n".join(report_lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
