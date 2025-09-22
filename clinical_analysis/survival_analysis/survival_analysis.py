"""
Survival analysis utilities for TCGA-BRCA clinical follow-up tables.

This script reproduces the Kaplan–Meier analyses from the notebook, with a CLI.
It can:
- Build a cohort with one latest follow-up row per patient across versions
- Plot overall KM curve
- Optionally, merge clustering labels and plot KM curves by cluster
- Compute log-rank tests and descriptive metrics per cluster
- Optional focused comparison between two clusters (Cox PH, RMST)

Example usage:
  python survival_analysis.py \
    --v15-path clinical_csv_files/clinical_follow_up_v1.5_brca.csv \
    --v21-path clinical_csv_files/clinical_follow_up_v2.1_brca.csv \
    --v40-path clinical_csv_files/clinical_follow_up_v4.0_brca.csv \
    --labels cluster_outputs/labels.csv \
    --label-id-col bcr_patient_uuid \
    --output-dir survival_outputs \
    --compare-clusters 1,2
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

# Non-interactive backend for script usage
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.statistics import (  # noqa: E402
    multivariate_logrank_test,
    pairwise_logrank_test,
    logrank_test,
)
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.utils import restricted_mean_survival_time, concordance_index  # noqa: E402


def _default_path(*parts: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *parts))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaplan–Meier survival analysis for TCGA-BRCA clinical follow-up",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--v15-path",
        default=_default_path("clinical_csv_files", "clinical_follow_up_v1.5_brca.csv"),
        help="Path to clinical_follow_up_v1.5_brca.csv",
    )
    parser.add_argument(
        "--v21-path",
        default=_default_path("clinical_csv_files", "clinical_follow_up_v2.1_brca.csv"),
        help="Path to clinical_follow_up_v2.1_brca.csv",
    )
    parser.add_argument(
        "--v40-path",
        default=_default_path("clinical_csv_files", "clinical_follow_up_v4.0_brca.csv"),
        help="Path to clinical_follow_up_v4.0_brca.csv",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional path to clustering labels CSV (must contain a 'cluster' column)",
    )
    parser.add_argument(
        "--label-id-col",
        default="bcr_patient_uuid",
        help="ID column name present in labels file used to merge with patients",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_path("survival_outputs"),
        help="Directory to save plots and reports",
    )
    parser.add_argument(
        "--compare-clusters",
        default="",
        help="Optional pair of clusters to compare, e.g. '1,2'",
    )

    return parser.parse_args(argv)


def _clean_vital_status(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace("[\n\r]", " ", regex=True).str.title()
    s = s.replace({"[Not Available]": np.nan, "Nan": np.nan, "Na": np.nan, "None": np.nan, "": np.nan, " ": np.nan})
    return s


def load_latest_followups(v15_path: str, v21_path: str, v40_path: str) -> pd.DataFrame:
    if not (os.path.exists(v15_path) and os.path.exists(v21_path) and os.path.exists(v40_path)):
        raise FileNotFoundError("One or more follow-up CSV paths do not exist.")

    v15 = pd.read_csv(v15_path, low_memory=False)
    v21 = pd.read_csv(v21_path, low_memory=False)
    v40 = pd.read_csv(v40_path, low_memory=False)

    # Add version order so that grouping by last keeps the latest
    v15["version"] = 1
    v21["version"] = 2
    v40["version"] = 3

    # Normalize vital_status for consistency
    for df in (v15, v21, v40):
        if "vital_status" in df.columns:
            df["vital_status"] = _clean_vital_status(df["vital_status"])

    df = pd.concat([v15, v21, v40], ignore_index=True)

    # Remove known sentinel/header-like rows sometimes present
    if "bcr_patient_uuid" in df.columns:
        sentinels = {"bcr_patient_uuid", "CDE_ID:"}
        df = df[~df["bcr_patient_uuid"].astype(str).str.strip().isin(sentinels)]

    # Ensure numeric time fields
    for col in ("death_days_to", "last_contact_days_to"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep last (latest version) record per patient
    missing_id = "bcr_patient_uuid" not in df.columns
    if missing_id:
        raise ValueError("Input tables must contain 'bcr_patient_uuid'.")

    df = df.sort_values(["bcr_patient_uuid", "version"])  # ascending: 1,2,3
    latest = df.groupby("bcr_patient_uuid").last().reset_index()

    # Construct KM-ready dataset
    if "vital_status" not in latest.columns:
        raise ValueError("Missing 'vital_status' column after merge.")

    latest["time"] = latest.apply(
        lambda row: row["death_days_to"] if pd.notna(row.get("death_days_to", np.nan)) else row.get("last_contact_days_to", np.nan),
        axis=1,
    )
    latest["event"] = latest["vital_status"].apply(lambda x: 1 if x == "Dead" else 0)

    km_data = latest[["bcr_patient_uuid", "time", "event"]].copy()

    # Clean: numeric and valid ranges
    km_data["time"] = pd.to_numeric(km_data["time"], errors="coerce")
    km_data["event"] = pd.to_numeric(km_data["event"], errors="coerce")
    before = len(km_data)
    km_data = km_data.dropna(subset=["time", "event"])  # lifelines allows time == 0
    km_data = km_data[(km_data["time"] >= 0) & (km_data["event"].isin([0, 1]))]
    after = len(km_data)
    print(f"[✓] Prepared KM dataset with N={after} (dropped {before - after})")

    return km_data


def plot_overall_km(km_data: pd.DataFrame, out_png: str) -> None:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=km_data["time"], event_observed=km_data["event"])

    plt.figure(figsize=(8, 6))
    kmf.plot_survival_function(ci_show=True)
    plt.title("Kaplan–Meier Survival Curve")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[✓] Saved overall KM plot → {out_png}")


def analyze_by_clusters(
    km_data: pd.DataFrame,
    labels_path: str,
    label_id_col: str,
    outdir: str,
    compare_clusters: Optional[Tuple[str, str]] = None,
) -> None:
    if not os.path.exists(labels_path):
        print(f"[⚠️] Labels file not found: {labels_path}. Skipping cluster analysis.")
        return

    labels = pd.read_csv(labels_path)
    if "cluster" not in labels.columns:
        raise ValueError("Labels CSV must contain a 'cluster' column.")
    if label_id_col not in labels.columns:
        raise ValueError(
            f"Labels CSV is missing the id column '{label_id_col}'. Use --label-id-col to match."
        )

    labels = labels[[label_id_col, "cluster"]].copy()
    labels[label_id_col] = labels[label_id_col].astype(str)
    labels["cluster"] = labels["cluster"].astype(str)

    df = km_data.copy()
    df[label_id_col] = df["bcr_patient_uuid"].astype(str)
    df = df.merge(labels, on=label_id_col, how="inner")

    # Basic cleaning
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df = df.dropna(subset=["time", "event"])  # ensure validity
    df = df[(df["time"] >= 0) & (df["event"].isin([0, 1]))]

    # Plot KM by cluster
    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    for clust, g in df.groupby("cluster"):
        kmf.fit(g["time"], g["event"], label=f"Cluster {clust} (n={len(g)})")
        kmf.plot_survival_function(ci_show=False)
    plt.title("Kaplan–Meier Survival by Cluster")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.legend(title="Clusters")
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join(outdir, "km_by_cluster.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[✓] Saved KM-by-cluster plot → {out_png}")

    # Log-rank tests
    logrank_txt = os.path.join(outdir, "logrank_tests.txt")
    with open(logrank_txt, "w", encoding="utf-8") as f:
        try:
            global_res = multivariate_logrank_test(df["time"], df["cluster"], df["event"])
            f.write("Global log-rank across clusters\n")
            f.write(str(global_res.summary))
            f.write("\n\n")
        except Exception as exc:
            f.write(f"Global log-rank failed: {exc}\n\n")

        try:
            pw = pairwise_logrank_test(
                event_durations=df["time"],
                groups=df["cluster"],
                event_observed=df["event"],
                p_adjust_method="holm",
            )
            f.write("Pairwise log-rank (Holm-adjusted)\n")
            f.write(str(pw.summary))
            f.write("\n\n")
        except Exception as exc:
            f.write(f"Pairwise log-rank failed: {exc}\n\n")

    print(f"[✓] Saved log-rank summaries → {logrank_txt}")

    # Descriptive metrics per cluster
    desc = []
    for clust, g in df.groupby("cluster"):
        kmf = KaplanMeierFitter().fit(g["time"], g["event"])
        median = kmf.median_survival_time_
        s_365 = float(kmf.predict(365)) if 365 <= kmf.timeline.max() else np.nan
        desc.append({"cluster": clust, "n": len(g), "median_survival": median, "S(365d)": s_365})
    desc_df = pd.DataFrame(desc).sort_values("cluster")
    desc_csv = os.path.join(outdir, "cluster_descriptives.csv")
    desc_df.to_csv(desc_csv, index=False)
    print(f"[✓] Saved cluster descriptives → {desc_csv}")

    # Optional focused comparison between two clusters
    if compare_clusters is not None:
        a, b = compare_clusters
        df_ab = df[df["cluster"].isin([str(a), str(b)])].copy()
        if df_ab.empty:
            print(f"[⚠️] No rows for clusters {a} and {b}. Skipping focused comparison.")
            return

        t1 = df_ab.loc[df_ab["cluster"] == str(a), "time"]
        e1 = df_ab.loc[df_ab["cluster"] == str(a), "event"]
        t2 = df_ab.loc[df_ab["cluster"] == str(b), "time"]
        e2 = df_ab.loc[df_ab["cluster"] == str(b), "event"]

        comp_txt = os.path.join(outdir, f"compare_{a}_vs_{b}.txt")
        with open(comp_txt, "w", encoding="utf-8") as f:
            # Log-rank
            try:
                lr = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
                f.write(f"Log-rank {a} vs {b}: chi2={lr.test_statistic:.4f}, p={lr.p_value:.6f}\n")
            except Exception as exc:
                f.write(f"Log-rank failed: {exc}\n")

            # Cox PH using binary indicator for cluster b
            try:
                df_cox = df_ab.copy()
                df_cox["cluster_b"] = (df_cox["cluster"] == str(b)).astype(int)
                cph = CoxPHFitter()
                cph.fit(df_cox[["time", "event", "cluster_b"]], duration_col="time", event_col="event")
                summ = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
                f.write("\nCox PH summary (cluster_b vs cluster_a):\n")
                f.write(str(summ))
                f.write("\n")
                cindex = concordance_index(df_cox["time"], -cph.predict_partial_hazard(df_cox), df_cox["event"])
                f.write(f"Concordance index (Cox): {cindex:.3f}\n")
            except Exception as exc:
                f.write(f"Cox PH failed: {exc}\n")

            # RMST difference at tau = 90th percentile
            try:
                tau = float(np.percentile(df_ab["time"], 90))
                km1 = KaplanMeierFitter().fit(t1, e1)
                km2 = KaplanMeierFitter().fit(t2, e2)
                rmst1 = float(restricted_mean_survival_time(km1, t=tau))
                rmst2 = float(restricted_mean_survival_time(km2, t=tau))
                f.write(
                    f"\nRMST @ {tau:.0f} days: Cluster{a}={rmst1:.1f}, Cluster{b}={rmst2:.1f}, Diff (b-a)={rmst2 - rmst1:.1f} days\n"
                )
            except Exception as exc:
                f.write(f"RMST failed: {exc}\n")

        print(f"[✓] Saved focused comparison {a} vs {b} → {comp_txt}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        km_data = load_latest_followups(args.v15_path, args.v21_path, args.v40_path)
    except Exception as exc:
        print(f"[✗] Failed to prepare KM dataset: {exc}", file=sys.stderr)
        return 2

    # Overall KM plot
    try:
        overall_png = os.path.join(args.output_dir, "overall_km.png")
        plot_overall_km(km_data, overall_png)
    except Exception as exc:
        print(f"[⚠️] Failed to save overall KM plot: {exc}")

    # Optional by-cluster analysis
    if args.labels is not None and str(args.labels).strip() != "":
        try:
            pair = None
            if args.compare_clusters:
                parts = [p.strip() for p in str(args.compare_clusters).split(",") if p.strip()]
                if len(parts) == 2:
                    pair = (parts[0], parts[1])
            analyze_by_clusters(
                km_data=km_data,
                labels_path=args.labels,
                label_id_col=args.label_id_col,
                outdir=args.output_dir,
                compare_clusters=pair,
            )
        except Exception as exc:
            print(f"[⚠️] Cluster-based analysis failed: {exc}")

    print("[✓] Survival analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


