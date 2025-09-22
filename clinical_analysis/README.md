## Overview

This folder contains three reproducible Python scripts and their dependencies for clustering patients, training a classifier for vital status, and performing survival analysis.

- `cluster_patients_gower.py`: Hierarchical clustering using Gower distance on mixed clinical data, with Markdown report and optional plots.
- `train_vital_status_classifier.py`: Trains multiple classifiers (Logistic Regression, Random Forest, HistGradientBoosting) to predict `vital_status`, with reports and feature importance plots.
- `survival_analysis.py`: Builds a Kaplan–Meier cohort from TCGA clinical follow-up tables, plots overall survival, and optionally analyzes survival by clusters.

Install dependencies with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to run MCA visualization or SHAP explanations, ensure `prince` and `shap` are installed (already included in `requirements.txt`).

## 1) Clustering patients (Gower distance)

Example:

```bash
python cluster_patients_gower.py \
  --csv clinical_csv_files/clinical_patient_brca.csv \
  --id-col bcr_patient_uuid \
  --linkage average \
  --min-k 2 --max-k 8 \
  --outdir cluster_outputs \
  --plot-dendrogram \
  --mca-components 2 \
  --drug-csv clinical_csv_files/clinical_drug_brca.csv \
  --save-metrics \
  --annotate-outliers 10 \
  --annotate-label-col bcr_patient_uuid
```

Outputs in `cluster_outputs/`:
- `labels.csv`: cluster assignment (1-based) with ID or row index
- `silhouette_over_k.csv`: silhouette scores over k
- `cluster_report.md`: compact per-cluster profile summary
- Optional: `dendrogram.png`, `mca_scatter.png`, `per_sample_metrics.csv`

Notes:
- Requires `gower`. MCA plot requires `prince`.
- If `--id-col` is not provided, a `row_index` column is used in outputs instead.

## 2) Vital status classifier

Example:

```bash
python train_vital_status_classifier.py \
  --csv-path clinical_csv_files/clinical_patient_brca.csv \
  --target vital_status \
  --output-dir classification_model_outputs \
  --test-size 0.2 \
  --random-state 42 \
  --drop-columns-file columns_to_exclude.txt
```

Outputs in `classification_model_outputs/`:
- `{Model}_report.txt`, `{Model}_roc_curve.png`, `{Model}_top20_importances.(csv|png)`
- `metrics.json`: per-model metrics such as ROC-AUC

Notes:
- The script standardizes the `vital_status` column to `Alive`/`Dead`, drops ID-like columns, and handles numeric/categorical preprocessing.
- SHAP plots are generated for tree models if `shap` is available.

## 3) Survival analysis

Overall KM and optional cluster-wise analysis:

```bash
python survival_analysis.py \
  --v15-path clinical_csv_files/clinical_follow_up_v1.5_brca.csv \
  --v21-path clinical_csv_files/clinical_follow_up_v2.1_brca.csv \
  --v40-path clinical_csv_files/clinical_follow_up_v4.0_brca.csv \
  --labels cluster_outputs/labels.csv \
  --label-id-col bcr_patient_uuid \
  --output-dir survival_outputs \
  --compare-clusters 1,2
```

Outputs in `survival_outputs/`:
- `overall_km.png`: overall Kaplan–Meier curve
- If `--labels` is provided:
  - `km_by_cluster.png`: KM curves per cluster
  - `logrank_tests.txt`: global and pairwise log-rank summaries
  - `cluster_descriptives.csv`: per-cluster median survival and 1-year survival
  - `compare_1_vs_2.txt` (if `--compare-clusters 1,2` is used): log-rank, Cox PH summary, RMST

Notes:
- The script constructs a single latest follow-up record per patient by combining v1.5, v2.1, and v4.0 tables.
- By-cluster analysis requires that the labels CSV contain `cluster` and an ID column matching `--label-id-col`.

## Data layout

Place clinical CSVs under `clinical_csv_files/` (as in this repo):
- `clinical_follow_up_v1.5_brca.csv`
- `clinical_follow_up_v2.1_brca.csv`
- `clinical_follow_up_v4.0_brca.csv`
- Optional for clustering: `clinical_drug_brca.csv`

You can change paths via the CLI flags shown above.


