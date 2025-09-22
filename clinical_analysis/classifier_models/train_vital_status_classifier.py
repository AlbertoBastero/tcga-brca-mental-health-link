import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifiers to predict vital_status with preprocessing and SHAP.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "clinical_csv_files",
                "clinical_patient_brca.csv",
            )
        ),
        help="Path to the clinical CSV file containing the target column.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="vital_status",
        help="Name of the target column (Alive/Dead).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "model_outputs")),
        help="Directory to save reports, plots, and artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction for train/test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--drop-columns",
        action="append",
        type=str,
        default=None,
        help=(
            "Columns to drop before modeling. Can be provided multiple times or as a comma-separated list. "
            "Example: --drop-columns death_days_to,age_at_initial_pathologic_diagnosis --drop-columns other_col"
        ),
    )
    parser.add_argument(
        "--drop-columns-file",
        type=str,
        default=None,
        help=(
            "Path to a text file listing columns to drop (one per line or comma-separated). "
            "If not provided, the script will look for 'columns_to_exclude.txt' next to this script."
        ),
    )
    return parser.parse_args()


ID_COLUMNS_DEFAULT = [
    "bcr_patient_uuid",
    "bcr_patient_barcode",
    "patient_id",
    "project_code",
]


def load_and_clean(csv_path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop the first two rows (non-data rows sometimes present in exports)
    df = df.iloc[2:].copy()

    # Normalize special missing tokens
    df = df.replace({"[Not Available]": np.nan})

    # Standardize target values and drop rows with missing/invalid targets
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    # Drop known header-like bogus row where the value equals the column name
    if "bcr_patient_uuid" in df.columns:
        df = df[df["bcr_patient_uuid"].astype(str) != "bcr_patient_uuid"].copy()

    # Strip whitespace and standardize case for target
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.replace("[\n\r]", " ", regex=True)
        .str.title()
    )

    # Treat placeholder values as missing
    df.loc[df[target_col].isin(["[Not Available]", "Nan", "Na", "None", "", " "]), target_col] = np.nan

    # Keep only Alive/Dead rows
    mask_valid = df[target_col].isin(["Alive", "Dead"]) & df[target_col].notna()
    df = df.loc[mask_valid].copy()

    return df


def split_feature_types(df: pd.DataFrame, target_col: str, id_columns: List[str]) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != target_col and c not in id_columns]

    # Attempt numeric coercion for better dtype inference
    df_infer = df[feature_cols].copy()
    for col in df_infer.columns:
        if df_infer[col].dtype == object:
            coerced = pd.to_numeric(df_infer[col], errors="coerce")
            # Consider a column numeric if at least 70% of non-null values are convertible
            non_null = df_infer[col].notna().sum()
            convertible = coerced.notna().sum()
            if non_null > 0 and convertible / max(non_null, 1) >= 0.7:
                df_infer[col] = coerced

    numeric_features = df_infer.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    # Add a preprocessing step to coerce numeric-like strings to numbers using a custom FunctionTransformer
    from sklearn.preprocessing import FunctionTransformer

    def _coerce_numeric_df(df_input: pd.DataFrame) -> pd.DataFrame:
        df_out = df_input.copy()
        for c in df_out.columns:
            if df_out[c].dtype == object:
                df_out[c] = coerce_numeric_like(df_out[c])
        return df_out

    numeric_pipeline = Pipeline(
        steps=[
            ("to_numeric", FunctionTransformer(_coerce_numeric_df, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def coerce_numeric_like(series: pd.Series) -> pd.Series:
    """Coerce a possibly object series with tokens like '>6', '<=10' into numeric.

    Extract the first numeric token (supports scientific notation), else NaN.
    """
    # Preserve existing NaNs
    mask_na = series.isna()
    s = series.astype(str)
    # Extract first numeric pattern
    extracted = s.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    coerced = pd.to_numeric(extracted, errors="coerce")
    coerced[mask_na] = np.nan
    return coerced


def make_models(random_state: int) -> Dict[str, object]:
    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, n_jobs=None, solver="lbfgs"
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=random_state, n_jobs=-1
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            learning_rate=0.1, max_depth=None, random_state=random_state
        ),
    }
    return models


def binarize_target(y: pd.Series) -> pd.Series:
    mapping = {"Alive": 0, "Dead": 1}
    y_std = y.map(mapping)
    if y_std.isna().any():
        raise ValueError("Target contains values other than 'Alive'/'Dead' after cleaning.")
    return y_std.astype(int)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    # Prefer the sklearn-native method when available
    try:
        names = preprocessor.get_feature_names_out()
        return names.tolist()
    except Exception:
        pass

    # Fallback: build names manually
    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            ohe: OneHotEncoder = transformer.named_steps["onehot"]
            try:
                ohe_names = ohe.get_feature_names_out().tolist()
            except Exception:
                ohe_names = ohe.get_feature_names_out(cols).tolist()
            feature_names.extend(ohe_names)
        else:
            # Numeric pipeline or passthrough
            if isinstance(cols, list):
                feature_names.extend(cols)
            else:
                feature_names.append(cols)
    return feature_names


def _aggregate_by_base_feature(feature_names: List[str], values: np.ndarray) -> pd.Series:
    """Aggregate OHE-level attributions to base feature names.

    Assumes OHE feature names from sklearn like 'cat__featureName_category'.
    If the transformer uses ColumnTransformer naming, names can be like 'cat__featureName_A', 'num__age'.
    We map everything before the last '_' after the original column name boundary.
    Conservative heuristic: split on '__' first to get transformer and original name, then take the original
    column name up to the first '[' or keep as-is if not OHE.
    """
    base_to_sum: Dict[str, float] = {}
    for fname, val in zip(feature_names, values):
        base = fname
        if "__" in base:
            parts = base.split("__", 1)
            base = parts[1]
        # For OHE, sklearn typically uses 'colname_category'. We take the part before the last '_'
        if "_" in base and not base.endswith("_" ):
            # Try to detect pattern colname_category. If the colname itself had underscores, this is ambiguous.
            # Heuristic: if splitting from the right yields at least 2 parts, take the left part as base col name.
            left, right = base.rsplit("_", 1)
            if right != "":
                base = left
        base_to_sum[base] = base_to_sum.get(base, 0.0) + float(abs(val))
    return pd.Series(base_to_sum)


def plot_top_features(model_name: str, feature_names: List[str], importances: np.ndarray, output_dir: str, top_k: int = 20) -> pd.DataFrame:
    imp_series_raw = pd.Series(importances, index=feature_names)
    imp_series_raw = imp_series_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Aggregate per base feature
    agg_series = _aggregate_by_base_feature(feature_names, imp_series_raw.values)
    agg_series = agg_series.sort_values(ascending=False)
    top = agg_series.head(top_k)

    # Save CSV
    top.to_csv(os.path.join(output_dir, f"{model_name}_top{top_k}_importances.csv"), header=["importance"])

    # Plot
    plt.figure(figsize=(10, max(4, int(0.4 * len(top)))))
    top.iloc[::-1].plot(kind="barh")
    plt.title(f"Top {top_k} features - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_top{top_k}_importances.png"), dpi=200)
    plt.close()

    return top.to_frame(name="importance")


def evaluate_and_report(model_name: str, y_test: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray, output_dir: str) -> Dict[str, float]:
    report = classification_report(y_test, y_pred, target_names=["Alive", "Dead"], digits=3)
    auc = roc_auc_score(y_test, y_proba)
    metrics = {"roc_auc": float(auc)}

    # Save text report
    with open(os.path.join(output_dir, f"{model_name}_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\n")
        f.write(json.dumps(metrics, indent=2))

    # Print to console
    print(f"\n==== {model_name} ====")
    print(report)
    print(f"ROC-AUC: {auc:.4f}")

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"), dpi=200)
    plt.close()

    return metrics


def compute_coefficients(clf, feature_names: List[str]) -> np.ndarray:
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return np.zeros(len(feature_names))
    # Binary classification: take the single row
    coef = np.asarray(coef)
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef[0]
    return coef


def compute_feature_importances_fallback(model, X_valid: np.ndarray, y_valid: np.ndarray, random_state: int) -> np.ndarray:
    perm = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=random_state, n_jobs=-1)
    return perm.importances_mean


def try_import_shap():
    try:
        import shap  # type: ignore
        return shap
    except Exception as e:
        print("Warning: SHAP is not available. Skipping SHAP plots.")
        print(str(e))
        return None


def shap_explain_tree_model(
    model_name: str,
    model,
    X_train_trans: np.ndarray,
    X_test_trans: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    sample_index: int = 0,
    max_samples: int = 200,
):
    shap = try_import_shap()
    if shap is None:
        return

    # Subsample for speed
    n_samples = X_test_trans.shape[0]
    if n_samples == 0:
        return
    idx = np.arange(n_samples)
    if n_samples > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(idx, size=max_samples, replace=False)
    X_test_sub = X_test_trans[idx]

    # Prefer TreeExplainer; fallback to generic Explainer on predict_proba
    explainer = None
    shap_values = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sub)
        # Newer shap returns Explanation; older returns list for binary
        if hasattr(shap_values, "values"):
            shap_vals_class1 = shap_values.values
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals_class1 = shap_values[1]
        else:
            shap_vals_class1 = shap_values
    except Exception:
        try:
            explainer = shap.Explainer(model.predict_proba, X_train_trans, feature_names=feature_names)
            shap_explanation = explainer(X_test_sub)
            # Expected shape: (n_samples, n_classes, n_features)
            vals = getattr(shap_explanation, "values", None)
            if vals is None:
                return
            if vals.ndim == 3:
                shap_vals_class1 = vals[:, 1, :]
            elif vals.ndim == 2:
                shap_vals_class1 = vals
            else:
                shap_vals_class1 = np.atleast_2d(vals)
        except Exception as e:
            print(f"Warning: Failed to compute SHAP values for {model_name}: {e}")
            return

    shap_vals_class1 = np.array(shap_vals_class1)
    if shap_vals_class1.ndim == 1:
        shap_vals_class1 = shap_vals_class1.reshape(1, -1)

    # Summary plot (aggregate per base feature for readability)
    try:
        # Aggregate SHAP absolute values per base feature
        shap_abs = np.abs(shap_vals_class1)
        imp_by_feature = _aggregate_by_base_feature(feature_names, shap_abs.mean(axis=0))
        imp_by_feature = imp_by_feature.sort_values(ascending=False)
        top = imp_by_feature.head(20)
        plt.figure(figsize=(10, max(4, int(0.4 * len(top)))))
        top.iloc[::-1].plot(kind="barh")
        plt.title(f"Top 20 SHAP (mean |value|) - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to save SHAP summary plot for {model_name}: {e}")

    # Force plot (save as HTML for interactivity)
    try:
        sample_index = min(max(0, sample_index), X_test_sub.shape[0] - 1)
        # Newer shap.force_plot can accept base_values + values + features
        # Try to get base value
        base_value = None
        try:
            if hasattr(explainer, "expected_value"):
                exp_val = explainer.expected_value
                if isinstance(exp_val, (list, np.ndarray)) and len(np.array(exp_val).shape) > 0:
                    # For binary classification, take index 1 if available
                    if isinstance(exp_val, list) and len(exp_val) > 1:
                        base_value = exp_val[1]
                    elif isinstance(exp_val, np.ndarray) and exp_val.size > 1:
                        base_value = exp_val.flat[1]
                    else:
                        base_value = np.array(exp_val).item()
                else:
                    base_value = float(exp_val)
        except Exception:
            base_value = None

        # For force plot, compress to top features for readability
        vals = shap_vals_class1[sample_index, :]
        top_idx = np.argsort(np.abs(vals))[-20:]
        vals_top = vals[top_idx]
        feats_top = X_test_sub[sample_index, :][top_idx]
        names_top = [feature_names[i] for i in top_idx]

        if base_value is not None:
            fp = shap.force_plot(
                base_value,
                vals_top,
                feats_top,
                feature_names=names_top,
                matplotlib=False,
                show=False,
            )
        else:
            fp = shap.force_plot(
                vals_top,
                matplotlib=False,
                show=False,
                feature_names=names_top,
            )

        out_html = os.path.join(output_dir, f"{model_name}_shap_force_sample{sample_index}.html")
        shap.save_html(out_html, fp)
    except Exception as e:
        print(f"Warning: Failed to save SHAP force plot for {model_name}: {e}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load and clean
    df = load_and_clean(args.csv_path, args.target)

    # 2) Drop ID columns
    id_columns = [c for c in ID_COLUMNS_DEFAULT if c in df.columns]
    df = df.drop(columns=id_columns, errors="ignore")

    # 2b) Optionally drop user-specified columns
    user_drop_cols: List[str] = []
    if args.drop_columns:
        for entry in args.drop_columns:
            if entry is None:
                continue
            # Support comma-separated lists
            parts = [p.strip() for p in str(entry).split(",") if p.strip()]
            user_drop_cols.extend(parts)
    # Read drop columns from file (explicit path or default next to script)
    drop_file = args.drop_columns_file
    if drop_file is None:
        default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "columns_to_exclude.txt"))
        if os.path.exists(default_path):
            drop_file = default_path
    if drop_file is not None and os.path.exists(drop_file):
        try:
            with open(drop_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = [p.strip() for p in line.replace("\t", ",").replace(";", ",").split(",") if p.strip()]
                    user_drop_cols.extend(parts)
        except Exception as e:
            print(f"Warning: Failed to read drop-columns file '{drop_file}': {e}")
    # Ensure target is not dropped accidentally
    user_drop_cols = [c for c in user_drop_cols if c != args.target]
    if user_drop_cols:
        df = df.drop(columns=[c for c in user_drop_cols if c in df.columns], errors="ignore")

    # 3) Split features
    numeric_features, categorical_features = split_feature_types(df, args.target, id_columns)

    # 4) Preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Prepare X, y
    y = binarize_target(df[args.target])
    X = df.drop(columns=[args.target])

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Fit preprocessor separately so we can reuse transformed features for SHAP/importance
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    # 6) Models
    models = make_models(args.random_state)

    all_metrics: Dict[str, Dict[str, float]] = {}

    for name, estimator in models.items():
        # Build a final pipeline for convenience during predict
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", estimator)])
        pipe.fit(X_train, y_train)

        # Predictions and metrics
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps["clf"], "decision_function"):
            # Scale decision scores to [0,1] via logistic for AUC computation
            scores = pipe.decision_function(X_test)
            y_proba = 1.0 / (1.0 + np.exp(-scores))
        else:
            # Fallback to predictions; AUC will be less meaningful
            y_proba = pipe.predict(X_test).astype(float)

        y_pred = pipe.predict(X_test)
        metrics = evaluate_and_report(name, y_test.values, y_proba, y_pred, args.output_dir)
        all_metrics[name] = metrics

        # 7) Feature importances
        importances: np.ndarray
        clf = pipe.named_steps["clf"]
        if isinstance(clf, LogisticRegression):
            importances = compute_coefficients(clf, feature_names)
            plot_top_features(name, feature_names, importances, args.output_dir)
        elif isinstance(clf, RandomForestClassifier):
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            else:
                importances = compute_feature_importances_fallback(clf, X_test_trans, y_test.values, args.random_state)
            plot_top_features(name, feature_names, importances, args.output_dir)
        elif isinstance(clf, HistGradientBoostingClassifier):
            # HGB does not expose feature_importances_ consistently; use permutation importance
            importances = compute_feature_importances_fallback(clf, X_test_trans, y_test.values, args.random_state)
            plot_top_features(name, feature_names, importances, args.output_dir)
        else:
            # Generic fallback
            importances = compute_feature_importances_fallback(clf, X_test_trans, y_test.values, args.random_state)
            plot_top_features(name, feature_names, importances, args.output_dir)

        # 8) SHAP explanations for tree models
        if isinstance(clf, (RandomForestClassifier, HistGradientBoostingClassifier)):
            shap_explain_tree_model(
                name,
                clf,
                X_train_trans,
                X_test_trans,
                feature_names,
                args.output_dir,
                sample_index=0,
            )

    # Save metrics JSON
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


