"""
Cluster breast-cancer patients using Hierarchical Clustering with Gower's distance.

This script clusters mostly categorical clinical data and produces labels
and a human-readable Markdown report with per-cluster profiles. It can also
optionally save a dendrogram and an MCA-based 2D scatter plot for visualization.

Example usage:
python cluster_patients_gower.py \
  --csv patients.csv \
  --id-col bcr_patient_uuid \
  --linkage average \
  --min-k 2 --max-k 8 \
  --plot-dendrogram \
  --mca-components 2

Dependencies:
- pandas, numpy, scikit-learn, scipy, matplotlib, gower, prince (only if MCA is requested)

If gower (or prince when requested) isn't installed, you'll get a clear error
message suggesting: pip install gower prince
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    import gower  # type: ignore
except Exception as exc:  # pragma: no cover
    print(
        "[✗] Missing dependency 'gower'. Install with: pip install gower",
        file=sys.stderr,
    )
    raise

# Third-party imports used later (conditional where possible)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from scipy.spatial.distance import squareform
import matplotlib

# Use non-interactive backend for script usage
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------- Data classes ------------------------------- #


@dataclass
class PreprocessResult:
    df_for_model: pd.DataFrame
    df_original_index: pd.Index
    id_series: Optional[pd.Series]
    categorical_cols: List[str]
    numeric_cols: List[str]


# ------------------------------ Core functions ----------------------------- #


EXPECTED_COLUMNS = [
    "gender",
    "race",
    "age_at_diagnosis",
    "tissue_source_site",
    "anatomic_neoplasm_subdivision",
    "first_surgical_procedure_other",
    "ajcc_nodes_pathologic_pn",
    "initial_pathologic_dx_year",
    "pr_status_ihc_percent_positive",
    "ajcc_pathologic_tumor_stage",
    "last_contact_days_to",
    "icd_o_3_histology",
    "histologic_diagnosis_other",
    "history_other_malignancy",
    "her2_fish_method",
    "metastasis_site",
    "ajcc_tumor_pathologic_pt",
    "method_initial_path_dx",
    "tumor_status",
    "er_positivity_method",
    "er_positivity_scale_other",
    "pr_positivity_define_method",
    "axillary_staging_method"
]

# Expected columns by Julie after classification
EXPECTED_COLUMNS = [
    "tumor_status",
    "ajcc_staging_edition",         
    "tissue_source_site",                
    "initial_pathologic_dx_year",
    "margin_status",
    "ajcc_nodes_pathologic_pn",
    "age_at_diagnosis",
    "her2_status_by_ihc",    
    "method_initial_path_dx",            
    "lymph_nodes_examined_count",        
    "her2_ihc_score",                    
    "metastatic_tumor_indicator",       
    "axillary_staging_method",           
    "anatomic_neoplasm_subdivision",     
    "her2_fish_status",                  
    "lymph_nodes_examined_he_count",     
    "ajcc_pathologic_tumor_stage",       
    "er_status_ihc_Percent_Positive",    
    "menopause_status",
    "vital_status"]

# Columns used by Julie-
# EXPECTED_COLUMNS = [
#     "gender",
#     "race",
#     "age_at_diagnosis",
#     # "ajcc_pathologic_tumor_stage",
# ]


# --------------------------- Drug cleaning helpers -------------------------- #


# Mapping dictionary for drug name cleaning (normalized matching applied)
drug_merge_dict: Dict[str, str] = {
    # 5-Fluorouracil
    "5 fluorouracil": "5-Fluorouracil",
    "5-FU": "5-Fluorouracil",
    "5-Flourouracil": "5-Fluorouracil",
    "FLOUROURACIL": "5-Fluorouracil",
    "FLUOROURACIL": "5-Fluorouracil",
    "Fluorouracil": "5-Fluorouracil",

    # Adriamycin/Doxorubicin
    "ADRIAMYCIN": "Doxorubicin",
    "Adriamycin": "Doxorubicin",
    "Adriamyicin": "Doxorubicin",
    "Adrimycin": "Doxorubicin",
    "adriamicin": "Doxorubicin",
    "adriamycin": "Doxorubicin",
    "adriamycin+cuclophosphamide": "Doxorubicin",
    "adriamycin+cyclophosphamid": "Doxorubicin",
    "adriamycin+cyclophosphamide": "Doxorubicin",
    "adrimicin+cyclophosphamide": "Doxorubicin",
    "adrimycin+cyclophosphamide": "Doxorubicin",
    "DOXORUBICIN": "Doxorubicin",
    "Doxorubicin": "Doxorubicin",
    "Doxorubicin Liposome": "Doxorubicin",
    "Doxorubicinum": "Doxorubicin",
    "doxorubicin HCL": "Doxorubicin",
    "doxorubicin+ cyclophosphamide": "Doxorubicin",
    "doxorubicin+cyclophosphamid": "Doxorubicin",
    "doxorubicine": "Doxorubicin",
    "doxorubicine cyclophosphamide tamoxifen": "Doxorubicin",
    "doxorubicine+cyclophosphamide": "Doxorubicin",
    "doxorubicine+cyclophosphamide+tamoxifen": "Doxorubicin",

    # Cyclophosphamide/Cytoxan
    "CYCLOPHOSPHAMIDE": "Cyclophosphamide",
    "Cyclophosphamide": "Cyclophosphamide",
    "Cyclophasphamide": "Cyclophosphamide",
    "Cyclophospamide": "Cyclophosphamide",
    "Cyclophosphane": "Cyclophosphamide",
    "cyclophosphamid": "Cyclophosphamide",
    "cyclophosphamide": "Cyclophosphamide",
    "cyclophosphamidum": "Cyclophosphamide",
    "CYTOXAN": "Cyclophosphamide",
    "Cyotxan": "Cyclophosphamide",
    "Cytoxan": "Cyclophosphamide",
    "Cytoxen": "Cyclophosphamide",
    "cytoxan": "Cyclophosphamide",
    "Cytoxan and Taxotere": "Cyclophosphamide",

    # Paclitaxel/Taxol
    "PACLITAXEL": "Paclitaxel",
    "Paclitaxel": "Paclitaxel",
    "Albumin-Bound Paclitaxel": "Paclitaxel",
    "Paclitaxel (Protein-Bound)": "Paclitaxel",
    "paclitaxel": "Paclitaxel",
    "paclitaxelum": "Paclitaxel",
    "TAXOL": "Paclitaxel",
    "Taxol": "Paclitaxel",
    "taxol": "Paclitaxel",
    "taxol+adriamycin+cyclophosphamide+herceptin": "Paclitaxel",

    # Docetaxel/Taxotere
    "DOCETAXEL": "Docetaxel",
    "Docetaxel": "Docetaxel",
    "Doxetaxel": "Docetaxel",
    "TAXOTERE": "Docetaxel",
    "Taxotere": "Docetaxel",
    "taxotere": "Docetaxel",
    "Taxotere/Cytoxan": "Docetaxel",

    # Tamoxifen
    "TAMOXIFEN": "Tamoxifen",
    "Tamoxifen": "Tamoxifen",
    "tamoxifen": "Tamoxifen",
    "tamoxifen citrate": "Tamoxifen",
    "TAMOXIFEN (NOVADEX)": "Tamoxifen",
    "Nolvadex": "Tamoxifen",
    "nolvadex": "Tamoxifen",
    "tamoxiphen+anastrazolum": "Tamoxifen",
    "tamoxiphene": "Tamoxifen",
    "tamoxiphene+anastrozolum": "Tamoxifen",
    "tamoxiphene+leuporeline+gosereline": "Tamoxifen",

    # Letrozole/Femara
    "LETROZOLE": "Letrozole",
    "Letrozole": "Letrozole",
    "letrozole": "Letrozole",
    "LETROZOLE (FEMARA)": "Letrozole",
    "Femara": "Letrozole",
    "FEMARA": "Letrozole",
    "Femara (Letrozole)": "Letrozole",
    "letrozolum": "Letrozole",
    "Letrozol": "Letrozole",

    # Exemestane/Aromasin
    "EXEMESTANE": "Exemestane",
    "Exemestane": "Exemestane",
    "EXEMESTANE (AROMASIN)": "Exemestane",
    "AROMASIN (EXEMESTANE)": "Exemestane",
    "Aromasin": "Exemestane",
    "aromasin": "Exemestane",
    "aromatase exemestane": "Exemestane",

    # Anastrozole/Arimidex
    "ANASTROZOLE": "Anastrozole",
    "Anastrozole": "Anastrozole",
    "Anastrazole": "Anastrozole",
    "ANASTROZOLE (ARIMIDEX)": "Anastrozole",
    "ARIMIDEX": "Anastrozole",
    "Arimidex": "Anastrozole",
    "ARIMIDEX (ANASTROZOLE)": "Anastrozole",
    "Arimidex (Anastrozole)": "Anastrozole",
    "anastrozolum": "Anastrozole",
    "arimidex": "Anastrozole",

    # Bevacizumab/Avastin
    "BEVACIZUMAB": "Bevacizumab",
    "Bevacizumab": "Bevacizumab",
    "BEVACIZUMAB (AVASTIN)/PLACEBO PROVIDED BY STUDY": "Bevacizumab",
    "Bevacizumab or Placebo": "Bevacizumab",
    "AVASTIN": "Bevacizumab",
    "Avastin": "Bevacizumab",
    "avastin": "Bevacizumab",

    # Capecitabine/Xeloda
    "CAPECITABINE": "Capecitabine",
    "Capecetabine": "Capecitabine",
    "XELODA": "Capecitabine",
    "Xeloda": "Capecitabine",
    "Xeloda (Capecitabine)": "Capecitabine",

    # Gemcitabine/Gemzar
    "GEMZAR": "Gemcitabine",
    "Gemzar": "Gemcitabine",
    "Gemcitabine": "Gemcitabine",
    "gemcitabine": "Gemcitabine",

    # Trastuzumab/Herceptin
    "TRASTUZUMAB": "Trastuzumab",
    "Trastuzumab": "Trastuzumab",
    "Trustuzumab": "Trastuzumab",
    "HERCEPTIN": "Trastuzumab",
    "Herceptin": "Trastuzumab",
    "herceptin": "Trastuzumab",

    # Zoledronic Acid/Zometa/Xgeva
    "ZOLEDRONIC ACID": "Zoledronic Acid",
    "Zoledronic Acid": "Zoledronic Acid",
    "Zoledronic acid": "Zoledronic Acid",
    "zoledronic acid": "Zoledronic Acid",
    "Zometa": "Zoledronic Acid",
    "Xgeva": "Zoledronic Acid",

    # Fulvestrant/Faslodex
    "FULVESTRANT": "Fulvestrant",
    "Fulvestrant": "Fulvestrant",
    "Fulvestrant (Faslodex)": "Fulvestrant",
    "Faslodex": "Fulvestrant",
    "faslodex": "Fulvestrant",

    # Clodronate/Clodronic Acid
    "Clodronate": "Clodronate",
    "clodronate": "Clodronate",
    "Clodronic acid": "Clodronate",
    "clodronic acid": "Clodronate",
}


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    # Normalize whitespace and case for robust matching
    txt = str(value).strip()
    txt = " ".join(txt.split())
    return txt


def _build_normalized_drug_map(mapping: Dict[str, str]) -> Dict[str, str]:
    norm_map: Dict[str, str] = {}
    for k, v in mapping.items():
        norm_map[_normalize_text(k).lower()] = v
    return norm_map


DRUG_NAME_MAP_NORM = _build_normalized_drug_map(drug_merge_dict)


def _clean_single_drug_name(raw_value: object) -> Optional[str]:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return None
    txt = _normalize_text(str(raw_value))
    if txt == "" or txt in {"[Not Available]", "[Unknown]", "[Not Applicable]"}:
        return None

    # Exact match first
    key = txt.lower()
    if key in DRUG_NAME_MAP_NORM:
        return DRUG_NAME_MAP_NORM[key]

    # Otherwise, split on common separators and map each token
    import re

    parts = re.split(r"[+/,;&]", txt)
    cleaned_parts: List[str] = []
    seen: set = set()
    for part in parts:
        p = _normalize_text(part)
        if p == "":
            continue
        k = p.lower()
        mapped = DRUG_NAME_MAP_NORM.get(k, None)
        candidate = mapped if mapped is not None else p.title()
        if candidate not in seen:
            seen.add(candidate)
            cleaned_parts.append(candidate)

    if not cleaned_parts:
        return None
    return " + ".join(cleaned_parts)


def load_and_clean_drug_feature(
    drug_csv_path: str,
    id_col: str,
) -> pd.DataFrame:
    """Read drug CSV, clean names, and aggregate per patient into a single 'drug' value.

    Returns a DataFrame with columns [id_col, 'drug'] ready to merge.
    """
    if not os.path.exists(drug_csv_path):
        raise FileNotFoundError(f"Drug CSV not found: {drug_csv_path}")

    drug_df = pd.read_csv(drug_csv_path)

    # Drop known non-data rows sometimes present in TCGA exports
    if id_col not in drug_df.columns:
        raise ValueError(
            f"Drug file missing id column '{id_col}'. Available columns: {list(drug_df.columns)}"
        )
    drug_df = drug_df[~drug_df[id_col].astype(str).str.strip().isin({id_col, "CDE_ID:"})]

    # Find a drug name column among expected options
    candidate_cols = [
        "pharmaceutical_therapy_drug_name",
        "drug_name",
    ]
    drug_name_col = None
    for c in candidate_cols:
        if c in drug_df.columns:
            drug_name_col = c
            break
    if drug_name_col is None:
        raise ValueError(
            "Could not locate drug name column. Expected one of: " + ", ".join(candidate_cols)
        )

    # Clean individual entries
    drug_series = drug_df[drug_name_col].apply(_clean_single_drug_name)
    tmp = pd.DataFrame({id_col: drug_df[id_col].astype(str), "_drug_clean": drug_series})

    def _aggregate(names: pd.Series) -> Optional[str]:
        # Flatten split components to ensure deduplication across rows
        tokens: List[str] = []
        seen: set = set()
        for val in names.dropna():
            for token in str(val).split(" + "):
                token = _normalize_text(token)
                if token and token not in seen:
                    seen.add(token)
                    tokens.append(token)
        if not tokens:
            return None
        return " + ".join(tokens)

    agg_series = tmp.groupby(id_col)["_drug_clean"].apply(_aggregate)
    agg = agg_series.reset_index(name="drug")
    return agg[[id_col, "drug"]]


# --------------------------- Numeric parsing helpers --------------------------- #


def _parse_percent_or_range_to_float(value: object) -> Optional[float]:
    """Parse values like '50%', '50 - 65%', '50-65', '60' into a float.

    - If a range is provided, returns the mean of the numeric endpoints.
    - Returns None for empty/unknown tokens.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s == "" or s in {"[Not Available]", "[Unknown]", "[Not Applicable]"}:
        return None
    import re

    # Extract all numeric tokens (supports decimals), ignore percent signs and text
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums:
        return None
    vals = [float(x) for x in nums]
    if len(vals) == 1:
        return vals[0]
    # For ranges or multiple numbers, use the mean as a robust representative
    return float(np.mean(vals))


def load_and_preprocess(
    csv_path: str,
    id_col: Optional[str],
    impute_categorical: str,
    impute_age_strategy: str,
    sample: Optional[int] = None,
    drug_csv: Optional[str] = None,
    exclude_ids: Optional[str] = None,
) -> PreprocessResult:
    """Load CSV, coerce types, and impute missing values.

    - Keep `id_col` intact if provided (not used for clustering).
    - Ensure numeric for age; impute with median/mean.
    - For categoricals: strip whitespace, standardize empties to NaN, then impute with provided value.
    - Only cluster on EXPECTED_COLUMNS (no leakage).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop known non-data rows sometimes present in TCGA exports
    sentinel_values = {"bcr_patient_uuid", "CDE_ID:"}
    for candidate_col in [id_col, "bcr_patient_uuid"]:
        if candidate_col is not None and candidate_col in df.columns:
            df = df[~df[candidate_col].astype(str).str.strip().isin(sentinel_values)]

    # Exclude rows by id if requested (before sampling)
    if exclude_ids is not None:
        if id_col is None:
            raise ValueError("--exclude-ids requires --id-col to know which column to match")
        excl_set = _load_exclusion_ids(exclude_ids, id_col)
        if len(excl_set) > 0:
            before = len(df)
            if id_col not in df.columns:
                raise ValueError(f"--exclude-ids given but id column '{id_col}' not found in main CSV")
            df[id_col] = df[id_col].astype(str)
            df = df[~df[id_col].astype(str).isin(excl_set)].copy()
            removed = before - len(df)
            print(f"[✓] Excluded {removed} rows based on --exclude-ids")

    if sample is not None and sample > 0:
        df = df.head(sample)

    if df.shape[0] < 3:
        raise ValueError(
            "Dataset has fewer than 3 rows. Increase data size or lower requirements."
        )

    # Keep original index for outputs when no ID column is provided
    original_index = df.index.copy()

    # Optional ID column
    id_series: Optional[pd.Series] = None
    if id_col is not None:
        if id_col not in df.columns:
            raise ValueError(
                f"--id-col '{id_col}' not found in CSV columns: {list(df.columns)}"
            )
        id_series = df[id_col].copy()

    # Optional drug merge before selecting model columns
    if drug_csv is not None:
        if id_col is None:
            raise ValueError("--drug-csv requires --id-col to merge by patient id")
        print("[ℹ️] Loading and cleaning drug feature...")
        drug_feature_df = load_and_clean_drug_feature(drug_csv, id_col=id_col)
        # ensure id type consistency
        df[id_col] = df[id_col].astype(str)
        drug_feature_df[id_col] = drug_feature_df[id_col].astype(str)
        df = df.merge(drug_feature_df, on=id_col, how="left")
        print("[✓] Merged drug feature")

    # Validate expected columns presence
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) + "\n"
            f"Expected exactly: {EXPECTED_COLUMNS}"
        )

    # Restrict to the expected columns plus optional 'drug' if present
    model_columns = list(EXPECTED_COLUMNS)
    if "drug" in df.columns:
        model_columns.append("drug")
    df = df[model_columns].copy()

    # Identify numeric vs categorical columns explicitly
    numeric_candidates = [
        "age_at_diagnosis",
        "initial_pathologic_dx_year",
        "lymph_nodes_examined_count",
        "lymph_nodes_examined_he_count",
        "her2_ihc_score",
        "er_status_ihc_Percent_Positive",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Parse/Coerce numeric columns
    for col in numeric_cols:
        if col == "er_status_ihc_Percent_Positive":
            parsed = df[col].apply(_parse_percent_or_range_to_float)
            df[col] = pd.to_numeric(parsed, errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical strings: strip whitespace; normalize empties to NaN
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip()
        df[col] = df[col].replace({"": pd.NA, "na": pd.NA, "Na": pd.NA, "NA": pd.NA, "NaN": pd.NA, "nan": pd.NA})

    # Impute categorical
    if len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna(impute_categorical)
        df[categorical_cols] = df[categorical_cols].astype(object)

    # Impute numeric columns
    if impute_age_strategy not in {"median", "mean"}:
        raise ValueError("--impute-age-strategy must be 'median' or 'mean'")
    for col in numeric_cols:
        if impute_age_strategy == "median":
            fill_value = df[col].median()
        else:
            fill_value = df[col].mean()
        if pd.isna(fill_value):
            fill_value = 0.0
            print(f"[⚠️] All values missing for numeric column '{col}'; imputing 0.0 as fallback.")
        df[col] = df[col].fillna(fill_value)

    # Ensure no NaNs remain in modeling columns
    if df[categorical_cols + numeric_cols].isna().any().any():
        raise ValueError(
            "Found NaNs after imputation. Check inputs and imputation settings."
        )

    return PreprocessResult(
        df_for_model=df[categorical_cols + numeric_cols].copy(),
        df_original_index=original_index,
        id_series=id_series,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )


def compute_gower_distance(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> np.ndarray:
    """Compute the Gower distance matrix for mixed data.

    Returns an (n_samples x n_samples) dense float array.
    """
    X = df[categorical_cols + numeric_cols]
    D = gower.gower_matrix(X)
    if not isinstance(D, np.ndarray):
        D = np.asarray(D)
    if D.shape[0] != D.shape[1]:
        raise ValueError("Gower distance must be a square matrix.")
    if np.isnan(D).any() or np.isinf(D).any():
        raise ValueError(
            "Distance matrix contains non-finite values. "
            "Tip: check for non-finite inputs after preprocessing."
        )
    if np.allclose(D, 0):
        print(
            "[⚠️] All pairwise distances are zero. Data rows appear identical; "
            "silhouette will be invalid."
        )
    print("[✓] Computed Gower distance")
    return D


def _safe_agglomerative_fit(
    distance_matrix: np.ndarray,
    n_clusters: int,
    linkage: str,
) -> AgglomerativeClustering:
    """Fit AgglomerativeClustering with precomputed distances, handling API variants."""
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage, metric="precomputed"
        )
        model.fit(distance_matrix)
        return model
    except TypeError:
        # Older sklearn versions use 'affinity'
        model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage, affinity="precomputed"
        )
        model.fit(distance_matrix)
        return model


def evaluate_k_range(
    distance_matrix: np.ndarray,
    linkage: str,
    min_k: int,
    max_k: int,
    random_state: int,
) -> Tuple[pd.DataFrame, int, str]:
    """Evaluate clustering for k in [min_k..max_k] using silhouette on precomputed distances.

    Returns (silhouette_df, best_k, effective_linkage).
    """
    n_samples = distance_matrix.shape[0]

    if min_k < 2:
        print("[⚠️] min_k < 2; setting min_k=2")
        min_k = 2
    if max_k >= n_samples:
        print(
            f"[⚠️] max_k={max_k} >= n_samples={n_samples}; capping to {n_samples - 1}."
        )
        max_k = n_samples - 1
    if min_k > max_k:
        print(
            f"[⚠️] min_k={min_k} > max_k={max_k}; setting both to 2 (if possible)."
        )
        min_k = max_k = min(2, n_samples - 1)

    effective_linkage = linkage
    if linkage == "ward":
        print(
            "[⚠️] 'ward' linkage requires Euclidean distances; falling back to 'average'."
        )
        effective_linkage = "average"

    rows = []
    for k in range(min_k, max_k + 1):
        try:
            model = _safe_agglomerative_fit(distance_matrix, k, effective_linkage)
            labels = model.labels_
            # Silhouette with precomputed distances
            try:
                score = float(
                    silhouette_score(distance_matrix, labels, metric="precomputed")
                )
            except Exception:
                score = float("nan")
        except Exception as exc:
            print(f"[⚠️] Failed to fit for k={k}: {exc}")
            score = float("nan")
        rows.append({"k": k, "silhouette": score})

    silhouette_df = pd.DataFrame(rows)

    # Choose best k: max silhouette, break ties by smaller k
    valid = silhouette_df.dropna(subset=["silhouette"])
    if valid.empty:
        best_k = int(min_k)
        print("[⚠️] All silhouette scores invalid; defaulting to k=min_k.")
    else:
        max_score = valid["silhouette"].max()
        candidates = valid[valid["silhouette"] == max_score]
        best_k = int(candidates.sort_values("k").iloc[0]["k"])

    print("[✓] Evaluated silhouette over k-range")
    return silhouette_df, best_k, effective_linkage


def fit_final_agglomerative(
    distance_matrix: np.ndarray,
    n_clusters: int,
    linkage: str,
    random_state: int,
) -> np.ndarray:
    """Fit the final AgglomerativeClustering and return 1-based labels."""
    model = _safe_agglomerative_fit(distance_matrix, n_clusters, linkage)
    labels_zero_based = model.labels_
    labels = labels_zero_based.astype(int) + 1
    print("[✓] Fitted final agglomerative model")
    return labels


def compute_per_sample_metrics(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    knn_k: int,
) -> pd.DataFrame:
    """Compute per-sample metrics to help spot outliers.

    Metrics:
    - silhouette: silhouette value per point (using precomputed distances)
    - nn_dist: distance to nearest neighbor (excluding self)
    - knn_mean_dist: mean distance to k nearest neighbors (excluding self)
    """
    n = distance_matrix.shape[0]
    if knn_k < 1:
        knn_k = 1
    if knn_k >= n:
        knn_k = max(1, n - 1)

    # Silhouette per sample (handle possible exceptions)
    try:
        sil = silhouette_samples(distance_matrix, labels, metric="precomputed")
    except Exception:
        sil = np.full(n, np.nan, dtype=float)

    # Sort distances per row, exclude self (zero on diagonal)
    # For safety, replace diagonal with +inf to ensure exclusion
    D = distance_matrix.copy().astype(float)
    np.fill_diagonal(D, np.inf)
    # Nearest neighbor distance
    nn_dist = np.min(D, axis=1)
    # kNN mean distance
    # argsort to get smallest distances per row
    idx_sorted = np.argsort(D, axis=1)
    # gather first k indices per row
    topk_idx = idx_sorted[:, :knn_k]
    rows = np.arange(n)[:, None]
    knn_vals = D[rows, topk_idx]
    knn_mean = np.mean(knn_vals, axis=1)

    df = pd.DataFrame(
        {
            "silhouette": sil.astype(float),
            "nn_dist": nn_dist.astype(float),
            "knn_mean_dist": knn_mean.astype(float),
            "cluster": labels.astype(int),
        }
    )
    return df


def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> Dict[int, Dict[str, object]]:
    """Build per-cluster profiles with numeric stats and top categories.

    Returns a dict keyed by cluster label (1..k), with sub-dicts containing:
    - size
    - numeric_stats: {column: {mean, median, std, min, max}}
    - categorical_top: {column: [(category, count, pct), ... up to 3]}
    """
    df_prof = df.copy()
    df_prof["__cluster__"] = labels

    profiles: Dict[int, Dict[str, object]] = {}
    for cluster_id, df_cluster in df_prof.groupby("__cluster__"):
        profile: Dict[str, object] = {"size": int(len(df_cluster))}

        # Numeric stats
        numeric_stats: Dict[str, Dict[str, float]] = {}
        for col in numeric_cols:
            s = df_cluster[col]
            numeric_stats[col] = {
                "mean": float(np.mean(s)),
                "median": float(np.median(s)),
                "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
                "min": float(np.min(s)),
                "max": float(np.max(s)),
            }
        profile["numeric_stats"] = numeric_stats

        # Categorical top categories (top 3)
        categorical_top: Dict[str, List[Tuple[str, int, float]]]= {}
        for col in categorical_cols:
            counts = df_cluster[col].value_counts(dropna=False)
            total = counts.sum()
            top3 = counts.head(3)
            triples: List[Tuple[str, int, float]] = []
            for cat, cnt in top3.items():
                pct = float(100.0 * cnt / total) if total > 0 else 0.0
                triples.append((str(cat), int(cnt), pct))
            categorical_top[col] = triples
        profile["categorical_top"] = categorical_top

        profiles[int(cluster_id)] = profile

    print("[✓] Built cluster profiles")
    return profiles


def save_markdown_report(
    outpath: str,
    dataset_size: int,
    chosen_k: int,
    linkage: str,
    silhouette_df: pd.DataFrame,
    profiles: Dict[int, Dict[str, object]],
) -> None:
    """Save a compact Markdown report summarizing clustering and profiles."""
    lines: List[str] = []
    lines.append("# Cluster Report")
    lines.append("")
    lines.append(f"- **Dataset size**: {dataset_size}")
    lines.append(f"- **Chosen k**: {chosen_k}")
    lines.append(f"- **Linkage**: {linkage}")
    lines.append("")
    lines.append("## Silhouette over k")
    lines.append("")
    lines.append("| k | silhouette |")
    lines.append("|---:|---:|")
    for _, row in silhouette_df.sort_values("k").iterrows():
        k = int(row["k"])  # type: ignore[index]
        score = row["silhouette"]
        if pd.isna(score):
            score_str = "NaN"
        else:
            score_str = f"{float(score):.4f}"
        lines.append(f"| {k} | {score_str} |")
    lines.append("")
    lines.append("## Cluster profiles")
    lines.append("")

    for cid in sorted(profiles.keys()):
        p = profiles[cid]
        lines.append(f"### Cluster {cid}")
        lines.append(f"- **Size**: {p['size']}")
        lines.append("")

        # Numeric stats
        num_stats: Dict[str, Dict[str, float]] = p.get("numeric_stats", {})  # type: ignore[assignment]
        if num_stats:
            lines.append("#### Numeric (age_at_diagnosis)")
            lines.append("| stat | value |")
            lines.append("|:--|--:|")
            for col, stats in num_stats.items():
                lines.append(f"| mean | {stats['mean']:.2f} |")
                lines.append(f"| median | {stats['median']:.2f} |")
                lines.append(f"| std | {stats['std']:.2f} |")
                lines.append(f"| min | {stats['min']:.2f} |")
                lines.append(f"| max | {stats['max']:.2f} |")
            lines.append("")

        # Categorical top categories
        cat_top: Dict[str, List[Tuple[str, int, float]]] = p.get("categorical_top", {})  # type: ignore[assignment]
        if cat_top:
            lines.append("#### Top categories per feature (top 3)")
            for col, triples in cat_top.items():
                lines.append(f"- **{col}**:")
                if not triples:
                    lines.append("  - (no data)")
                else:
                    for val, cnt, pct in triples:
                        lines.append(f"  - {val}: {cnt} ({pct:.1f}%)")
            lines.append("")

    report_md = "\n".join(lines)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(report_md)
    print("[✓] Saved Markdown report")


def plot_and_save_dendrogram(distance_matrix: np.ndarray, outpath: str) -> None:
    """Plot a dendrogram using SciPy with 'average' linkage on condensed distances."""
    # Convert squareform to condensed
    try:
        condensed = squareform(distance_matrix, checks=False)
    except Exception:
        # As a fallback, force checks
        condensed = squareform(distance_matrix, checks=True)

    Z = scipy_linkage(condensed, method="average")
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram (average linkage)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print("[✓] Saved dendrogram plot")


def plot_mca_scatter(
    df_cats: pd.DataFrame,
    labels: np.ndarray,
    outpath: str,
    n_components: int,
    random_state: int,
    annotate_points: Optional[pd.DataFrame] = None,
    annotate_top_n: int = 0,
    annotate_label_col: Optional[str] = None,
) -> None:
    """Run MCA on categorical columns and save a 2D scatter colored by clusters."""
    try:
        import prince  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(
            "[✗] Missing dependency 'prince'. Install with: pip install prince",
            file=sys.stderr,
        )
        return

    if n_components < 1:
        n_components = 1

    mca = prince.MCA(n_components=n_components, random_state=random_state)
    mca_fit = mca.fit(df_cats)
    coords = mca_fit.transform(df_cats)

    # Ensure we have 2D for plotting
    if coords.shape[1] == 1:
        coords2d = np.column_stack([coords.iloc[:, 0].to_numpy(), np.zeros(len(coords))])
    else:
        coords2d = coords.iloc[:, :2].to_numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords2d[:, 0],
        coords2d[:, 1],
        c=labels,
        cmap="tab10",
        s=20,
        alpha=0.8,
        edgecolor="k",
        linewidths=0.2,
    )
    plt.title("MCA Scatter (colored by cluster)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Optional annotation of outliers
    if annotate_points is not None and annotate_top_n > 0:
        # Align indices to df_cats (original row order)
        ann = annotate_points.copy()
        # Choose a ranking metric: prefer 'knn_mean_dist', else 'nn_dist', else -silhouette
        metric = None
        for m in ["knn_mean_dist", "nn_dist", "silhouette"]:
            if m in ann.columns:
                metric = m
                break
        if metric is not None:
            if metric == "silhouette":
                ann = ann.sort_values(metric, ascending=True)
            else:
                ann = ann.sort_values(metric, ascending=False)
            ann = ann.head(int(annotate_top_n))
            # Build labels
            texts = []
            for idx in ann.index.tolist():
                x = coords2d[idx, 0]
                y = coords2d[idx, 1]
                if annotate_label_col is not None and annotate_label_col in ann.columns:
                    lbl = str(ann.loc[idx, annotate_label_col])
                else:
                    lbl = str(idx)
                plt.scatter([x], [y], c="none", edgecolor="red", s=80, linewidths=1.0)
                texts.append(plt.text(x, y, lbl, fontsize=8, color="red"))

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print("[✓] Saved MCA scatter plot")


# --------------------------------- Exclusions -------------------------------- #


def _load_exclusion_ids(spec: str, id_col: str) -> Set[str]:
    """Load an exclusion set from a comma-separated list or a file path.

    Accepted formats for files:
    - .txt: one ID per line
    - .csv/.tsv: use the first column; header allowed; also try a column named like id_col
    """
    spec = str(spec).strip()
    if spec == "":
        return set()
    # If the string looks like a path to an existing file, load from file
    if os.path.exists(spec) and os.path.isfile(spec):
        ext = os.path.splitext(spec)[1].lower()
        try:
            if ext in {".txt", ""}:
                with open(spec, "r", encoding="utf-8") as f:
                    values = [line.strip() for line in f if line.strip()]
                return {v for v in values}
            elif ext in {".csv", ".tsv"}:
                sep = "," if ext == ".csv" else "\t"
                df_ids = pd.read_csv(spec, sep=sep)
                if id_col in df_ids.columns:
                    series = df_ids[id_col]
                else:
                    # fallback to first column
                    series = df_ids.iloc[:, 0]
                return {str(v).strip() for v in series.dropna().astype(str).tolist() if str(v).strip()}
            else:
                # Try reading as csv with pandas as a generic approach
                df_ids = pd.read_csv(spec)
                if id_col in df_ids.columns:
                    series = df_ids[id_col]
                else:
                    series = df_ids.iloc[:, 0]
                return {str(v).strip() for v in series.dropna().astype(str).tolist() if str(v).strip()}
        except Exception as exc:
            raise ValueError(f"Failed to read --exclude-ids file '{spec}': {exc}")
    # Otherwise, treat as a comma-separated list
    return {v.strip() for v in spec.split(",") if v.strip()}


# ----------------------------------- CLI ----------------------------------- #


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster breast-cancer patients using Gower distance and hierarchical clustering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python cluster_patients_gower.py \
                --csv patients.csv \
                --id-col patient_id \
                --linkage average \
                --min-k 2 --max-k 8 \
                --plot-dendrogram \
                --mca-components 2
            """
        ),
    )

    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--id-col",
        default=None,
        help="Optional column name to carry through outputs (not used in clustering)",
    )
    parser.add_argument(
        "--linkage",
        choices=["average", "complete", "ward"],
        default="average",
        help="Linkage strategy (ward will auto-fallback to average for Gower)",
    )
    parser.add_argument("--min-k", type=int, default=2, help="Minimum number of clusters")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum number of clusters")
    parser.add_argument(
        "--impute-categorical",
        default="Unknown",
        help="Value to impute for missing categorical values",
    )
    parser.add_argument(
        "--impute-age-strategy",
        choices=["median", "mean"],
        default="median",
        help="Strategy to impute age",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: only use first N rows for quick runs",
    )
    parser.add_argument(
        "--outdir",
        default="./cluster_outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--plot-dendrogram",
        action="store_true",
        help="If set, save dendrogram plot (average linkage).",
    )
    parser.add_argument(
        "--mca-components",
        type=int,
        default=None,
        help="If provided, run MCA for visualization and save 2D scatter.",
    )
    parser.add_argument(
        "--drug-csv",
        default=None,
        help="Optional: path to clinical drug CSV to join by --id-col and add 'drug' feature",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="If set, save per-sample metrics (silhouette, kNN distances) to CSV.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="k for mean kNN distance in per-sample metrics (default: 5)",
    )
    parser.add_argument(
        "--annotate-outliers",
        type=int,
        default=0,
        help="If >0, annotate top-N outliers on MCA plot (based on metrics).",
    )
    parser.add_argument(
        "--annotate-label-col",
        default=None,
        help="Optional column to use as label when annotating outliers (requires --id-col)",
    )
    parser.add_argument(
        "--exclude-ids",
        default=None,
        help=(
            "Comma-separated IDs or path to a txt/csv/tsv file containing IDs to exclude."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    print("[ℹ️] Loading and preprocessing data...")
    try:
        prep = load_and_preprocess(
            csv_path=args.csv,
            id_col=args.id_col,
            impute_categorical=args.impute_categorical,
            impute_age_strategy=args.impute_age_strategy,
            sample=args.sample,
            drug_csv=args.drug_csv,
            exclude_ids=args.exclude_ids,
        )
    except Exception as exc:
        print(f"[✗] Preprocessing failed: {exc}", file=sys.stderr)
        return 2

    df_model = prep.df_for_model
    n_samples = df_model.shape[0]
    print(f"[✓] Preprocessed dataset with {n_samples} rows and {df_model.shape[1]} columns")
    # Log which columns are treated as categorical vs numeric for clustering
    try:
        print(f"[ℹ️] Categorical columns ({len(prep.categorical_cols)}): {prep.categorical_cols}")
        print(f"[ℹ️] Numeric columns ({len(prep.numeric_cols)}): {prep.numeric_cols}")
    except Exception:
        # Be robust to any unexpected issues while logging
        pass

    # Compute distance
    print("[ℹ️] Computing Gower distance...")
    try:
        D = compute_gower_distance(df_model, prep.categorical_cols, prep.numeric_cols)
    except Exception as exc:
        print(f"[✗] Gower distance computation failed: {exc}", file=sys.stderr)
        return 3

    # Evaluate k range
    print("[ℹ️] Evaluating clusters over k range...")
    try:
        silhouette_df, best_k, effective_linkage = evaluate_k_range(
            D, args.linkage, args.min_k, args.max_k, args.random_state
        )
    except Exception as exc:
        print(f"[✗] Evaluation over k failed: {exc}", file=sys.stderr)
        return 4

    # Save silhouette table
    sil_path = os.path.join(args.outdir, "silhouette_over_k.csv")
    silhouette_df.to_csv(sil_path, index=False)
    print(f"[✓] Saved silhouette table → {sil_path}")

    # Fit final model
    print(f"[ℹ️] Fitting final model with k={best_k}, linkage={effective_linkage}...")
    try:
        labels = fit_final_agglomerative(D, best_k, effective_linkage, args.random_state)
    except Exception as exc:
        print(f"[✗] Final fit failed: {exc}", file=sys.stderr)
        return 5

    # Save labels
    labels_df = pd.DataFrame({"cluster": labels})
    if prep.id_series is not None:
        labels_df.insert(0, args.id_col, prep.id_series.values)
    else:
        labels_df.insert(0, "row_index", prep.df_original_index.values)
    labels_path = os.path.join(args.outdir, "labels.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"[✓] Saved labels → {labels_path}")

    # Profile clusters and save report
    print("[ℹ️] Building cluster profiles and report...")
    try:
        profiles = profile_clusters(
            df_model, labels, prep.categorical_cols, prep.numeric_cols
        )
    except Exception as exc:
        print(f"[✗] Profiling failed: {exc}", file=sys.stderr)
        return 6

    report_path = os.path.join(args.outdir, "cluster_report.md")
    try:
        save_markdown_report(
            outpath=report_path,
            dataset_size=n_samples,
            chosen_k=int(best_k),
            linkage=effective_linkage,
            silhouette_df=silhouette_df,
            profiles=profiles,
        )
    except Exception as exc:
        print(f"[✗] Saving report failed: {exc}", file=sys.stderr)
        return 7

    # Optional dendrogram
    if args.plot_dendrogram:
        try:
            dendro_path = os.path.join(args.outdir, "dendrogram.png")
            plot_and_save_dendrogram(D, dendro_path)
        except Exception as exc:
            print(f"[⚠️] Dendrogram generation failed: {exc}")

    # Optional per-sample metrics
    metrics_df: Optional[pd.DataFrame] = None
    if args.save_metrics or (args.mca_components is not None and args.annotate_outliers > 0):
        print("[ℹ️] Computing per-sample metrics...")
        try:
            metrics_df = compute_per_sample_metrics(D, labels, knn_k=int(args.knn_k))
            if args.save_metrics:
                metrics_path = os.path.join(args.outdir, "per_sample_metrics.csv")
                # Attach an ID column if available
                if prep.id_series is not None:
                    metrics_df = metrics_df.copy()
                    metrics_df.insert(0, args.id_col, prep.id_series.values)
                else:
                    metrics_df = metrics_df.copy()
                    metrics_df.insert(0, "row_index", prep.df_original_index.values)
                metrics_df.to_csv(metrics_path, index=False)
                print(f"[✓] Saved per-sample metrics → {metrics_path}")
        except Exception as exc:
            print(f"[⚠️] Per-sample metrics failed: {exc}")

    # Optional MCA scatter
    if args.mca_components is not None:
        try:
            df_cats = df_model[prep.categorical_cols].copy()
            mca_path = os.path.join(args.outdir, "mca_scatter.png")
            annotate_df = None
            if metrics_df is not None and int(args.annotate_outliers) > 0:
                annotate_df = metrics_df
            plot_mca_scatter(
                df_cats=df_cats,
                labels=labels,
                outpath=mca_path,
                n_components=int(args.mca_components),
                random_state=int(args.random_state),
                annotate_points=annotate_df,
                annotate_top_n=int(args.annotate_outliers),
                annotate_label_col=args.annotate_label_col,
            )
        except Exception as exc:
            print(f"[⚠️] MCA plotting failed: {exc}")

    print("[✓] All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


