# Clinical Analysis Scripts

This directory contains modular scripts extracted from the Jupyter notebook `project_Clinical_Analysis.ipynb`. Each script focuses on a specific aspect of the clinical analysis pipeline.

## Scripts Overview

### 1. Vital Status Classifier (`classifier_models/vital_status_classifier.py`)

**Purpose**: Implements a Random Forest classifier to predict patient vital status (Dead/Alive) from clinical features with SHAP interpretability.

**Key Features**:
- Data preprocessing and drug name standardization
- One-hot encoding of categorical features
- Random Forest training with hyperparameter configuration
- ROC curve analysis
- SHAP-based feature importance analysis
- Model saving/loading capabilities

**Usage**:
```python
from classifier_models.vital_status_classifier import VitalStatusClassifier

# Initialize classifier
classifier = VitalStatusClassifier(n_estimators=400, random_state=42)

# Load and preprocess data
df = classifier.load_and_preprocess_data(clinical_file, drug_file)

# Prepare features and train
X, y = classifier.prepare_features(df)
results = classifier.train(X, y, test_size=0.3)

# Analyze feature importance
top_features = classifier.analyze_feature_importance_shap(X, top_n=20)

# Save trained model
classifier.save_model("output/vital_status_classifier")
```

### 2. Patient Clustering (`clustering/patient_clustering.py`)

**Purpose**: Performs KMeans clustering to identify patient subgroups based on clinical characteristics with PCA visualization.

**Key Features**:
- Patient clustering using clinical features
- Optimal cluster number determination via silhouette analysis
- PCA-based 2D visualization
- Comprehensive cluster characterization and profiling
- Statistical analysis of cluster differences

**Usage**:
```python
from clustering.patient_clustering import PatientClustering

# Initialize clustering
clustering = PatientClustering(n_clusters=3, random_state=42)

# Load and prepare data
df = clustering.load_and_preprocess_data(clinical_file, drug_file)
cluster_df, X_encoded = clustering.prepare_clustering_data(df)

# Find optimal clusters
silhouette_scores, optimal_k = clustering.find_optimal_clusters(X_encoded)

# Perform clustering
cluster_labels = clustering.perform_clustering(X_encoded, n_clusters=optimal_k)

# Visualize and analyze
X_pca = clustering.visualize_clusters_pca(X_encoded, cluster_labels)
analysis_df = clustering.analyze_cluster_characteristics(cluster_df, cluster_labels)
```

### 3. Patient Profiling (`patient_profiling.py`)

**Purpose**: Creates detailed patient profiles based on different clinical criteria (drug treatments, HER2 status, vital status).

**Key Features**:
- Drug-based patient profiling
- HER2 status-based profiling
- Vital status-based profiling
- Comprehensive statistical summaries
- Automated visualization generation
- Profile comparison capabilities

**Usage**:
```python
from patient_profiling import PatientProfiler

# Initialize profiler
profiler = PatientProfiler()

# Load data
df = profiler.load_and_preprocess_data(clinical_file, drug_file)

# Create profiles
drug_profile = profiler.profile_by_drug("Tamoxifen", "output/profiles")
her2_profile = profiler.profile_by_her2_status("Negative", "output/profiles")
vital_profile = profiler.profile_by_vital_status("Dead", "output/profiles")

# Compare profiles
profiler.compare_profiles("vital", ["Dead", "Alive"], "output/comparisons")
```

## Data Requirements

All scripts expect two main data files:
- **Clinical data**: `nationwidechildrens.org_clinical_patient_brca.csv`
- **Drug data**: `nationwidechildrens.org_clinical_drug_brca.csv`

## Dependencies

Ensure you have the following packages installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

## Output Structure

Each script generates organized outputs:

```
output/
├── vital_status_classifier/
│   ├── vital_status_classifier.pkl
│   ├── feature_encoder.pkl
│   └── plots/
├── clustering_analysis/
│   ├── clustered_patients.csv
│   ├── cluster_summary.csv
│   └── plots/
└── patient_profiles/
    ├── drug_profile_[drug_name]/
    ├── her2_profile_[status]/
    ├── vital_status_profile_[status]/
    └── comparisons/
```

## Key Clinical Features Used

### Classifier Features
- All clinical features except excluded identifiers and target variable
- One-hot encoded categorical variables
- Numeric features (age, counts, etc.)

### Clustering Features
- `tumor_status`, `ajcc_staging_edition`, `tissue_source_site`
- `initial_pathologic_dx_year`, `birth_days_to`, `margin_status`
- `ajcc_nodes_pathologic_pn`, `age_at_diagnosis`, `her2_status_by_ihc`
- `method_initial_path_dx`, `lymph_nodes_examined_count`, `her2_ihc_score`
- `metastatic_tumor_indicator`, `axillary_staging_method`
- `anatomic_neoplasm_subdivision`, `her2_fish_status`
- `lymph_nodes_examined_he_count`, `ajcc_pathologic_tumor_stage`
- `er_status_ihc_Percent_Positive`, `menopause_status`, `vital_status`

### Profiling Parameters
- `tumor_status`, `age_at_diagnosis`, `ajcc_pathologic_tumor_stage`
- `er_status_ihc_Percent_Positive`, `her2_status_by_ihc`, `menopause_status`
- `vital_status`, `gender`, `race`, `drug_1_clean`

## Drug Name Standardization

All scripts include comprehensive drug name cleaning that standardizes:
- **Doxorubicin**: ADRIAMYCIN, Adriamycin, etc.
- **Cyclophosphamide**: CYTOXAN, Cytoxan, etc.
- **Paclitaxel**: TAXOL, Taxol, etc.
- **Tamoxifen**: TAMOXIFEN, Nolvadex, etc.
- **Trastuzumab**: HERCEPTIN, Herceptin, etc.
- And many more drug variants

## Customization

### Modifying Features
To change the features used in clustering:
```python
clustering.cluster_columns = ["your", "custom", "features"]
```

### Adjusting Model Parameters
```python
# Classifier
classifier = VitalStatusClassifier(n_estimators=500, random_state=123)

# Clustering
clustering = PatientClustering(n_clusters=4, random_state=123)
```

### Adding New Profiling Parameters
```python
profiler.wanted_parameters.extend(["new_parameter1", "new_parameter2"])
if "new_parameter1" in numeric_cols:
    profiler.numeric_parameters.append("new_parameter1")
```

## Error Handling

All scripts include comprehensive error handling for:
- Missing data files
- Empty filtered datasets
- Invalid parameter values
- Missing required columns

## Performance Notes

- **Classifier**: Training time scales with dataset size and number of features
- **Clustering**: PCA reduces computational complexity for visualization
- **Profiling**: Plot generation can be time-intensive for large datasets

## Integration with Original Notebook

These scripts maintain compatibility with the original Jupyter notebook workflow while providing:
- Better code organization
- Reusability across projects
- Enhanced error handling
- Comprehensive documentation
- Modular design for easy customization

Each script can be run independently or integrated into larger analysis pipelines.
