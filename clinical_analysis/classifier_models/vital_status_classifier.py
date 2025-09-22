#!/usr/bin/env python3
"""
Vital Status Classifier using Random Forest

This script implements a Random Forest classifier to predict vital status (Dead/Alive)
from clinical features, with SHAP analysis for feature importance interpretation.

Author: Clinical Analysis Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import shap
import os


class VitalStatusClassifier:
    """
    Random Forest classifier for predicting vital status with SHAP interpretability.
    """
    
    def __init__(self, n_estimators=400, random_state=42):
        """
        Initialize the classifier.
        
        Args:
            n_estimators (int): Number of trees in the random forest
            random_state (int): Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classifier = None
        self.encoder = None
        self.feature_columns = None
        self.excluded_columns = [
            "bcr_patient_uuid", "last_contact_days_to", "vital_status", 
            "bcr_patient_barcode", "form_completion_date", "prospective_collection",
            "retrospective_collection", "death_days_to", "informed_consent_verified",
            "patient_id", "project_code", "nte_er_positivity_other_scale",
            "nte_er_positivity_define_method", "nte_pr_positivity_other_scale",
            "nte_pr_positivity_define_method", "nte_her2_positivity_other_scale",
            "nte_her2_positivity_method", "nte_her2_signal_number",
            "nte_cent_17_signal_number", "her2_cent17_counted_cells_count",
            "nte_cent17_her2_other_scale", "nte_her2_fish_define_method",
            "clinical_T", "days_to_patient_progression_free", "days_to_tumor_progression",
            "stage_other", "nte_cent_17_her2_ratio"
        ]
        
    def load_and_preprocess_data(self, clinical_file_path, drug_file_path):
        """
        Load and preprocess clinical and drug data.
        
        Args:
            clinical_file_path (str): Path to clinical data CSV
            drug_file_path (str): Path to drug data CSV
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Load clinical data
        df = pd.read_csv(clinical_file_path)
        df = df.drop([0, 1]).reset_index(drop=True)
        
        # Convert age to numeric
        df["age_at_diagnosis"] = pd.to_numeric(df["age_at_diagnosis"], errors="coerce")
        
        # Load and process drug data
        df2 = pd.read_csv(drug_file_path)
        df2_sorted = df2.sort_values(["bcr_patient_uuid", "pharmaceutical_therapy_drug_name"])
        df2_sorted["drug_order"] = df2_sorted.groupby("bcr_patient_uuid").cumcount() + 1
        
        # Pivot drug data
        df2_wide = df2_sorted.pivot(index="bcr_patient_uuid", columns="drug_order", 
                                   values="pharmaceutical_therapy_drug_name")
        df2_wide = df2_wide.rename(columns={1: "drug_1", 2: "drug_2"})
        
        # Merge with clinical data
        df = df.merge(df2_wide[["drug_1", "drug_2"]], 
                     left_on="bcr_patient_uuid", right_index=True, how="left")
        
        # Clean drug names
        df = self._clean_drug_names(df)
        
        print(f"Total rows in dataset: {len(df)}")
        print(f"Total columns in dataset: {len(df.columns)}")
        
        return df
    
    def _clean_drug_names(self, df):
        """Clean and standardize drug names."""
        drug_merge_dict = {
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
            "DOXORUBICIN": "Doxorubicin",
            "Doxorubicin": "Doxorubicin",
            "Doxorubicin Liposome": "Doxorubicin",
            "Doxorubicinum": "Doxorubicin",
            
            # Cyclophosphamide/Cytoxan
            "CYCLOPHOSPHAMIDE": "Cyclophosphamide",
            "Cyclophosphamide": "Cyclophosphamide",
            "Cyclophasphamide": "Cyclophosphamide",
            "Cyclophospamide": "Cyclophosphamide",
            "Cyclophosphane": "Cyclophosphamide",
            "cyclophosphamid": "Cyclophosphamide",
            "cyclophosphamide": "Cyclophosphamide",
            "CYTOXAN": "Cyclophosphamide",
            "Cytoxan": "Cyclophosphamide",
            
            # Paclitaxel/Taxol
            "PACLITAXEL": "Paclitaxel",
            "Paclitaxel": "Paclitaxel",
            "Albumin-Bound Paclitaxel": "Paclitaxel",
            "Paclitaxel (Protein-Bound)": "Paclitaxel",
            "paclitaxel": "Paclitaxel",
            "TAXOL": "Paclitaxel",
            "Taxol": "Paclitaxel",
            "taxol": "Paclitaxel",
            
            # Docetaxel/Taxotere
            "DOCETAXEL": "Docetaxel",
            "Docetaxel": "Docetaxel",
            "Doxetaxel": "Docetaxel",
            "TAXOTERE": "Docetaxel",
            "Taxotere": "Docetaxel",
            "taxotere": "Docetaxel",
            
            # Tamoxifen
            "TAMOXIFEN": "Tamoxifen",
            "Tamoxifen": "Tamoxifen",
            "tamoxifen": "Tamoxifen",
            "tamoxifen citrate": "Tamoxifen",
            "Nolvadex": "Tamoxifen",
            "nolvadex": "Tamoxifen",
            
            # Letrozole/Femara
            "LETROZOLE": "Letrozole",
            "Letrozole": "Letrozole",
            "letrozole": "Letrozole",
            "Femara": "Letrozole",
            "FEMARA": "Letrozole",
            "letrozolum": "Letrozole",
            
            # Exemestane/Aromasin
            "EXEMESTANE": "Exemestane",
            "Exemestane": "Exemestane",
            "Aromasin": "Exemestane",
            "aromasin": "Exemestane",
            
            # Anastrozole/Arimidex
            "ANASTROZOLE": "Anastrozole",
            "Anastrozole": "Anastrozole",
            "Anastrazole": "Anastrozole",
            "ARIMIDEX": "Anastrozole",
            "Arimidex": "Anastrozole",
            "anastrozolum": "Anastrozole",
            "arimidex": "Anastrozole",
            
            # Bevacizumab/Avastin
            "BEVACIZUMAB": "Bevacizumab",
            "Bevacizumab": "Bevacizumab",
            "AVASTIN": "Bevacizumab",
            "Avastin": "Bevacizumab",
            "avastin": "Bevacizumab",
            
            # Capecitabine/Xeloda
            "CAPECITABINE": "Capecitabine",
            "Capecetabine": "Capecitabine",
            "XELODA": "Capecitabine",
            "Xeloda": "Capecitabine",
            
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
            
            # Zoledronic Acid/Zometa
            "ZOLEDRONIC ACID": "Zoledronic Acid",
            "Zoledronic Acid": "Zoledronic Acid",
            "Zoledronic acid": "Zoledronic Acid",
            "zoledronic acid": "Zoledronic Acid",
            "Zometa": "Zoledronic Acid",
            "Xgeva": "Zoledronic Acid",
            
            # Fulvestrant/Faslodex
            "FULVESTRANT": "Fulvestrant",
            "Fulvestrant": "Fulvestrant",
            "Faslodex": "Fulvestrant",
            "faslodex": "Fulvestrant",
            
            # Clodronate
            "Clodronate": "Clodronate",
            "clodronate": "Clodronate",
            "Clodronic acid": "Clodronate",
            "clodronic acid": "Clodronate",
        }
        
        # Clean drug columns
        df["drug_1_clean"] = df["drug_1"].replace(drug_merge_dict)
        df["drug_2_clean"] = df["drug_2"].replace(drug_merge_dict)
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (X_encoded, y, feature_names)
        """
        # Select features (exclude target and unwanted columns)
        X = df.drop(columns=self.excluded_columns)
        y = df["vital_status"]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # One-hot encode categorical features
        self.encoder = OneHotEncoder(sparse_output=False)
        X_encoded = self.encoder.fit_transform(X)
        
        print(f"Number of features after encoding: {X_encoded.shape[1]}")
        
        return X_encoded, y
    
    def train(self, X, y, test_size=0.3):
        """
        Train the Random Forest classifier.
        
        Args:
            X (array-like): Features
            y (array-like): Target variable
            test_size (float): Test set size
            
        Returns:
            dict: Training results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        self.classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store test data for SHAP analysis
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_proba = y_proba
        
        print(f"Accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
    
    def plot_roc_curve(self, y_test=None, y_proba=None):
        """Plot ROC curve."""
        if y_test is None:
            y_test = self.y_test
        if y_proba is None:
            y_proba = self.y_proba
            
        fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label='Dead')
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        
        return roc_auc
    
    def analyze_feature_importance_shap(self, X, top_n=20):
        """
        Analyze feature importance using SHAP values.
        
        Args:
            X (array-like): Features for SHAP analysis
            top_n (int): Number of top features to display
            
        Returns:
            pd.Series: Top feature importances
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first")
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.classifier)
        shap_values = explainer.shap_values(X)
        
        # Handle SHAP values shape (3D array or list of arrays)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_mean = np.abs(shap_values).mean(axis=2)
        elif isinstance(shap_values, list):
            shap_mean = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            shap_mean = np.abs(shap_values)
        
        # Average across samples
        shap_importance = shap_mean.mean(axis=0)
        
        # Map OneHot columns to original features
        encoded_features = self.encoder.get_feature_names_out()
        feature_map = {col: col.rsplit("_", 1)[0] for col in encoded_features}
        
        # Build DataFrame and group by original features
        shap_imp_df = pd.DataFrame({
            "encoded_feature": encoded_features,
            "importance": shap_importance
        })
        shap_imp_grouped = shap_imp_df.groupby(
            shap_imp_df["encoded_feature"].map(feature_map)
        )["importance"].sum().sort_values(ascending=False)
        
        # Plot top features
        top_features = shap_imp_grouped.head(top_n)
        plt.figure(figsize=(10, 6))
        top_features[::-1].plot(kind="barh")
        plt.xlabel("Mean(|SHAP value|)", fontsize=12)
        plt.ylabel("Original features", fontsize=12)
        plt.title(f"Top {top_n} most important features (SHAP)", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    def save_model(self, output_dir):
        """Save the trained model and encoder."""
        import joblib
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.classifier is not None:
            joblib.dump(self.classifier, os.path.join(output_dir, "vital_status_classifier.pkl"))
        if self.encoder is not None:
            joblib.dump(self.encoder, os.path.join(output_dir, "feature_encoder.pkl"))
            
        print(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir):
        """Load a trained model and encoder."""
        import joblib
        
        self.classifier = joblib.load(os.path.join(model_dir, "vital_status_classifier.pkl"))
        self.encoder = joblib.load(os.path.join(model_dir, "feature_encoder.pkl"))
        
        print(f"Model loaded from {model_dir}")


def main():
    """Main function to run the vital status classifier."""
    
    # Configuration - Update these paths as needed
    clinical_file = "path/to/nationwidechildrens.org_clinical_patient_brca.csv"
    drug_file = "path/to/nationwidechildrens.org_clinical_drug_brca.csv"
    output_dir = "output/vital_status_classifier"
    
    # Initialize classifier
    classifier = VitalStatusClassifier(n_estimators=400, random_state=42)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = classifier.load_and_preprocess_data(clinical_file, drug_file)
    
    # Prepare features
    print("Preparing features...")
    X, y = classifier.prepare_features(df)
    
    # Train model
    print("Training model...")
    results = classifier.train(X, y, test_size=0.3)
    
    # Plot ROC curve
    print("Plotting ROC curve...")
    roc_auc = classifier.plot_roc_curve()
    
    # Analyze feature importance with SHAP
    print("Analyzing feature importance with SHAP...")
    top_features = classifier.analyze_feature_importance_shap(X, top_n=20)
    
    print("\nTop 20 most important features:")
    print(top_features)
    
    # Save model
    print("Saving model...")
    classifier.save_model(output_dir)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
