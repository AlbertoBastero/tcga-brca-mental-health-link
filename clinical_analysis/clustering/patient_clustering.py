#!/usr/bin/env python3
"""
Patient Clustering Analysis using KMeans

This script implements KMeans clustering to identify patient groups based on clinical features,
with PCA visualization and silhouette analysis for optimal cluster determination.

Author: Clinical Analysis Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os


class PatientClustering:
    """
    KMeans clustering analysis for patient segmentation with visualization and interpretation.
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        """
        Initialize the clustering analysis.
        
        Args:
            n_clusters (int): Number of clusters for KMeans
            random_state (int): Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.encoder = None
        self.pca = None
        self.cluster_columns = [
            "tumor_status",
            "ajcc_staging_edition",
            "tissue_source_site",
            "initial_pathologic_dx_year",
            "birth_days_to",
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
            "vital_status"
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
        
        # Clean drug names (simplified version)
        df = self._clean_drug_names(df)
        
        print(f"Total rows in dataset: {len(df)}")
        print(f"Total columns in dataset: {len(df.columns)}")
        
        return df
    
    def _clean_drug_names(self, df):
        """Clean and standardize drug names (simplified version)."""
        drug_merge_dict = {
            # Key drug name mappings (simplified from the original)
            "ADRIAMYCIN": "Doxorubicin", "Adriamycin": "Doxorubicin", "adriamycin": "Doxorubicin",
            "DOXORUBICIN": "Doxorubicin", "Doxorubicin": "Doxorubicin",
            "CYCLOPHOSPHAMIDE": "Cyclophosphamide", "Cyclophosphamide": "Cyclophosphamide",
            "CYTOXAN": "Cyclophosphamide", "Cytoxan": "Cyclophosphamide", "cytoxan": "Cyclophosphamide",
            "PACLITAXEL": "Paclitaxel", "Paclitaxel": "Paclitaxel", "paclitaxel": "Paclitaxel",
            "TAXOL": "Paclitaxel", "Taxol": "Paclitaxel", "taxol": "Paclitaxel",
            "TAMOXIFEN": "Tamoxifen", "Tamoxifen": "Tamoxifen", "tamoxifen": "Tamoxifen",
            "LETROZOLE": "Letrozole", "Letrozole": "Letrozole", "letrozole": "Letrozole",
            "ANASTROZOLE": "Anastrozole", "Anastrozole": "Anastrozole", "anastrozole": "Anastrozole",
            "HERCEPTIN": "Trastuzumab", "Herceptin": "Trastuzumab", "herceptin": "Trastuzumab",
            "TRASTUZUMAB": "Trastuzumab", "Trastuzumab": "Trastuzumab"
        }
        
        df["drug_1_clean"] = df["drug_1"].replace(drug_merge_dict)
        df["drug_2_clean"] = df["drug_2"].replace(drug_merge_dict)
        
        return df
    
    def prepare_clustering_data(self, df):
        """
        Prepare data for clustering by selecting relevant columns and handling missing values.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (cluster_df, X_encoded)
        """
        # Select clustering columns and drop rows with missing values
        cluster_df = df[self.cluster_columns].dropna()
        
        print(f"Rows after removing missing values: {len(cluster_df)}")
        print(f"Columns used for clustering: {len(self.cluster_columns)}")
        
        # One-hot encode categorical features
        self.encoder = OneHotEncoder(sparse_output=False)
        X_encoded = self.encoder.fit_transform(cluster_df[self.cluster_columns])
        
        print(f"Features after encoding: {X_encoded.shape[1]}")
        
        return cluster_df, X_encoded
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using silhouette analysis.
        
        Args:
            X (array-like): Encoded features
            max_clusters (int): Maximum number of clusters to test
            
        Returns:
            tuple: (silhouette_scores, optimal_k)
        """
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For k={k}, silhouette score = {silhouette_avg:.3f}")
        
        # Find optimal k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # Plot silhouette analysis
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, silhouette_scores, 'o-')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis for Optimal k")
        plt.axvline(x=optimal_k, color='red', linestyle='--', 
                   label=f'Optimal k = {optimal_k}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Optimal number of clusters: {optimal_k}")
        
        return silhouette_scores, optimal_k
    
    def perform_clustering(self, X, n_clusters=None):
        """
        Perform KMeans clustering.
        
        Args:
            X (array-like): Encoded features
            n_clusters (int): Number of clusters (if None, uses self.n_clusters)
            
        Returns:
            array: Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
            
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        cluster_labels = self.kmeans.fit_predict(X)
        
        print(f"Clustering completed with {n_clusters} clusters")
        print(f"Cluster distribution: {np.bincount(cluster_labels)}")
        
        return cluster_labels
    
    def visualize_clusters_pca(self, X, cluster_labels, title="KMeans Clusters Visualization"):
        """
        Visualize clusters using PCA dimensionality reduction.
        
        Args:
            X (array-like): Encoded features
            cluster_labels (array): Cluster assignments
            title (str): Plot title
        """
        # Reduce to 2D using PCA
        self.pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                            cmap="tab10", alpha=0.7, s=50)
        
        # Add cluster centers
        if self.kmeans is not None:
            centers_pca = self.pca.transform(self.kmeans.cluster_centers_)
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='x', s=300, linewidths=3, label='Centroids')
        
        plt.xlabel(f"PCA Component 1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PCA Component 2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title(title)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster')
        
        if self.kmeans is not None:
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        return X_pca
    
    def analyze_cluster_characteristics(self, cluster_df, cluster_labels, output_dir=None):
        """
        Analyze and visualize cluster characteristics.
        
        Args:
            cluster_df (pd.DataFrame): DataFrame with cluster data
            cluster_labels (array): Cluster assignments
            output_dir (str): Directory to save plots (optional)
        """
        # Add cluster labels to dataframe
        analysis_df = cluster_df.copy()
        analysis_df['cluster'] = cluster_labels
        
        clusters = sorted(analysis_df['cluster'].unique())
        total_n = len(analysis_df)
        
        print("\n" + "="*60)
        print("CLUSTER CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        # 1. Cluster size analysis
        self._plot_cluster_sizes(analysis_df, total_n, output_dir)
        
        # 2. Detailed cluster profiles
        self._analyze_cluster_profiles(analysis_df, clusters, output_dir)
        
        # 3. Age distribution by cluster
        self._plot_age_distribution(analysis_df, output_dir)
        
        # 4. Categorical variable distributions
        self._plot_categorical_distributions(analysis_df, clusters, output_dir)
        
        return analysis_df
    
    def _plot_cluster_sizes(self, analysis_df, total_n, output_dir):
        """Plot cluster sizes."""
        counts = analysis_df['cluster'].value_counts().sort_index()
        
        plt.figure(figsize=(8, 5))
        ax = counts.plot(kind="bar", color=sns.color_palette("Set2", n_colors=len(counts)))
        ax.set_ylabel("Count")
        ax.set_xlabel("Cluster")
        ax.set_title("Cluster Sizes")
        
        # Add percentage annotations
        for p in ax.patches:
            h = p.get_height()
            pct = h / total_n * 100
            ax.annotate(f"{int(h)}\n({pct:.1f}%)", 
                       (p.get_x() + p.get_width()/2, h),
                       ha="center", va="bottom", fontsize=10)
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "cluster_sizes.png"), dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def _analyze_cluster_profiles(self, analysis_df, clusters, output_dir):
        """Analyze detailed cluster profiles."""
        for cluster_id in clusters:
            print(f"\n{'='*20} CLUSTER {cluster_id} {'='*20}")
            sub = analysis_df[analysis_df['cluster'] == cluster_id]
            print(f"Size: {len(sub)} patients")
            
            for col in self.cluster_columns:
                if col in sub.columns:
                    if sub[col].dtype in ['object', 'category']:
                        print(f"\n{col}:")
                        value_counts = sub[col].value_counts().head(5)
                        for val, count in value_counts.items():
                            pct = count / len(sub) * 100
                            print(f"  {val}: {count} ({pct:.1f}%)")
                    else:
                        print(f"\n{col}: mean = {sub[col].mean():.2f}, std = {sub[col].std():.2f}")
    
    def _plot_age_distribution(self, analysis_df, output_dir):
        """Plot age distribution by cluster."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=analysis_df, x="cluster", y="age_at_diagnosis", 
                   hue="cluster", palette="Set2", legend=False)
        plt.title("Age at Diagnosis by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Age at Diagnosis")
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "age_distribution_by_cluster.png"), 
                       dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def _plot_categorical_distributions(self, analysis_df, clusters, output_dir):
        """Plot distributions of categorical variables by cluster."""
        categorical_cols = [col for col in self.cluster_columns 
                           if analysis_df[col].dtype in ['object', 'category']]
        
        for col in categorical_cols:
            if col not in analysis_df.columns:
                continue
                
            # Create proportion table
            prop_table = pd.crosstab(analysis_df['cluster'], analysis_df[col], 
                                   normalize='index').loc[clusters]
            
            # Plot stacked bar chart
            plt.figure(figsize=(12, 6))
            ax = prop_table.plot(kind="bar", stacked=True, colormap="tab20", 
                               figsize=(12, 6))
            ax.set_ylabel("Proportion within cluster")
            ax.set_xlabel("Cluster")
            ax.set_title(f"Distribution of {col} by Cluster")
            ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.xticks(rotation=0)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"), 
                           dpi=300, bbox_inches="tight")
            
            plt.show()
    
    def save_results(self, analysis_df, output_dir):
        """Save clustering results to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save clustered data
        analysis_df.to_csv(os.path.join(output_dir, "clustered_patients.csv"), index=False)
        
        # Save cluster summary
        summary = analysis_df.groupby('cluster').agg({
            'age_at_diagnosis': ['count', 'mean', 'std'],
            'vital_status': lambda x: (x == 'Dead').sum() / len(x) * 100
        }).round(2)
        
        summary.columns = ['Count', 'Mean_Age', 'Std_Age', 'Mortality_Rate_%']
        summary.to_csv(os.path.join(output_dir, "cluster_summary.csv"))
        
        print(f"Results saved to {output_dir}")
        
        return summary


def main():
    """Main function to run the clustering analysis."""
    
    # Configuration - Update these paths as needed
    clinical_file = "path/to/nationwidechildrens.org_clinical_patient_brca.csv"
    drug_file = "path/to/nationwidechildrens.org_clinical_drug_brca.csv"
    output_dir = "output/clustering_analysis"
    
    # Initialize clustering analysis
    clustering = PatientClustering(n_clusters=3, random_state=42)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = clustering.load_and_preprocess_data(clinical_file, drug_file)
    
    # Prepare clustering data
    print("Preparing clustering data...")
    cluster_df, X_encoded = clustering.prepare_clustering_data(df)
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    silhouette_scores, optimal_k = clustering.find_optimal_clusters(X_encoded, max_clusters=10)
    
    # Perform clustering with optimal k
    print(f"Performing clustering with k={optimal_k}...")
    cluster_labels = clustering.perform_clustering(X_encoded, n_clusters=optimal_k)
    
    # Visualize clusters
    print("Visualizing clusters...")
    X_pca = clustering.visualize_clusters_pca(X_encoded, cluster_labels)
    
    # Analyze cluster characteristics
    print("Analyzing cluster characteristics...")
    analysis_df = clustering.analyze_cluster_characteristics(cluster_df, cluster_labels, output_dir)
    
    # Save results
    print("Saving results...")
    summary = clustering.save_results(analysis_df, output_dir)
    
    print("\nCluster Summary:")
    print(summary)
    
    print("Clustering analysis complete!")


if __name__ == "__main__":
    main()
