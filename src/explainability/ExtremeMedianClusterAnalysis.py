import os
from pathlib import Path
import pickle
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from matplotlib.sankey import Sankey
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import yaml

from utils.geoutils import plot_clusters_on_world_map

class SHAPDataLoader:
    """
    Loads and preprocesses SHAP data from a pickle file.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initializes the SHAPDataLoader instance.

        Args:
            file_path (str): Path to the pickle file containing SHAP data.
        """
        self.file_path = file_path
        self.aggregated = None
        self.feature_names: List[str] = []
        self.basin_ids: List[str] = []
        self.X: Optional[np.ndarray] = None

    def load(self) -> None:
        """
        Loads the SHAP data from the pickle file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(self.file_path):
            logging.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        with open(self.file_path, "rb") as f:
            data = pickle.load(f)
        self.aggregated = data.get("aggregated", {})
        self.feature_names = data.get("feature_names", [])
        self.basin_ids = list(self.aggregated.keys())
        self.X = np.array([self.aggregated[bid]["signed"] for bid in self.basin_ids])

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Retrieves the loaded SHAP data.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
                - X: Array of signed SHAP values.
                - feature_names: List of feature names.
                - basin_ids: List of basin identifiers.
        """
        return self.X, self.feature_names, self.basin_ids

class ClusterAnalyzer:
    """
    Performs clustering analysis on provided data.
    Handles normalization, PCA-based dimensionality reduction, and clustering using various methods.
    Available clustering methods:
      - "KMeans" (default)
      - "GMM" (Gaussian Mixture Models)
      - "Agglomerative" (Hierarchical clustering)

    The grid search function is supported for methods with a defined number of clusters.
    """

    def __init__(self, 
                 X: np.ndarray, 
                 n_clusters: int, 
                 n_dim: int, 
                 clustering_method: str,
                 apply_pca: bool = True) -> None:
        """
        Initializes the ClusterAnalyzer.

        Args:
            X (np.ndarray): Data array to be analyzed.
            n_clusters (int): Default number of clusters (if applicable).
            n_dim (int): Number of PCA dimensions to reduce to.
            clustering_method (str, optional): Clustering algorithm to use.
        """
        self.X = X
        self.n_clusters = n_clusters
        self.n_dim = n_dim
        self.clustering_method = clustering_method.lower()
        self.apply_pca = apply_pca

        self.scaler = StandardScaler()
        self.X_scaled: Optional[np.ndarray] = None
        self.X_reduced: Optional[np.ndarray] = None
        self.clusters: Optional[np.ndarray] = None

    def normalize(self) -> None:
        """Normalizes the input data using StandardScaler."""
        self.X_scaled = self.scaler.fit_transform(self.X)

    def reduce_dimension(self, n_components: Optional[int] = None) -> np.ndarray:
        """
        Transforms the normalized data by applying PCA if `apply_pca` is True.
        If `apply_pca` is False, returns the scaled data without applying any dimensionality reduction.
        
        Args:
            n_components (Optional[int], optional): Number of PCA components to use.
                If None, uses the default self.n_dim. Defaults to None.

        Returns:
            np.ndarray: PCA-reduced data.
        """
        if self.X_scaled is None:
            raise ValueError("Data must be normalized before dimension reduction.")
        n_components = n_components or self.n_dim
        if self.apply_pca:
            pca = PCA(n_components=n_components)
            self.X_reduced = pca.fit_transform(self.X_scaled)
        else:
            self.X_reduced = self.X_scaled
        return self.X_reduced

    def perform_clustering(self) -> None:
        """
        Performs clustering on the PCA-reduced data using the specified clustering method.
        If the data has not been reduced, it calls reduce_dimension() first.
        """
        if self.X_reduced is None:
            self.reduce_dimension()

        method = self.clustering_method.lower()
        if method == "kmeans":
            model = KMeans(
                n_clusters=self.n_clusters,
                n_init=100,
                max_iter=1000,
                random_state=42,
            )
            self.clusters = model.fit_predict(self.X_reduced)
        elif method == "gmm":
            model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=42,
            )
            model.fit(self.X_reduced)
            self.clusters = model.predict(self.X_reduced)
        elif method in ["agglomerative", "hierarchical"]:
            model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
            )
            self.clusters = model.fit_predict(self.X_reduced)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

    def grid_search(self, components_range: np.ndarray, clusters_range: np.ndarray) -> pd.DataFrame:
        """
        Performs grid search over PCA components and number of clusters (if applicable).

        Args:
            components_range (np.ndarray): Array of PCA component numbers to test.
            clusters_range (np.ndarray): Array of cluster numbers to test.

        Returns:
            pd.DataFrame: DataFrame with silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
        """
        results = []
        for n_components in tqdm(components_range, desc="Grid Search PCA Components..."):
            X_red = self.reduce_dimension(n_components=n_components)
            for k in clusters_range:
                if self.clustering_method == "kmeans":
                    model = KMeans(
                        n_clusters=k,
                        n_init=100,
                        max_iter=1000,
                        random_state=42,
                    )
                    clusters = model.fit_predict(X_red)
                elif self.clustering_method == "gmm":
                    model = GaussianMixture(
                        n_components=k,
                        max_iter=500,
                        n_init=10,
                        random_state=42,
                    )
                    model.fit(X_red)
                    clusters = model.predict(X_red)
                elif self.clustering_method in ["agglomerative", "hierarchical"]:
                    model = AgglomerativeClustering(
                        n_clusters=k,
                    )
                    clusters = model.fit_predict(X_red)
                else:
                    continue
                silhouette_avg = silhouette_score(X_red, clusters)
                calinski = calinski_harabasz_score(X_red, clusters)
                davies = davies_bouldin_score(X_red, clusters)
                results.append({
                    'n_components': n_components,
                    'n_clusters': k,
                    'silhouette': silhouette_avg,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies
                })
        return pd.DataFrame(results)

    def find_optimal_dimensions(self, output_folder: str) -> Dict[str, Any]:
        """
        Finds the optimal number of PCA components based on cumulative explained variance.

        Args:
            output_folder (str): Folder path where the explained variance plot will be saved.

        Returns:
            Dict: Dictionary with explained variance metrics.
        """
        if self.X_scaled is None:
            raise ValueError("Data must be normalized before computing PCA optimal dimensions.")
        pca = PCA(n_components=min(self.X_scaled.shape[1], self.X_scaled.shape[0]))
        pca.fit(self.X_scaled)

        # Calculate cumulative explained variance
        cum_explained = np.cumsum(pca.explained_variance_ratio_)
        n_comp = {
            'n_components_80': int(np.argmax(cum_explained >= 0.8) + 1),
            'n_components_90': int(np.argmax(cum_explained >= 0.9) + 1),
            'n_components_95': int(np.argmax(cum_explained >= 0.95) + 1)
        }

        colors = {'80%': 'r', '90%': 'g', '95%': 'orange'}

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cum_explained)+1), cum_explained, 'bo-')
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")

        for thresh, label in zip([0.8, 0.9, 0.95], ["80%", "90%", "95%"]):
            plt.axhline(y=thresh, linestyle="--", color=colors[label], label=f"{label} Explained Variance")
            plt.axvline(x=n_comp[f'n_components_{label[:2]}'], linestyle="--", color=colors[label], alpha=0.3)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "pca_explained_variance.png"), dpi=300)
        plt.close()

        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_explained_variance': cum_explained,
            **n_comp
        }

    def find_optimal_clusters(self, output_folder: str, n_components: int = 2, max_clusters: int = 15) -> Dict[str, List[float]]:
        """
        Determines the optimal number of clusters using evaluation metrics for methods with a defined cluster count.

        Args:
            output_folder (str): Folder path where plots will be saved.
            n_components (int, optional): Number of PCA components to use. Defaults to 2.
            max_clusters (int, optional): Maximum number of clusters to test. Defaults to 10.

        Returns:
            Dict[str, List[float]]: Dictionary containing evaluation metrics.
        """
        if self.X_scaled is None:
            raise ValueError("Data must be normalized before computing optimal clusters.")
        X_reduced = self.reduce_dimension(n_components=n_components)

        results = {
            'n_values': list(range(2, max_clusters + 1)),
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }

        for k in results['n_values']:
            if self.clustering_method == "kmeans":
                model = KMeans(n_clusters=k, n_init=100, max_iter=1000, random_state=42)
                clusters = model.fit_predict(X_reduced)
                inertia = model.inertia_
            elif self.clustering_method == "gmm":
                model = GaussianMixture(n_components=k, random_state=42)
                model.fit(X_reduced)
                clusters = model.predict(X_reduced)
                inertia = np.nan  # Inertia is not defined for GMM
            elif self.clustering_method in ["agglomerative", "hierarchical"]:
                model = AgglomerativeClustering(n_clusters=k)
                clusters = model.fit_predict(X_reduced)
                inertia = np.nan  # Inertia is not defined for hierarchical clustering
            else:
                raise ValueError(f"Optimal clusters search not supported for method {self.clustering_method}")
            results['inertia'].append(inertia)
            results['silhouette'].append(silhouette_score(X_reduced, clusters))
            results['calinski_harabasz'].append(calinski_harabasz_score(X_reduced, clusters))
            results['davies_bouldin'].append(davies_bouldin_score(X_reduced, clusters))
        
        # Plot the evaluation metrics
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        axs[0, 0].plot(results['n_values'], results['inertia'], 'bo-')
        axs[0, 0].set_xlabel('Number of clusters (k)')
        axs[0, 0].set_ylabel('Inertia (WCSS)')
        axs[0, 0].set_title('Elbow Method')
        
        axs[0, 1].plot(results['n_values'], results['silhouette'], 'ro-')
        axs[0, 1].set_xlabel('Number of clusters (k)')
        axs[0, 1].set_ylabel('Silhouette Score')
        axs[0, 1].set_title('Silhouette Method (higher is better)')
        
        axs[1, 0].plot(results['n_values'], results['calinski_harabasz'], 'go-')
        axs[1, 0].set_xlabel('Number of clusters (k)')
        axs[1, 0].set_ylabel('Calinski-Harabasz Score')
        axs[1, 0].set_title('Calinski-Harabasz Index (higher is better)')
        
        axs[1, 1].plot(results['n_values'], results['davies_bouldin'], 'mo-')
        axs[1, 1].set_xlabel('Number of clusters (k)')
        axs[1, 1].set_ylabel('Davies-Bouldin Score')
        axs[1, 1].set_title('Davies-Bouldin Index (lower is better)')
        
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f"optimal_k_analysis_pca{n_components}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return results

class ExtremeMedianClusterAnalysis:
    """
    Runs clustering on both extreme and median datasets.
    Produces individual cluster plots for each condition (scatter, bar, and radar)
    and comparative visualizations (transition matrices, Sankey diagram, feature shift analysis).
    """

    def __init__(self, 
                 run_dir: str, 
                 epoch: int, 
                 period: str, 
                 extreme_n_clusters: int, 
                 median_n_clusters: int,
                 extreme_n_dim: int,
                 median_n_dim: int,
                 clustering_method: str) -> None:
        """
        Initializes the ExtremeMedianClusterAnalysis.

        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number for model selection.
            period (str): Period identifier (e.g., train, validation, test).
            extreme_n_clusters (int): Number of clusters for extreme condition.
            median_n_clusters (int): Number of clusters for median condition.
            extreme_n_dim (int): Number of PCA dimensions for extreme condition.
            median_n_dim (int): Number of PCA dimensions for median condition.
        """
        self.run_dir = run_dir
        self.epoch = epoch
        self.period = period
        self.extreme_n_clusters = extreme_n_clusters
        self.median_n_clusters = median_n_clusters
        self.extreme_n_dim = extreme_n_dim
        self.median_n_dim = median_n_dim
        self.clustering_method = clustering_method.lower()
        # Data loaders and analyzers for extreme and median conditions
        self.extreme_loader: Optional[SHAPDataLoader] = None
        self.median_loader: Optional[SHAPDataLoader] = None
        self.extreme_analyzer: Optional[ClusterAnalyzer] = None
        self.median_analyzer: Optional[ClusterAnalyzer] = None
        self.feature_names: List[str] = []
        self.common_basin_ids: List[str] = []
        self.feature_shift_df: Optional[pd.DataFrame] = None
        self.transition_matrix: Optional[np.ndarray] = None
        self.gauge_mapping: Dict[str, Dict] = {}
        self.setup_output_folders()

    def setup_output_folders(self):
        """
        Sets up the output folders for saving analysis results and plots.
        """
        base_results_folder = os.path.join(
            self.run_dir, 
            self.period, 
            f"model_epoch{self.epoch:03d}", 
            "shap", 
            "basins_clustered", 
            self.clustering_method
            )
        self.results_folder = os.path.join(base_results_folder, "cluster_analysis")
        self.extreme_folder = os.path.join(base_results_folder, "extreme")
        self.median_folder = os.path.join(base_results_folder, "median")
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.extreme_folder, exist_ok=True)
        os.makedirs(self.median_folder, exist_ok=True)

    def load_data(self) -> None:
        """
        Loads extreme and median SHAP data, and determines common basin IDs.
        """
        extreme_file = os.path.join(self.run_dir, self.period, f"model_epoch{self.epoch:03d}", "shap", "aggregated_shap_extreme.p")
        median_file = os.path.join(self.run_dir, self.period, f"model_epoch{self.epoch:03d}", "shap", "aggregated_shap_median.p")
        
        extreme_exists = os.path.exists(extreme_file)
        median_exists = os.path.exists(median_file)
        
        if not extreme_exists and not median_exists:
            raise FileNotFoundError("Neither extreme nor median SHAP file found.")
        
        if extreme_exists:
            self.extreme_loader = SHAPDataLoader(extreme_file)
            self.extreme_loader.load()
        else:
            logging.warning(f"Extreme file not found: {extreme_file}.")
        
        if median_exists:
            self.median_loader = SHAPDataLoader(median_file)
            self.median_loader.load()
        else:
            logging.warning(f"Median file not found: {median_file}.")
        
        # Determine common basins if both datasets are present
        if self.extreme_loader and self.median_loader:
            extreme_ids = set(self.extreme_loader.basin_ids)
            median_ids = set(self.median_loader.basin_ids)
            self.common_basin_ids = list(extreme_ids.intersection(median_ids))
        elif self.extreme_loader:
            self.common_basin_ids = self.extreme_loader.basin_ids
        elif self.median_loader:
            self.common_basin_ids = self.median_loader.basin_ids

    def load_gauge_mapping(self, base_dir: str = "/sci/labs/efratmorin/lab_share/FloodsML/data/Caravan/attributes"):
        """
        Loads gauge latitude and longitude information from CSV files in the specified directory.
        It reads the 'attributes_other_{source}.csv' file from each relevant subdirectory.

        Args:
            base_dir (str, optional): Base directory containing the attribute files. 
                                      Defaults to "/sci/labs/efratmorin/lab_share/FloodsML/data/Caravan/attributes".
        """
        directories = ["camels", "camelsaus", "camelsbr", "camelscl", "camelsgb", "hysets", "lamah"]
        mapping = {}
        for d in directories:
            file_path = os.path.join(base_dir, d, f"attributes_other_{d}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Expecting columns 'gauge_id', 'gauge_lat', and 'gauge_lon'
                for _, row in df.iterrows():
                    mapping[row["gauge_id"]] = {"gauge_lat": row["gauge_lat"], "gauge_lon": row["gauge_lon"]}
            else:
                logging.warning(f"File {file_path} not found.")
        self.gauge_mapping = mapping

    def setup_analyzers(self) -> None:
        """
        Initializes the ClusterAnalyzer for both extreme and median datasets and
        performs normalization and dimensionality reduction.
        """
        if self.extreme_loader:
            extreme_data = np.array([self.extreme_loader.aggregated[bid]["signed"] for bid in self.common_basin_ids])
            self.extreme_analyzer = ClusterAnalyzer(extreme_data, self.extreme_n_clusters, self.extreme_n_dim,
                                                    clustering_method=self.clustering_method)
            self.extreme_analyzer.normalize()
            self.extreme_analyzer.reduce_dimension()
        
        if self.median_loader:
            median_data = np.array([self.median_loader.aggregated[bid]["signed"] for bid in self.common_basin_ids])
            self.median_analyzer = ClusterAnalyzer(median_data, self.median_n_clusters, self.median_n_dim,
                                                    clustering_method=self.clustering_method)
            self.median_analyzer.normalize()
            self.median_analyzer.reduce_dimension()

    def run_clustering(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Performs clustering for both extreme and median datasets.

        Returns:
            Tuple[Dict[str, int], Dict[str, int]]:
                - extreme_clusters: Mapping of basin ID to cluster for extreme condition.
                - median_clusters: Mapping of basin ID to cluster for median condition.
        """
        extreme_clusters = None
        median_clusters = None
        
        if self.extreme_analyzer:
            self.extreme_analyzer.perform_clustering()
            extreme_clusters = {bid: cluster for bid, cluster in zip(self.common_basin_ids, self.extreme_analyzer.clusters)}
        
        if self.median_analyzer:
            self.median_analyzer.perform_clustering()
            median_clusters = {bid: cluster for bid, cluster in zip(self.common_basin_ids, self.median_analyzer.clusters)}
        
        return extreme_clusters, median_clusters
    
    def plot_geographic_clusters(self, clusters: Dict[str, int], condition: str) -> None:
        """
        Uses the helper function to plot geographic clusters based on gauge mapping.

        Args:
            clusters (Dict[str, int]): Mapping of basin ID to cluster label.
            condition (str): Condition identifier ('extreme' or 'median').
        """
        params = {
            "extreme": {
                "analyzer": self.extreme_analyzer,
                "n_dim": self.extreme_n_dim,
                "n_clusters": self.extreme_n_clusters,
                "folder": self.extreme_folder
            },
            "median": {
                "analyzer": self.median_analyzer,
                "n_dim": self.median_n_dim,
                "n_clusters": self.median_n_clusters,
                "folder": self.median_folder
            }
        }[condition]

        pca_dim = params["n_dim"] if params["analyzer"].apply_pca else "None"
        title = f"{condition.capitalize()}, PCA Dim: {pca_dim}, Clusters: {params['n_clusters']}"

        plot_clusters_on_world_map(
            basins=self.common_basin_ids,
            cluster_dict=clusters,
            gauge_mapping=self.gauge_mapping,
            output_dir=params["folder"],
            title_details=title
        )

    def cluster_size_distribution(self, extreme_clusters: Dict[str, int], median_clusters: Dict[str, int]) -> None:
        """
        Saves the cluster sizes for both extreme and median conditions to a CSV file.

        Args:
            extreme_clusters (Dict[str, int]): Mapping of basin ID to cluster for extreme condition.
            median_clusters (Dict[str, int]): Mapping of basin ID to cluster for median condition.
        """
        extreme_counts = (
            pd.Series(extreme_clusters)
            .value_counts()
            .sort_index()
            .rename_axis('cluster')
            .reset_index(name='count')
        )
        extreme_counts['condition'] = 'extreme'
        
        median_counts = (
            pd.Series(median_clusters)
            .value_counts()
            .sort_index()
            .rename_axis('cluster')
            .reset_index(name='count')
        )
        median_counts['condition'] = 'median'
        
        cluster_sizes = pd.concat([extreme_counts, median_counts], ignore_index=True)
        csv_path = os.path.join(self.results_folder, "cluster_sizes.csv")
        cluster_sizes.to_csv(csv_path, index=False)

    def load_scaler_from_file(self) -> dict:
        """Loads the scaler dictionary from the run directory.
        Returns:
            dict: The scaler dictionary containing means and standard deviations.
        """
        scaler_path = Path(self.run_dir) / 'train_data' / 'train_data_scaler.yml'
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        with open(scaler_path, 'r') as fp:
            scaler = yaml.safe_load(fp)
        return scaler

    def _aggregate_unnormalized_data_per_basin(self, condition: str, basin_id_to_print: Optional[str] = "camelscl_9402001") -> Dict[str, Dict[str, np.ndarray]]:
        """
        Aggregates un-normalized input data per basin, calculating both mean and median.

        Args:
            condition (str): Either 'extreme' or 'median' indicating the condition.
            basin_id_to_print (Optional[str]): If provided, print the aggregated mean and median
                                            for this specific basin for debugging.

        Returns:
            dict: Mapping from basin ID to a dictionary containing 'mean' and 'median'
                aggregated un-normalized data vectors.
                Example: {'basin_X': {'mean': np.array([...]), 'median': np.array([...])}}
        """
        # Load Normalized Data
        inputs_path = os.path.join(self.run_dir, self.period, f"model_epoch{self.epoch:03d}", "shap", f"inputs_{condition}.npz")
        if not os.path.exists(inputs_path):
            raise FileNotFoundError(f"Inputs file not found for condition '{condition}' at {inputs_path}")
        inputs = np.load(inputs_path)
        norm_x_d = inputs['x_d']
        norm_x_s = inputs['x_s']
        basin_ids = inputs['basin_ids']

        # Load the Scaler
        scaler = self.load_scaler_from_file()
        xarray_center_dict = scaler.get('xarray_feature_center', {})
        xarray_scale_dict = scaler.get('xarray_feature_scale', {})
        attribute_means_dict = scaler.get('attribute_means', {})
        attribute_stds_dict = scaler.get('attribute_stds', {})

        # Get Feature Names and Split into Dynamic and Static
        data_loader = self.extreme_loader if condition == "extreme" else self.median_loader
        if not data_loader or not data_loader.feature_names:
            raise ValueError(f"Feature names not loaded for condition '{condition}'.")
        num_dynamic_features = norm_x_d.shape[2]
        dynamic_features = data_loader.feature_names[:num_dynamic_features]
        static_features = data_loader.feature_names[num_dynamic_features:]

        if len(dynamic_features) + len(static_features) != len(data_loader.feature_names):
            raise ValueError("Feature name split mismatch.")
        if len(dynamic_features) != norm_x_d.shape[2] or len(static_features) != norm_x_s.shape[1]:
            raise ValueError("Feature counts don't match data shapes.")

        # Dynamic feature means and stds
        dyn_centers = np.zeros(num_dynamic_features)
        dyn_scales = np.ones(num_dynamic_features)
        for j, feat_name in enumerate(dynamic_features):
            center_data = xarray_center_dict.get('data_vars', {}).get(feat_name, {}).get('data')
            scale_data = xarray_scale_dict.get('data_vars', {}).get(feat_name, {}).get('data')
            dyn_centers[j] = center_data if center_data is not None else 0.0
            dyn_scales[j] = scale_data if scale_data is not None and scale_data != 0 else 1.0

        # Static feature means and stds
        stat_centers = np.zeros(len(static_features))
        stat_scales = np.ones(len(static_features))
        for k, feat_name in enumerate(static_features):
            stat_centers[k] = attribute_means_dict.get(feat_name, 0.0)
            scale_val = attribute_stds_dict.get(feat_name, 1.0)
            stat_scales[k] = scale_val if scale_val != 0 else 1.0

        # Un-normalize data sample by sample
        unnormalized_samples = []
        last_day_dyn_norm = norm_x_d[:, -1, :]
        for i in range(len(norm_x_d)):
            raw_x_d_last_day_i = (last_day_dyn_norm[i] * dyn_scales) + dyn_centers
            raw_x_s_i = (norm_x_s[i] * stat_scales) + stat_centers
            unnormalized_combined = np.concatenate([raw_x_d_last_day_i, raw_x_s_i])
            unnormalized_samples.append(unnormalized_combined)
        unnormalized_samples = np.array(unnormalized_samples)

        # Aggregate per basin (mean and median)
        combined_features = dynamic_features + static_features
        df = pd.DataFrame(unnormalized_samples, columns=combined_features)
        df['basin_id'] = basin_ids
        grouped_stats = df.groupby('basin_id')[combined_features].agg(['mean', 'median'])
        # grouped_stats DataFrame will have multi-level columns: (feature_name, 'mean') and (feature_name, 'median')

        aggregated_per_basin = {}
        for basin_id, row in grouped_stats.iterrows():
            mean_vector = row.loc[(slice(None), 'mean')].values 
            median_vector = row.loc[(slice(None), 'median')].values
            aggregated_per_basin[basin_id] = {"mean": mean_vector, "median": median_vector}

        return aggregated_per_basin

    def aggregate_raw_data_per_cluster(self, condition, basin_to_cluster: dict[str, int]) -> pd.DataFrame:
        """
        Aggregates basin-level un-normalized data into representative vectors (mean and median)
        per cluster. Saves the results to a CSV file.

        Args:
            condition (str): Either 'extreme' or 'median' indicating the condition.
            basin_to_cluster (dict): Mapping from basin_id to cluster label.

        Returns:
            pd.DataFrame: DataFrame with features as index and multi-level columns
                        (ClusterX_mean, ClusterX_median) containing aggregated un-normalized data.
        """
        # Get per-basin aggregations
        aggregation_key_per_basin = 'median' # Choose 'mean' or 'median' per-basin vectors as input
        aggregated_per_basin_unnormalized = self._aggregate_unnormalized_data_per_basin(condition)

        # Prepare data for cluster aggregation
        missing_basins = [bid for bid in self.common_basin_ids if bid not in aggregated_per_basin_unnormalized]
        if missing_basins:
            logging.warning(f"Missing unnormalized data for basins: {missing_basins}. Excluded.")

        valid_common_basins = [bid for bid in self.common_basin_ids if bid in aggregated_per_basin_unnormalized]
        if not valid_common_basins:
            logging.error(f"No common basins with unnormalized data found for '{condition}'. Cannot aggregate.")
            return pd.DataFrame()

        vectors = np.array([aggregated_per_basin_unnormalized[basin_id][aggregation_key_per_basin]
                            for basin_id in valid_common_basins])
        clusters = np.array([basin_to_cluster[basin_id] for basin_id in valid_common_basins])

        data_loader = self.extreme_loader if condition == "extreme" else self.median_loader
        if not data_loader or not data_loader.feature_names:
            raise ValueError("Data loader or feature names not available.")
        feature_names = data_loader.feature_names

        df_cluster_input = pd.DataFrame(vectors, columns=feature_names)
        df_cluster_input['cluster'] = clusters

        # Calculate mean and median per cluster
        cluster_means = df_cluster_input.groupby('cluster')[feature_names].mean().T
        cluster_means.columns = [f"Cluster{c+1}_mean" for c in cluster_means.columns]
        cluster_medians = df_cluster_input.groupby('cluster')[feature_names].median().T
        cluster_medians.columns = [f"Cluster{c+1}_median" for c in cluster_medians.columns]

        df_final_output = pd.concat([cluster_means, cluster_medians], axis=1)

        # Sort columns for consistent order (Cluster0_mean, Cluster0_median, Cluster1_mean, ...)
        df_final_output = df_final_output.sort_index(axis=1)

        output_folder = self.extreme_folder if condition == "extreme" else self.median_folder
        output_file = os.path.join(output_folder, f"cluster_aggregated_raw_data_{condition}.csv")
        logging.info(f"Saving cluster aggregated raw data to {output_file}")
        df_final_output.to_csv(output_file, index=True, index_label="feature")

        return df_final_output

    def generate_cluster_visualizations(self, condition: str, analyzer: ClusterAnalyzer, loader: SHAPDataLoader, n_clusters: int) -> None:
        """
        Generates cluster visualization plots including scatter, bar, and radar plots for the specified condition.

        Args:
            condition (str): Either 'extreme' or 'median' indicating the condition.
            analyzer (ClusterAnalyzer): Analyzer instance used for clustering.
            loader (SHAPDataLoader): Data loader instance used to access SHAP data.
            n_clusters (int): Number of clusters used in the clustering.
        """
        pca_dim = self.extreme_n_dim if condition == "extreme" else self.median_n_dim
        pca_dim = pca_dim if analyzer.apply_pca else "None"
        X_reduced = analyzer.X_reduced
        clusters = analyzer.clusters

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap="viridis", s=50, alpha=0.8)
        plt.title(f"Clustered Basins ({condition.capitalize()}, PCA Dim: {pca_dim}, Clusters: {n_clusters})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(self.extreme_folder if condition == "extreme" else self.median_folder, f"{n_clusters}_clusters_plot.png"), dpi=300)
        plt.close()

        # For each cluster, plot aggregated feature profiles (bar & radar)
        clusters_dict: Dict[int, List[str]] = {}
        for bid, cluster in zip(self.common_basin_ids, clusters):
            clusters_dict.setdefault(cluster, []).append(bid)
        for cluster, basin_ids in clusters_dict.items():
            profiles = [loader.aggregated[bid]["signed"] for bid in basin_ids]
            profiles_abs = [loader.aggregated[bid]["absolute"] for bid in basin_ids]
            median_profile = np.median(np.array(profiles), axis=0)
            median_profile_abs = np.median(np.array(profiles_abs), axis=0)
            self._generate_bar_plot(
                condition=condition,
                loader=loader,
                cluster=cluster + 1,
                median_profile=median_profile,
                median_profile_abs=median_profile_abs,
                pca_dim=pca_dim,
            )
            self._generate_radar_plot(
                condition=condition,
                loader=loader,
                cluster=cluster + 1,
                median_profile=median_profile,
                pca_dim=pca_dim,
            )

    def _generate_radar_plot(self, condition: str, loader: SHAPDataLoader, cluster: int, median_profile: np.ndarray, pca_dim: int) -> None:
        """
        Generates a radar plot for a specific cluster.

        Args:
            condition (str): Condition identifier ('extreme' or 'median').
            loader (SHAPDataLoader): Data loader to access feature names.
            cluster (int): Cluster number.
            median_profile (np.ndarray): Median SHAP profile for the cluster.
        """
        num_vars = len(median_profile)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        values = median_profile.tolist()
        values += values[:1]
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], loader.feature_names, fontsize=6)
        ax.plot(angles, values, color='b', linewidth=2)
        ax.fill(angles, values, color='b', alpha=0.25)
        plt.title(f"Cluster {cluster} Aggregated Feature Profile ({condition.capitalize()}, PCA Dim: {pca_dim})", y=1.08)
        plt.tight_layout()
        plt.savefig(os.path.join(self.extreme_folder if condition == "extreme" else self.median_folder, f"cluster_{cluster}_radar_plot.png"), dpi=300)
        plt.close()

    def _generate_bar_plot(self, condition: str, loader: SHAPDataLoader, cluster: int, median_profile: np.ndarray,
                            median_profile_abs: np.ndarray, pca_dim: int) -> None:
        """
        Generates a horizontal bar plot for a specific cluster.

        Args:
            condition (str): Condition identifier ('extreme' or 'median').
            loader (SHAPDataLoader): Data loader to access feature names.
            cluster (int): Cluster number.
            median_profile (np.ndarray): Median signed SHAP values for the cluster.
            median_profile_abs (np.ndarray): Median absolute SHAP values for the cluster.
        """
        num_features = len(loader.feature_names)
        y = np.arange(num_features)
        bar_height = 0.35
        plt.figure(figsize=(10, 8))
        plt.barh(y - bar_height/2, median_profile, height=bar_height, color='skyblue', label='Signed Median')
        plt.barh(y + bar_height/2, median_profile_abs, height=bar_height, color='salmon', alpha=0.3, label='Absolute Median')
        plt.yticks(y, loader.feature_names, fontsize=6)
        plt.gca().invert_yaxis()
        plt.xlabel("Aggregated SHAP Value")        
        plt.title(f"Cluster {cluster} Aggregated Feature Profile ({condition.capitalize()}, PCA Dim: {pca_dim})", fontsize=12)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.extreme_folder if condition == "extreme" else self.median_folder, f"cluster_{cluster}_bar_plot.png"), dpi=300)
        plt.close()

    def create_transition_matrix(self, extreme_clusters: Dict[str, int], median_clusters: Dict[str, int]) -> np.ndarray:
        """
        Creates a transition matrix representing the shifts from extreme clusters to median clusters.

        Args:
            extreme_clusters (Dict[str, int]): Mapping of basin IDs to extreme clusters.
            median_clusters (Dict[str, int]): Mapping of basin IDs to median clusters.

        Returns:
            np.ndarray: Transition matrix of shape (n_extreme_clusters, n_median_clusters).
        """
        transition_matrix = np.zeros((self.extreme_n_clusters, self.median_n_clusters))
        for bid in self.common_basin_ids:
            transition_matrix[extreme_clusters[bid], median_clusters[bid]] += 1
        self.transition_matrix = transition_matrix
        return transition_matrix

    def plot_transition_matrices(self, transition_matrix: np.ndarray) -> None:
        """
        Plots heatmaps of the transition matrices between extreme and median clusters.

        Args:
            transition_matrix (np.ndarray): Transition matrix to visualize.
        """
        # Extreme-to-Median heatmap
        plt.figure(figsize=(10, 8))
        percentage_ext = (transition_matrix / transition_matrix.sum(axis=1, keepdims=True)) * 100
        ax = sns.heatmap(percentage_ext, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
        ax.set_xlabel("Median Cluster")
        ax.set_ylabel("Extreme Cluster")
        ax.set_title("Transition Matrix: Extreme to Median")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, "transition_matrix_extreme_to_median.png"), dpi=300)
        plt.close()
        # Median-to-Extreme heatmap
        plt.figure(figsize=(10, 8))
        percentage_med = (transition_matrix.T / transition_matrix.T.sum(axis=1, keepdims=True)) * 100
        ax = sns.heatmap(percentage_med, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
        ax.set_xlabel("Extreme Cluster")
        ax.set_ylabel("Median Cluster")
        ax.set_title("Transition Matrix: Median to Extreme")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, "transition_matrix_median_to_extreme.png"), dpi=300)
        plt.close()

    def plot_sankey_diagram(self, transition_matrix: np.ndarray) -> None:
        """
        Plots a Sankey diagram to visualize transitions from extreme clusters to median clusters.

        Args:
            transition_matrix (np.ndarray): Transition matrix used for constructing the Sankey diagram.
        """
        plt.figure(figsize=(12, 10))
        sankey = Sankey(ax=plt.gca(), scale=0.01, offset=0.2, head_angle=150, margin=0.05)
        for ext_cluster in range(self.extreme_n_clusters):
            flows, labels = [], []
            for med_cluster in range(self.median_n_clusters):
                count = transition_matrix[ext_cluster, med_cluster]
                if count > 0:
                    flows.append(count)
                    labels.append(f"Median {med_cluster}")
            if flows:
                sankey.add(flows=flows, labels=labels, orientations=[0]*len(flows),
                           trunklength=1.0, rotation=90, label=f"Extreme {ext_cluster}")
        sankey.finish()
        plt.title("Basin Transitions from Extreme to Median Clusters")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, "sankey_diagram.png"), dpi=300)
        plt.close()
    
    def calculate_feature_shifts(self, extreme_clusters: Dict[str, int], median_clusters: Dict[str, int]) -> pd.DataFrame:
        """
        Calculates the shifts in SHAP values between extreme and median conditions for each basin.
        Saves the resulting DataFrame as a CSV file.

        Args:
            extreme_clusters (Dict[str, int]): Mapping of basin IDs to extreme clusters.
            median_clusters (Dict[str, int]): Mapping of basin IDs to median clusters.
        """
        feature_shifts = []
        for bid in self.common_basin_ids:
            ext_cluster = extreme_clusters[bid]
            med_cluster = median_clusters[bid]
            ext_shap = self.extreme_loader.aggregated[bid]["signed"]
            med_shap = self.median_loader.aggregated[bid]["signed"]
            shift = ext_shap - med_shap
            feature_shifts.append({
                "basin_id": bid,
                "extreme_cluster": ext_cluster,
                "median_cluster": med_cluster,
                "transition": f"Extreme {ext_cluster} → Median {med_cluster}",
                **{f"shift_{name}": shift[i] for i, name in enumerate(self.feature_names)}
            })
        self.feature_shift_df = pd.DataFrame(feature_shifts)
        self.feature_shift_df.to_csv(os.path.join(self.results_folder, "feature_shifts.csv"), index=False)
    
    def plot_top_feature_shifts(self, n_features: int = 10) -> None:
        """
        Plots the top features that exhibit the largest shifts in SHAP values.
        Generates a boxplot for feature shifts and a heatmap of average shifts per transition.

        Args:
            n_features (int, optional): Number of top features to display. Defaults to 10.
        
        Raises:
            ValueError: If feature shifts have not been calculated.
        """
        if self.feature_shift_df is None:
            raise ValueError("Feature shifts not calculated. Run calculate_feature_shifts() first.")
        feature_cols = [col for col in self.feature_shift_df.columns if col.startswith("shift_")]
        avg_abs_shifts = {col: abs(self.feature_shift_df[col]).mean() for col in feature_cols}
        sorted_features = sorted(avg_abs_shifts.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature[0] for feature in sorted_features[:n_features]]
        
        # Create boxplot for top features
        plt.figure(figsize=(12, 8))
        selected_data = self.feature_shift_df[["transition"] + top_features]
        melted_data = pd.melt(
            selected_data, 
            id_vars=["transition"], 
            value_vars=top_features,
            var_name="Feature",
            value_name="SHAP Shift"
        )
        melted_data["Feature"] = melted_data["Feature"].str.replace("shift_", "")
        sns.boxplot(x="Feature", y="SHAP Shift", data=melted_data)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Top {n_features} Features with Largest SHAP Value Shifts")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, "top_feature_shifts.png"), dpi=300)
        plt.close()
        
        # Plot heatmap of average shift per transition
        plt.figure(figsize=(14, 10))
        transitions = self.feature_shift_df["transition"].unique()
        avg_shifts = {}
        for transition in transitions:
            mask = self.feature_shift_df["transition"] == transition
            avg_shifts[transition] = {
                feature.replace("shift_", ""): self.feature_shift_df.loc[mask, feature].mean() 
                for feature in top_features
            }
        heatmap_df = pd.DataFrame(avg_shifts).T
        sns.heatmap(
            heatmap_df, 
            cmap="coolwarm", 
            center=0, 
            annot=True, 
            fmt=".2f",
            linewidths=0.5
        )
        plt.title("Average SHAP Value Shifts per Transition")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, "transition_feature_shifts.png"), dpi=300)
        plt.close()

    def calculate_cluster_stability(self, extreme_clusters: Dict[str, int], median_clusters: Dict[str, int]) -> pd.DataFrame:
        """
        Calculates cluster stability metrics to quantify how basins shift between clusters.
        Saves the stability metrics and individual basin stability data as CSV files.

        Args:
            extreme_clusters (Dict[str, int]): Mapping of basin IDs to extreme clusters.
            median_clusters (Dict[str, int]): Mapping of basin IDs to median clusters.

        Returns:
            pd.DataFrame: DataFrame containing cluster stability metrics.
        
        Raises:
            ValueError: If the transition matrix has not been calculated.
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not calculated. Run create_transition_matrix() first.")
        predominant_mapping = {}
        for ext_cluster in range(self.extreme_n_clusters):
            median_counts = self.transition_matrix[ext_cluster]
            predominant_median = np.argmax(median_counts)
            predominant_mapping[ext_cluster] = predominant_median
        
        stability_data = []
        for basin_id in self.common_basin_ids:
            ext_cluster = extreme_clusters[basin_id]
            med_cluster = median_clusters[basin_id]
            follows_predominant = (med_cluster == predominant_mapping[ext_cluster])
            stability_data.append({
                "basin_id": basin_id,
                "extreme_cluster": ext_cluster,
                "median_cluster": med_cluster,
                "follows_predominant_transition": follows_predominant
            })
        stability_df = pd.DataFrame(stability_data)
        stability_metrics = []
        for ext_cluster in range(self.extreme_n_clusters):
            mask = stability_df["extreme_cluster"] == ext_cluster
            cluster_df = stability_df[mask]
            stability_metrics.append({
                "extreme_cluster": ext_cluster,
                "predominant_median_cluster": predominant_mapping[ext_cluster],
                "total_basins": len(cluster_df),
                "stable_basins": cluster_df["follows_predominant_transition"].sum(),
                "stability_ratio": cluster_df["follows_predominant_transition"].mean() * 100
            })
        metrics_df = pd.DataFrame(stability_metrics)
        metrics_df.to_csv(os.path.join(self.results_folder, "cluster_stability.csv"), index=False)
        stability_df.to_csv(os.path.join(self.results_folder, "basin_stability.csv"), index=False)
        return metrics_df

    def run(self) -> None:
        """
        Runs the full extreme and median cluster analysis pipeline.
        """
        self.load_data()
        self.load_gauge_mapping()
        self.setup_analyzers()
        extreme_clusters, median_clusters = self.run_clustering()

        if self.extreme_loader:
            self.aggregate_raw_data_per_cluster("extreme", extreme_clusters)
            self.generate_cluster_visualizations("extreme", self.extreme_analyzer, self.extreme_loader, self.extreme_n_clusters)
            self.plot_geographic_clusters(extreme_clusters, "extreme")
        if self.median_loader:
            self.aggregate_raw_data_per_cluster("median", median_clusters)
            self.generate_cluster_visualizations("median", self.median_analyzer, self.median_loader, self.median_n_clusters)
            self.plot_geographic_clusters(median_clusters, "median")
        
        # Run comparative analyses if both conditions are available
        if self.extreme_loader and self.median_loader:
            self.cluster_size_distribution(extreme_clusters, median_clusters)
            transition_matrix = self.create_transition_matrix(extreme_clusters, median_clusters)
            self.plot_transition_matrices(transition_matrix)
            self.plot_sankey_diagram(transition_matrix)
            self.calculate_feature_shifts(extreme_clusters, median_clusters)
            # self.plot_top_feature_shifts()
            self.calculate_cluster_stability(extreme_clusters, median_clusters)
            logging.info(f"Comparative analysis completed. Results saved in: {self.results_folder}")
        else:
            logging.info(f"Single condition analysis completed. Results saved in: {self.results_folder}")

class OptimalParameterSearch:
    """
    Runs a grid search to find optimal PCA dimensions and number of clusters
    for a single condition (extreme or median). Saves a CSV of results and a heatmap.
    Also runs the optimal dimensions analysis.
    """

    def __init__(self, 
                 run_dir: str, 
                 epoch: int, 
                 period: str, 
                 clustering_method: str,
                 filter_type: str,
                 components_range: Tuple[int, int] = (2, 66),
                 clusters_range: Tuple[int, int] = (2, 16)) -> None:
        """
        Initializes the OptimalParameterSearch.

        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number for model selection.
            period (str): Period identifier (e.g., train, validation, test).
            filter_type (str): Condition filter type ('extreme' or 'median').
            components_range (Tuple[int, int], optional): Range (min, max) for PCA components. Defaults to (2, 66).
            clusters_range (Tuple[int, int], optional): Range (min, max) for number of clusters. Defaults to (2, 16).
        """
        self.run_dir = run_dir
        self.epoch = epoch
        self.period = period
        self.filter_type = filter_type.lower()
        self.clustering_method = clustering_method.lower()
        self.components_range = np.arange(*components_range)
        self.clusters_range = np.arange(*clusters_range)
        self.results_folder = os.path.join(
            self.run_dir, 
            self.period, 
            f"model_epoch{self.epoch:03d}", 
            "shap", 
            "basins_clustered",
            self.clustering_method,
            self.filter_type
        )
        os.makedirs(self.results_folder, exist_ok=True)
        self.data_loader: Optional[SHAPDataLoader] = None
        self.cluster_analyzer: Optional[ClusterAnalyzer] = None

    def load_data(self) -> None:
        """
        Loads the SHAP data for the specified filter type.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        file_path = os.path.join(self.run_dir, self.period, f"model_epoch{self.epoch:03d}", "shap", f"aggregated_shap_{self.filter_type}.p")
        self.data_loader = SHAPDataLoader(file_path)
        self.data_loader.load()

    def run_grid_search(self) -> None:
        """
        Runs a grid search over PCA components and number of clusters.
        Saves the grid search results as a CSV file and a heatmap plot.
        """
        grid_results = self.cluster_analyzer.grid_search(self.components_range, self.clusters_range)
        csv_path = os.path.join(self.results_folder, "grid_search_results.csv")
        grid_results.to_csv(csv_path, index=False)
        pivot = grid_results.pivot(index='n_components', columns='n_clusters', values='silhouette')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f"{self.filter_type.capitalize()} condition: Silhouette Score Heatmap")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Number of PCA Components")
        plt.tight_layout()
        heatmap_path = os.path.join(self.results_folder, "grid_search_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        logging.info(f"Grid search completed for {self.filter_type}. Results saved in: {self.results_folder}")

    def run_optimal_analysis(self) -> None:
        """
        Runs the complete optimal parameter analysis pipeline.
        """
        self.load_data()
        X, _, _ = self.data_loader.get_data()
        self.cluster_analyzer = ClusterAnalyzer(X, n_clusters=3, n_dim=3, clustering_method=self.clustering_method)
        self.cluster_analyzer.normalize()
        # Run grid search first
        # self.run_grid_search()
        # Run optimal dimensions analysis
        # dim_results = self.cluster_analyzer.find_optimal_dimensions(self.results_folder)
        # optimal_components = dim_results['n_components_90']
        self.cluster_analyzer.find_optimal_clusters(self.results_folder, n_components=7)

def parse_args():
    """
    Parses command-line arguments for the SHAP Clustering Analysis.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    # Insert the default subcommand if none is provided
    if len(sys.argv) <= 1 or sys.argv[1].startswith('-'):
        sys.argv.insert(1, "comparative")
    
    parser = argparse.ArgumentParser(description="SHAP Clustering Analysis")
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory.")
    parent_parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load the model from.")
    parent_parser.add_argument("--period", type=str, default="test", help="Period (train, validation, or test).")
    parent_parser.add_argument("--cluster_method", type=str, default="KMeans", choices=["KMeans", "GMM", "Agglomerative"], help="Clustering method.")
    parent_parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, etc.).")
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands: comparative, optimal")
    
    # Comparative analysis subcommand
    parser_comp = subparsers.add_parser("comparative", parents=[parent_parser],
                                          help="Run clustering on both extreme and median data with provided parameters")
    parser_comp.add_argument("--extreme_n_clusters", type=int, default=6, help="Number of clusters for extreme conditions.")
    parser_comp.add_argument("--median_n_clusters", type=int, default=5, help="Number of clusters for median conditions.")
    parser_comp.add_argument("--extreme_n_dim", type=int, default=36, help="Number of PCA dimensions for extreme conditions.")
    parser_comp.add_argument("--median_n_dim", type=int, default=35, help="Number of PCA dimensions for median conditions.")
    
    # Optimal parameter search subcommand
    parser_opt = subparsers.add_parser("optimal", parents=[parent_parser],
                                         help="Run grid search to find optimal parameters for a single condition")
    parser_opt.add_argument("--filter_type", type=str, default="extreme", choices=["extreme", "median"])
    parser_opt.add_argument("--pca_range", type=int, nargs=2, default=[2,66], help="Min and max PCA components")
    parser_opt.add_argument("--clusters_range", type=int, nargs=2, default=[2,16], help="Min and max number of clusters")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
    
    if args.command == "comparative":
        analysis = ExtremeMedianClusterAnalysis(
            run_dir=args.run_dir,
            epoch=args.epoch,
            period=args.period,
            extreme_n_clusters=args.extreme_n_clusters,
            median_n_clusters=args.median_n_clusters,
            extreme_n_dim=args.extreme_n_dim,
            median_n_dim=args.median_n_dim,
            clustering_method=args.cluster_method
        )
        analysis.run()
    
    elif args.command == "optimal":
        optimal_search = OptimalParameterSearch(
            run_dir=args.run_dir,
            epoch=args.epoch,
            period=args.period,
            clustering_method=args.cluster_method,
            filter_type=args.filter_type,
            components_range=tuple(args.pca_range),
            clusters_range=tuple(args.clusters_range)
        )
        optimal_search.run_optimal_analysis()
