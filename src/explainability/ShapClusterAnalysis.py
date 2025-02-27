import os
import pickle
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd

from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


class SHAPClusterAnalysis:
    """
    Perform clustering analysis on SHAP values using PCA for dimensionality reduction
    and K-Means for clustering.
    """
    def __init__(self, run_dir: str, epoch: int, period: str, filter_type: str, n_clusters: int, n_dim: int, random_state: int = 42, find_optimal_params: bool = False) -> None:
        """
        Initialize SHAPClusterAnalysis.
        
        Args:
            run_dir (str): Path to the run directory.
            epoch (int): The model epoch number.
            period (str): The data period (train, validation, test).
            filter_type (str): Type of filter ('extreme' or 'median').
            n_clusters (int): Number of clusters for KMeans.
            n_dim (int): Number of PCA dimensions to reduce to.
            random_state (int, optional): Random seed for clustering. Defaults to 42.
            find_optimal_params (bool, optional): Whether to find optimal parameters. Defaults to False.
        """
        self.run_dir = run_dir
        self.period = period
        self.epoch = epoch
        self.filter_type = filter_type.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.find_optimal_params = find_optimal_params
        self.results_folder = self._setup_results_folder()
        self.aggregated = None
        self.feature_names = None
        self.basin_ids = None
        self.X_signed = None
        self.X_absolute = None
        self.X_scaled = None
        self.clusters = None
        self.n_dim = n_dim

    def load_data(self) -> None:
        """
        Loads the aggregated SHAP values from a pickle file.
        Raises FileNotFoundError if the file is missing.
        """
        file_path = os.path.join(
            self.run_dir,
            self.period,
            f"model_epoch{self.epoch:03d}",
            "shap",
            f"aggregated_shap_{self.filter_type}.p"
        )
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self.aggregated = data.get("aggregated", {})
        self.feature_names = data.get("feature_names", [])
        self.basin_ids = list(self.aggregated.keys())
        # Extract signed and absolute aggregated vectors for each basin
        self.X_signed = np.array([self.aggregated[bid]["signed"] for bid in self.basin_ids])
        self.X_absolute = np.array([self.aggregated[bid]["absolute"] for bid in self.basin_ids])

    def _setup_results_folder(self) -> str:
        """
        Set up the folder structure for saving results under:
        run_dir/<period>/model_epoch<epoch>/shap/basins_clustered/<filter_type>

        Returns:
            str: Path to the results folder.
        """
        results_folder = os.path.join(
            self.run_dir,
            self.period,
            f"model_epoch{self.epoch:03d}",
            "shap",
            "basins_clustered",
            self.filter_type
        )
        os.makedirs(results_folder, exist_ok=True)
        return results_folder

    def _normalize_data(self) -> None:
        """
        Normalizes the signed SHAP values using StandardScaler.
        """
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X_signed)

    def reduce_dimension(self, n_components: int = 2) -> np.ndarray:
        """
        Reduces dimensionality of the SHAP values using PCA.

        Args:
            n_components (int, optional): Number of components for PCA. Defaults to 2.

        Returns:
            np.ndarray: Transformed data after PCA.
        """
        pca = PCA(n_components=n_components)
        return pca.fit_transform(self.X_scaled)

    def perform_clustering(self) -> None:
        """
        Performs KMeans clustering on the PCA-reduced SHAP data.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100, max_iter=1000, random_state=self.random_state)
        X_reduced = self.reduce_dimension(n_components=self.n_dim)
        self.clusters = kmeans.fit_predict(X_reduced)

    def plot_clusters(self) -> None:
        """
        Generates a scatter plot of clusters using the first two PCA components.
        """
        plt.figure(figsize=(8, 6))
        X_reduced = self.reduce_dimension()
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=self.clusters, cmap="viridis", s=50, alpha=0.8)
        plt.title(f"Clusters of Basins (k={self.n_clusters}) Based on Aggregated SHAP Values ({self.filter_type})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plot_path = os.path.join(self.results_folder, f"{self.n_clusters}_clusters_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def group_clusters(self) -> Dict[int, List[str]]:
        """
        Groups basins into their respective clusters.

        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster numbers to basin IDs.
        """
        clusters_dict = {}
        for bid, cluster in zip(self.basin_ids, self.clusters):
            clusters_dict.setdefault(cluster, []).append(bid)
        return clusters_dict

    def plot_cluster_bar(self, cluster: int, signed_profile: np.ndarray, absolute_profile: np.ndarray) -> None:
        """
        Generates a bar plot of the aggregated feature profile for a cluster.

        Args:
            cluster (int): Number of clusters.
            signed_profile (np.ndarray): Signed aggregated profile.
            absolute_profile (np.ndarray): Absolute aggregated profile.
        """
        num_features = len(signed_profile)
        y = np.arange(num_features)
        bar_height = 0.35

        plt.figure(figsize=(10, 8))
        plt.barh(y - bar_height/2, signed_profile, height=bar_height, color='skyblue', label='Signed Median')
        plt.barh(y + bar_height/2, absolute_profile, height=bar_height, color='salmon', alpha=0.3, label='Absolute Median')
        plt.yticks(y, self.feature_names, fontsize=6)
        plt.gca().invert_yaxis()
        plt.xlabel("Aggregated SHAP Value", fontsize=10)
        plt.title(f"Cluster {cluster} Aggregated Feature Profile", fontsize=12)
        plt.legend(loc='lower right')
        plt.tight_layout()

        file_path = os.path.join(self.results_folder, f"cluster_{cluster}_bar_plot.png")
        plt.savefig(file_path, dpi=300)
        plt.close()

    def plot_cluster_radar(self, cluster: int, profile: np.ndarray) -> None:
        """
        Generates a radar plot of the aggregated feature profile for a cluster.

        Args:
            cluster (int): Number of clusters.
            profile (np.ndarray): Aggregated profile (signed).
        """
        num_vars = len(profile)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        values = profile.tolist()
        values += values[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], self.feature_names, fontsize=6)
        ax.plot(angles, values, color='b', linewidth=2)
        ax.fill(angles, values, color='b', alpha=0.25)
        plt.title(f"Cluster {cluster} Aggregated Feature Profile", y=1.08)
        plt.tight_layout()
        file_path = os.path.join(self.results_folder, f"cluster_{cluster}_radar_plot.png")
        plt.savefig(file_path, dpi=300)
        plt.close()

    def save_combined_feature_profiles(self, cluster_profiles_signed: Dict[int, np.ndarray], cluster_profiles_absolute: Dict[int, np.ndarray]) -> None:
        """
        Save CSV file containing the feature profiles for each cluster. 
        Each row corresponds to a feature, and columns are the signed and absolute values for each cluster.

        Args:
            cluster_profiles_signed (Dict[int, np.ndarray]): A dictionary where keys are cluster 
            indices and values are numpy arrays representing the signed feature profiles 
            for each cluster.
            cluster_profiles_absolute (Dict[int, np.ndarray]): A dictionary where keys are cluster 
            indices and values are numpy arrays representing the absolute feature profiles 
            for each cluster.
        """
        clusters_sorted = sorted(cluster_profiles_signed.keys())
        csv_file = os.path.join(self.results_folder, "cluster_feature_profiles.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Feature"] + [f"Cluster_{cluster}_signed" for cluster in clusters_sorted] + [f"Cluster_{cluster}_absolute" for cluster in clusters_sorted]
            writer.writerow(header)
            num_features = len(self.feature_names)
            for i in range(num_features):
                row = [self.feature_names[i]]
                for cluster in clusters_sorted:
                    row.append(cluster_profiles_signed[cluster][i])
                for cluster in clusters_sorted:
                    row.append(cluster_profiles_absolute[cluster][i])
                writer.writerow(row)

    def analyze_clusters(self) -> Dict[int, List[str]]:
        """
        Performs cluster analysis, generates plots, and saves results.

        Returns:
            Dict[int, List[str]]: Clusters mapped to basin IDs.
        """
        logging.info("Analyzing clusters...")
        self.plot_clusters()
        clusters_dict = self.group_clusters()
        cluster_profiles_signed = {}
        cluster_profiles_absolute = {}
        for cluster, basin_ids in clusters_dict.items():
            indices = [self.basin_ids.index(bid) for bid in basin_ids]
            # Compute median across basins for signed and absolute values
            aggregated_profile_signed = np.median(self.X_signed[indices], axis=0)
            aggregated_profile_absolute = np.median(self.X_absolute[indices], axis=0)
            cluster_profiles_signed[cluster] = aggregated_profile_signed
            cluster_profiles_absolute[cluster] = aggregated_profile_absolute
            self.plot_cluster_bar(cluster, aggregated_profile_signed, aggregated_profile_absolute)
            self.plot_cluster_radar(cluster, aggregated_profile_signed)
        self.save_combined_feature_profiles(cluster_profiles_signed, cluster_profiles_absolute)
        logging.info(f"Cluster analysis completed. The results are saved in: {self.results_folder}")
        return clusters_dict

    def find_optimal_dimensions(self) -> None:
        """
        Performs Principal Component Analysis (PCA) to find optimal dimensionality reduction parameters
        to determine optimal number of components for different variance thresholds (80%, 90%, 95%).
        Returns:
            dict: Dictionary containing:
                - explained_variance_ratio (array): Explained variance ratio for each component
                - cumulative_explained_variance (array): Cumulative sum of explained variance ratios
                - n_components_80/90/95 (int): Number of components needed to explain 80%, 90%, 95% variance
        """
        # Perform PCA with maximum components (up to the number of features)
        pca = PCA(n_components=min(len(self.feature_names), self.X_scaled.shape[0]))
        pca.fit(self.X_scaled)
        
        # Calculate cumulative explained variance
        cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cum_explained_variance) + 1), cum_explained_variance, 'bo-')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        
        # Add horizontal lines at common thresholds
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% Explained Variance')
        plt.axhline(y=0.9, color='g', linestyle='--', label='90% Explained Variance')
        plt.axhline(y=0.95, color='orange', linestyle='--', label='95% Explained Variance')
        
        # Find components needed for different variance thresholds
        n_components_80 = np.argmax(cum_explained_variance >= 0.8) + 1
        n_components_90 = np.argmax(cum_explained_variance >= 0.9) + 1
        n_components_95 = np.argmax(cum_explained_variance >= 0.95) + 1
        
        plt.axvline(x=n_components_80, color='r', alpha=0.3)
        plt.axvline(x=n_components_90, color='g', alpha=0.3)
        plt.axvline(x=n_components_95, color='orange', alpha=0.3)
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_folder, "pca_explained_variance.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_explained_variance': cum_explained_variance,
            'n_components_80': n_components_80,
            'n_components_90': n_components_90,
            'n_components_95': n_components_95
        }

    def find_optimal_clusters(self, n_components: int = 2, max_clusters: int = 10) -> Dict[str, List[float]]:
        """
        Determine the optimal number of clusters using multiple methods
        
        Args:
            n_components: Number of PCA components to use (determined from find_optimal_dimensions)
            max_clusters: Maximum number of clusters to test

        Returns:
            dict: Dictionary containing the following metrics for each k value:
                - n_values: List of k values
                - inertia: Sum of squared distances to closest cluster center
                - silhouette: Silhouette score (higher is better)
                - calinski_harabasz: Calinski-Harabasz Index (higher is better)
                - davies_bouldin: Davies-Bouldin Index (lower is better)
        """
        X_reduced = self.reduce_dimension(n_components=n_components)
        
        # Store results
        results = {
            'n_values': list(range(2, max_clusters + 1)),
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        # Calculate metrics for different k values
        for n in results['n_values']:
            # KMeans clustering
            kmeans = KMeans(n_clusters=n, n_init=100, max_iter=1000, random_state=self.random_state)
            clusters = kmeans.fit_predict(X_reduced)
            
            # Store inertia (sum of squared distances)
            results['inertia'].append(kmeans.inertia_)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_reduced, clusters)
            results['silhouette'].append(silhouette_avg)
            
            # Calculate Calinski-Harabasz Index (higher is better)
            calinski = calinski_harabasz_score(X_reduced, clusters)
            results['calinski_harabasz'].append(calinski)
            
            # Calculate Davies-Bouldin Index (lower is better)
            davies = davies_bouldin_score(X_reduced, clusters)
            results['davies_bouldin'].append(davies)
        
        # Plot results
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Elbow method plot
        axs[0, 0].plot(results['n_values'], results['inertia'], 'bo-')
        axs[0, 0].set_xlabel('Number of clusters (n)')
        axs[0, 0].set_ylabel('Inertia (WCSS)')
        axs[0, 0].set_title('Elbow Method')
        
        # Silhouette score plot
        axs[0, 1].plot(results['n_values'], results['silhouette'], 'ro-')
        axs[0, 1].set_xlabel('Number of clusters (n)')
        axs[0, 1].set_ylabel('Silhouette Score')
        axs[0, 1].set_title('Silhouette Method (higher is better)')
        
        # Calinski-Harabasz plot
        axs[1, 0].plot(results['n_values'], results['calinski_harabasz'], 'go-')
        axs[1, 0].set_xlabel('Number of clusters (n)')
        axs[1, 0].set_ylabel('Calinski-Harabasz Score')
        axs[1, 0].set_title('Calinski-Harabasz Index (higher is better)')
        
        # Davies-Bouldin plot
        axs[1, 1].plot(results['n_values'], results['davies_bouldin'], 'mo-')
        axs[1, 1].set_xlabel('Number of clusters (n)')
        axs[1, 1].set_ylabel('Davies-Bouldin Score')
        axs[1, 1].set_title('Davies-Bouldin Index (lower is better)')
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_folder, f"optimal_k_analysis_pca{n_components}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return results

    def grid_search_best_parameters(self, components_range: np.ndarray = np.arange(2, 66), 
                                    clusters_range: np.ndarray = np.arange(2, 16)) -> None:
        """
        Perform a grid search over different combinations of PCA components and cluster numbers
        to determine the best combination based on silhouette score.

        Args:
            components_range (np.ndarray): Range of PCA components to test.
            clusters_range (np.ndarray): Range of cluster numbers (k-values) to test in KMeans.
        
        Saves:
            - A CSV file with all the results (`grid_search_results.csv`).
            - A heatmap visualization of the silhouette scores (`grid_search_heatmap.png`).
        """
        results = []
        
        for n_components in components_range:
            X_reduced = self.reduce_dimension(n_components=n_components)
            
            for k in clusters_range:
                logging.info(f"Running KMeans with {k} clusters and {n_components} components...")
                kmeans = KMeans(n_clusters=k, n_init=100, max_iter=1000, random_state=self.random_state)
                clusters = kmeans.fit_predict(X_reduced)
                
                silhouette_avg = silhouette_score(X_reduced, clusters)
                calinski = calinski_harabasz_score(X_reduced, clusters)
                davies = davies_bouldin_score(X_reduced, clusters)
                
                result = {
                    'n_components': n_components,
                    'n_clusters': k,
                    'silhouette': silhouette_avg,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies
                }
                results.append(result)
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        csv_path = os.path.join(self.results_folder, "grid_search_results.csv")
        results_df.to_csv(csv_path, index=False)
        
        # Create heatmap of silhouette scores
        pivot_data = results_df.pivot(index='n_components', columns='n_clusters', values='silhouette')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Silhouette Score for Different PCA Components and Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Number of PCA Components')
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_folder, "grid_search_heatmap.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def run(self) -> None:
        """
        Run the SHAP cluster analysis.
        """
        self.load_data()
        self._normalize_data()
        
        if self.find_optimal_params:
            # self.grid_search_best_parameters()

            # Find optimal dimensions first
            dim_results = self.find_optimal_dimensions()
            optimal_components = dim_results['n_components_90']
            
            # Find optimal number of clusters using the chosen dimensions
            self.find_optimal_clusters(n_components=5)
            
            # At this point, you would examine the plots and decide on the optimal k
        else:
            self.perform_clustering()
            self.analyze_clusters()


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster basins based on aggregated SHAP values with normalization and PCA.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load the model from.")
    parser.add_argument("--period", type=str, default="test", help="Period (train, validation, or test).")
    parser.add_argument("--filter_type", type=str, default="extreme", help="Filter type: 'extreme' or 'median'.")
    parser.add_argument("--n_clusters", type=int, default=6, help="Number of clusters for K-means.")
    parser.add_argument("--n_dim", type=int, default=6, help="Number of PCA dimensions to reduce to.")
    parser.add_argument("--find_optimal_params", action="store_true", help="If set, run optimal parameter search and exit.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, etc.).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    shap_cluster_analysis = SHAPClusterAnalysis(
        run_dir=args.run_dir,
        epoch=args.epoch,
        period=args.period,
        filter_type=args.filter_type,
        n_clusters=args.n_clusters,
        n_dim=args.n_dim,
        find_optimal_params=args.find_optimal_params
    )
    shap_cluster_analysis.run()

    """
    extreme: 6 clusters, 6 dimensions (PCA)
    median: 4 clusters, 4 dimensions (PCA)
    """