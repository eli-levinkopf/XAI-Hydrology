import logging
import os
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap
from typing import Any, Dict, Optional

from model.model_analyzer import ModelAnalyzer


class HiddenStateClusterer:
    """
    Class for clustering hidden states from neural network models.
    """
    
    def __init__(
        self, 
        analyzer: ModelAnalyzer,
        n_clusters: int = 10,
        random_state: int = 42
    ) -> None:
        """
        Args:
            analyzer (ModelAnalyzer): Instance to extract hidden states.
            n_clusters (int, optional): Number of clusters. Defaults to 5.
            random_state (int, optional): For reproducibility. Defaults to 42.
        """
        self.analyzer = analyzer
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.run_dir = analyzer.run_dir
        
        self.hidden_states: Optional[torch.Tensor] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.embedding: Optional[np.ndarray] = None
    
    def extract_hidden_states(self) -> torch.Tensor:
        """
        Extract the hidden states from the model.
        
        Returns:
            torch.Tensor: Hidden states tensor of shape (num_samples, hidden_size)
        """
        self.hidden_states = self.analyzer.get_hidden_states()
        return self.hidden_states
    
    def perform_clustering(self) -> np.ndarray:
        """
        Perform k-means clustering on the extracted hidden states.
        
        Returns:
            np.ndarray: Cluster labels for each hidden state
        """   
        if self.hidden_states is None:
            self.extract_hidden_states()
        hidden_np = self.reduce_dimensionality(n_components=self.pca_dim)
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init=10 
        )
        self.cluster_labels = kmeans.fit_predict(hidden_np)
        return self.cluster_labels
    
    def reduce_dimensionality(self, n_components: int = 2) -> np.ndarray:
        """
        Reduce the dimensionality of hidden states for visualization using PCA.
        
        Args:
            n_components (int, optional): Number of components for PCA. Defaults to 2
            
        Returns:
            np.ndarray: PCA-reduced embedding
        """
        hidden_np = self.hidden_states.numpy()
        pca = PCA(n_components=n_components)
        reduced_states = pca.fit_transform(hidden_np)
        return reduced_states
    
    def find_optimal_dimensions(self, output_folder: str) -> Dict[str, Any]:
        """
        Finds the optimal number of PCA components based on cumulative explained variance.

        Args:
            output_folder (str): Folder path where the explained variance plot will be saved.

        Returns:
            Dict: Dictionary with explained variance metrics.
        """
        if self.hidden_states is None:
            self.extract_hidden_states()
        hidden_np = self.hidden_states.numpy()
        pca = PCA(n_components=hidden_np.shape[1])
        pca.fit(hidden_np)

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

    def plot_clusters(self, output_path: Optional[str] = None) -> None:
        """
        Plot the clusters in a 2D space using UMAP-reduced hidden states.
        
        Args:
            output_path (Optional[str], optional): Path to save the plot. 
                If None, saves to run_dir. Defaults to None.
        """  
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            self.embedding[:, 0], 
            self.embedding[:, 1],
            c=self.cluster_labels, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.colorbar(scatter, label="Cluster")
        plt.title("Clustering of Hidden States")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        
        if output_path is None:
            output_path = os.path.join(str(self.run_dir), "hidden_state_clusters.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate(self, k_min: int = 10, k_max: int = 50, output_path: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate clustering metrics over a range of k values. Computes the inertia, silhouette score,
        Calinski-Harabasz Index, and Davies-Bouldin Index for each k, then plots all metrics.
        
        Args:
            k_min (int, optional): Minimum number of clusters (must be at least 2). Defaults to 10.
            k_max (int, optional): Maximum number of clusters. Defaults to 50.
            output_path (Optional[str], optional): Path to save the multi-panel plot.
                If None, saves to run_dir.
                
        Returns:
            Dict[int, Dict[str, Any]]: A dictionary mapping each k to its metrics.
                For each k, the metrics include:
                - 'inertia'
                - 'silhouette'
                - 'calinski_harabasz'
                - 'davies_bouldin'
        """
        if self.hidden_states is None:
            self.extract_hidden_states()
        
        hidden_np = self.hidden_states.numpy()
        ks = list(range(k_min, k_max + 1, 2))
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        # for k in tqdm(ks, desc="Evaluating Clustering Metrics"):
        for k in ks:
            logging.info(f"Starting clustering for k={k}")
            logging.info("Initializing KMeans...")
            kmeans = KMeans(
                n_clusters=k, 
                random_state=self.random_state, 
                n_init=10
                )
            logging.info("Fitting KMeans...")
            labels = kmeans.fit_predict(hidden_np)
            logging.info("Calculating inertia...")
            inertias.append(kmeans.inertia_)
            logging.info("Calculating silhouette score...")
            silhouette_val = silhouette_score(hidden_np, labels, sample_size=80000, random_state=self.random_state)
            silhouette_scores.append(silhouette_val)
            logging.info("Calculating Calinski-Harabasz score...")
            calinski_val = calinski_harabasz_score(hidden_np, labels)
            calinski_scores.append(calinski_val)
            logging.info("Calculating Davies-Bouldin score...")
            davies_val = davies_bouldin_score(hidden_np, labels)
            davies_scores.append(davies_val)
            logging.info(f"Finished clustering for k={k}")
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        axs[0, 0].plot(ks, inertias, 'bo-')
        axs[0, 0].set_title("Elbow Method")
        axs[0, 0].set_xlabel("Number of Clusters")
        axs[0, 0].set_ylabel("Inertia (WCSS)")
        
        axs[0, 1].plot(ks, silhouette_scores, 'ro-')
        axs[0, 1].set_title('Silhouette Score (higher is better)')
        axs[0, 1].set_xlabel("Number of Clusters")
        axs[0, 1].set_ylabel('Silhouette Score')
        
        axs[1, 0].plot(ks, calinski_scores, 'go-')
        axs[1, 0].set_title("Calinski-Harabasz Score (higher is better)")
        axs[1, 0].set_xlabel("Number of Clusters")
        axs[1, 0].set_ylabel("Calinski-Harabasz Score")
        
        axs[1, 1].plot(ks, davies_scores, 'mo-')
        axs[1, 1].set_ylabel('Davies-Bouldin Score')
        axs[1, 1].set_xlabel("Number of Clusters")
        axs[1, 1].set_title('Davies-Bouldin Index (lower is better)')

        
        plt.tight_layout()
        if output_path is None:
            output_path = os.path.join(str(self.run_dir), "evaluation_metrics_k_clusters.png")
        plt.savefig(os.path.join(output_path, "evaluation_metrics_k_clusters.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Assemble metrics into a dictionary
        metrics_dict = {}
        for i, k in enumerate(ks):
            metrics_dict[k] = {
                'inertia': inertias[i],
                'silhouette': silhouette_scores[i],
                'calinski_harabasz': calinski_scores[i],
                'davies_bouldin': davies_scores[i]
            }
        
        return metrics_dict
