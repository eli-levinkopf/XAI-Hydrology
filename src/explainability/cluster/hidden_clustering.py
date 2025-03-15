import os
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap
from typing import Optional, Tuple

from model.model_analyzer import ModelAnalyzer


class HiddenStateClusterer:
    """
    Class for clustering hidden states from neural network models.
    """
    
    def __init__(
        self, 
        analyzer: ModelAnalyzer,
        n_clusters: int = 5,
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

        hidden_np = self.hidden_states.numpy()
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init=10 
        )
        self.cluster_labels = kmeans.fit_predict(hidden_np)
        return self.cluster_labels
    
    def reduce_dimensionality(self, n_components: int = 2) -> np.ndarray:
        """
        Reduce the dimensionality of hidden states for visualization using UMAP.
        
        Args:
            n_components (int, optional): Number of components for UMAP. Defaults to 2
            
        Returns:
            np.ndarray: UMAP-reduced embedding
        """
        hidden_np = self.hidden_states.numpy()
        reducer = umap.UMAP(
            n_components=n_components, 
            random_state=self.random_state
        )
        self.embedding = reducer.fit_transform(hidden_np)
        return self.embedding
    
    def plot_clusters(self, save_path: Optional[str] = None) -> None:
        """
        Plot the clusters in a 2D space using UMAP-reduced hidden states.
        
        Args:
            save_path (Optional[str], optional): Path to save the plot. 
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
        
        if save_path is None:
            save_path = os.path.join(str(self.run_dir), "hidden_state_clusters.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

