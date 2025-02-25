import os
import pickle
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class SHAPClusterAnalysis:
    def __init__(self, run_dir, epoch, period, n_clusters, random_state=42):
        self.run_dir = run_dir
        self.period = period
        self.epoch = epoch
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.file_path = os.path.join(
            self.run_dir,
            self.period,
            f"model_epoch{self.epoch:03d}",
            "shap",
            "aggregated_shap_extreme.p"
        )
        self.aggregated = None
        self.feature_names = None
        self.basin_ids = None
        self.X = None
        self.X_scaled = None
        self.X_reduced = None
        self.clusters = None

    def load_data(self):
        logging.info(f"Loading aggregated SHAP data from: {self.file_path}")
        if not os.path.exists(self.file_path):
            logging.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, "rb") as f:
            data = pickle.load(f)

        self.aggregated = data.get("aggregated", {})
        self.feature_names = data.get("feature_names", [])
        self.basin_ids = list(self.aggregated.keys())
        self.X = np.array([self.aggregated[bid] for bid in self.basin_ids])
        logging.info(f"Loaded SHAP vectors for {self.X.shape[0]} basins with {self.X.shape[1]} features each.")

    def normalize_data(self):
        logging.info("Normalizing data (zero mean, unit variance)...")
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        logging.info("Data normalization complete.")

    def reduce_dimension(self, n_components=2):
        logging.info(f"Reducing dimensionality to {n_components} dimensions using PCA...")
        pca = PCA(n_components=n_components)
        self.X_reduced = pca.fit_transform(self.X_scaled)
        logging.info("Dimensionality reduction complete.")
        logging.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    def perform_clustering(self):
        logging.info(f"Clustering data using K-means with {self.n_clusters} clusters...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.clusters = kmeans.fit_predict(self.X_reduced)
        logging.info("K-means clustering complete.")

    def plot_clusters(self):
        if self.X_reduced is None or self.clusters is None:
            logging.error("Data has not been reduced or clustered yet.")
            return

        logging.info("Plotting clusters...")
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.X_reduced[:, 0], self.X_reduced[:, 1], c=self.clusters, cmap="viridis", s=50, alpha=0.8)
        plt.title("Clusters of Basins Based on Aggregated SHAP Values (PCA Reduced)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster")
        # Optionally annotate a few points for reference (here we annotate a maximum of 10 points)
        for i, bid in enumerate(self.basin_ids[:10]):
            plt.annotate(bid, (self.X_reduced[i, 0], self.X_reduced[i, 1]), fontsize=8, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def group_clusters(self):
        logging.info("Grouping basin IDs by cluster...")
        clusters_dict = {}
        for bid, cluster in zip(self.basin_ids, self.clusters):
            clusters_dict.setdefault(cluster, []).append(bid)
        return clusters_dict

    def run(self):
        self.load_data()
        self.normalize_data()
        self.reduce_dimension(n_components=2)
        self.perform_clustering()
        self.plot_clusters()
        clusters_dict = self.group_clusters()
        logging.info("Basins grouped by cluster:")
        for cluster, bids in clusters_dict.items():
            logging.info(f"Cluster {cluster} (n={len(bids)}): {bids}")
        return clusters_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster basins based on aggregated SHAP values with normalization and PCA."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load the model from.")
    parser.add_argument("--period", type=str, default="test", help="Period (train, validation, or test).")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters for K-means.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, etc.).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    shap_cluster_analysis = SHAPClusterAnalysis(
        run_dir=args.run_dir,
        epoch=args.epoch,
        period=args.period,
        n_clusters=args.n_clusters,
    )
    clusters = shap_cluster_analysis.run()
