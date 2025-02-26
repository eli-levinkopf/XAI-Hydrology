import os
import pickle
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class SHAPClusterAnalysis:
    def __init__(self, run_dir, epoch, period, filter_type, n_clusters, random_state=42):
        self.run_dir = run_dir
        self.period = period
        self.epoch = epoch
        self.filter_type = filter_type.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.results_folder = self._setup_results_folder()
        self.aggregated = None
        self.feature_names = None
        self.basin_ids = None
        self.X_signed = None
        self.X_absolute = None
        self.X_scaled = None
        self.X_reduced = None
        self.clusters = None

    def load_data(self):
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

    def normalize_data(self):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X_signed)

    def reduce_dimension(self, n_components=2):
        pca = PCA(n_components=n_components)
        self.X_reduced = pca.fit_transform(self.X_scaled)

    def perform_clustering(self):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100, max_iter=1000, random_state=self.random_state)
        self.clusters = kmeans.fit_predict(self.X_reduced)

    def plot_clusters(self):
        if self.X_reduced is None or self.clusters is None:
            logging.error("Data has not been reduced or clustered yet.")
            return

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.X_reduced[:, 0], self.X_reduced[:, 1], c=self.clusters, cmap="viridis", s=50, alpha=0.8)
        plt.title(f"Clusters of Basins Based on Aggregated SHAP Values ({self.filter_type})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plot_path = os.path.join(self.results_folder, f"{self.n_clusters}_clusters_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def group_clusters(self):
        clusters_dict = {}
        for bid, cluster in zip(self.basin_ids, self.clusters):
            clusters_dict.setdefault(cluster, []).append(bid)
        return clusters_dict

    def plot_cluster_bar(self, cluster, signed_profile, absolute_profile):
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
        plt.legend()
        plt.tight_layout()

        file_path = os.path.join(self.results_folder, f"cluster_{cluster}_bar_plot.png")
        plt.savefig(file_path, dpi=300)
        plt.close()

    def plot_cluster_radar(self, cluster, profile):
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

    def save_combined_feature_profiles(self, cluster_profiles_signed, cluster_profiles_absolute):
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

    def analyze_clusters(self):
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

    def run(self):
        self.load_data()
        self.normalize_data()
        self.reduce_dimension(n_components=2)
        self.perform_clustering()
        self.analyze_clusters()


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster basins based on aggregated SHAP values with normalization and PCA.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load the model from.")
    parser.add_argument("--period", type=str, default="test", help="Period (train, validation, or test).")
    parser.add_argument("--filter_type", type=str, default="extreme", help="Filter type: 'extreme' or 'median'.")
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
        filter_type=args.filter_type,
        n_clusters=args.n_clusters,
    )
    shap_cluster_analysis.run()
