import gc
import logging
import os
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import torch.multiprocessing
from torch import Tensor, nn
import numpy as np
from tqdm import tqdm
import shap
from typing import Dict, Optional, Tuple

from explainability.cluster.hidden_clustering import HiddenStateClusterer
from model.model_analyzer import ModelAnalyzer

torch.multiprocessing.set_sharing_strategy('file_system') # TODO: fix this issue and remove it
torch.backends.cudnn.enabled = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WrappedModel(nn.Module):
    """Wrapper for the original model to handle reshaping inputs."""
    def __init__(
        self, 
        original_model: nn.Module, 
        seq_length: int, 
        num_dynamic: int, 
        num_static: int
    ) -> None:
        """Initialize wrapped model."""
        super().__init__()
        self.original_model = original_model
        self.seq_length = seq_length
        self.num_dynamic = num_dynamic
        self.num_static = num_static
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Reshape inputs and forward through original model.
        
        Args:
            inputs (Tensor): Shape [batch, seq_length*num_dynamic + num_static]
        
        Returns:
            Tensor: Model outputs, shape [batch, 1]
        """
        # Extract and reshape inputs
        x_d_flat = inputs[:, :self.seq_length * self.num_dynamic]
        x_s = inputs[:, self.seq_length * self.num_dynamic:]
        x_d = x_d_flat.view(-1, self.seq_length, self.num_dynamic)
        # Prepare input dictionary for original model
        model_inputs = {"x_d": x_d, "x_s": x_s}
        out_dict = self.original_model(model_inputs)
        # Return prediction (last timestep, first output dimension)
        return out_dict["y_hat"][:, -1, 0].unsqueeze(1)

class ShapClusterAnalyzer:
    """
    Analyzer class for generating SHAP values for clustered hidden states.
    """
    BACKGROUND_SIZE = 1024
    BATCH_SIZE = 256
    CLUSTER_SAMPLE_FRACTION = 0.01
    
    def __init__(self, run_dir: Path, epoch: int, period: str = "test", n_clusters: int = 5, reuse_shap: bool = False) -> None:
        """
        Initialize the SHAP cluster analyzer.  

        Args:
            run_dir (Path): Path to the run directory.
            epoch (int): Epoch number for the model.
            period (str, optional): Dataset period to analyze. Defaults to "test".
            n_clusters (int, optional): Number of clusters. Defaults to 5.
            reuse_shap (bool, optional): If True, use precomputed SHAP values. Defaults to False.
        """
        self.analyzer = ModelAnalyzer(run_dir=run_dir, epoch=epoch, period=period)
        self.clusterer = HiddenStateClusterer(self.analyzer, n_clusters=n_clusters)

        self.run_dir = run_dir
        self.epoch = epoch
        self.period = period
        self.cfg = self.analyzer.cfg
        self.inputs: Optional[np.ndarray] = None
        self.seq_length = self.cfg.seq_length
        self.dynamic_features = self.cfg.dynamic_inputs
        self.static_features = self.cfg.static_attributes
        self.results_folder = run_dir / period / f"model_epoch{epoch:03d}" / "shap" / "hidden_clusters"
        os.makedirs(self.results_folder, exist_ok=True)

        # If reuse_shap is True, load precomputed SHAP values, inputs, and sample indices from the SHAP analysis folder
        self.reuse_shap = reuse_shap
        self.loaded_shap_values = None
        self.loaded_inputs = None
        self.loaded_sample_indices = None
        if self.reuse_shap:
            self._load_precomputed_shap()

    def _load_precomputed_shap(self) -> None:
        """
        Load the precomputed SHAP values, inputs, and sample indices from the SHAP analysis folder.
        Assumes that shap_analysis.py saved these files in:
        run_dir/<period>/model_epoch<epoch>/shap/
        """
        base_folder = self.run_dir / self.period / f"model_epoch{self.epoch:03d}" / "shap"
        shap_values_path = os.path.join(base_folder, "shap_values.npy")
        inputs_path = os.path.join(base_folder, "inputs.npz")
        sample_indices_path = os.path.join(base_folder, "sample_indices.npy")
        if os.path.exists(shap_values_path) and os.path.exists(inputs_path) and os.path.exists(sample_indices_path):
            logging.info("Loading precomputed SHAP values and metadata...")
            self.loaded_shap_values = np.load(shap_values_path, mmap_mode='r')
            self.loaded_inputs = np.load(inputs_path, mmap_mode='r')
            self.loaded_sample_indices = np.load(sample_indices_path, mmap_mode='r')
        else:
            raise FileNotFoundError(f"Precomputed SHAP files or sample indices not found in {base_folder}. Please run SHAP analysis first, or set reuse_shap=False.")
    
    def _get_inputs(self) -> Dict[str, np.ndarray]:
        """
        Extract the dynamic and static input features from the dataset.
        
        Returns:
            dict: {
                "x_d": np.ndarray of shape [num_samples, T, D],
                "x_s": np.ndarray of shape [num_samples, S]
            }
            where:
                - num_samples is the total number of samples
                - T is the number of timesteps (sequence length)
                - D is the number of dynamic features
                - S is the number of static features 
        """
        self.inputs = self.analyzer.get_inputs()
        return self.inputs
    
    def _wrap_model(self) -> nn.Module:
        """
        Create a wrapper so we can pass a single input tensor of shape
        [batch, seq_length * num_dynamic + num_static] directly to the model
        
        Returns:
            nn.Module: A wrapped model that reshapes the inputs and calls the original model
        """
        device = self.cfg.device
        return WrappedModel(
            self.analyzer.model,
            seq_length=self.cfg.seq_length,
            num_dynamic=len(self.dynamic_features),
            num_static=len(self.static_features)
        ).to(device).eval()
    
    def run_shap_for_cluster(
        self, 
        cluster_id: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run SHAP analysis on the input samples that belong to the specified cluster.
        
        Args:
            cluster_id (int): The cluster for which to run SHAP analysis
                
        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]:
                shap_values (np.ndarray): SHAP values for the cluster.
                inputs (dict): A dictionary containing:
                    - "x_d" (np.ndarray): Dynamic input values, shape [N, T, D].
                    - "x_s" (np.ndarray): Static input values, shape [N, S].
        """
        # Get indices of samples belonging to the given cluster
        cluster_indices = np.where(self.clusterer.cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            raise ValueError(f"No samples found for cluster {cluster_id}.")

        # Randomly sample a fraction of cluster samples (at least one sample)
        cluster_indices = np.random.choice(cluster_indices, 
                                           size=max(1, int(self.CLUSTER_SAMPLE_FRACTION * len(cluster_indices))), 
                                           replace=False)

        x_d = self.inputs["x_d"][cluster_indices] # shape: [N, T, D]
        x_s = self.inputs["x_s"][cluster_indices] # shape: [N, S]

        # Flatten x_d to [N, T*D] and concatenate with x_s to get [N, T*D + S]
        combined_inputs = np.hstack([
            x_d.reshape(len(x_d), -1),
            x_s
        ])

        inputs_tensor = torch.tensor(combined_inputs, dtype=torch.float32).to(self.cfg.device)
        
        background_indices = np.random.choice(
            range(len(inputs_tensor)),
            size=min(self.BACKGROUND_SIZE, len(inputs_tensor)),
            replace=False,
        )
        background_tensor = inputs_tensor[background_indices].clone().detach().requires_grad_(True)
        
        wrapped_model = self._wrap_model()
        explainer = shap.GradientExplainer(
            wrapped_model, 
            background_tensor, 
            batch_size=self.BATCH_SIZE
        )
        
        # Calculate SHAP values in batches
        shap_values_batches = []
        with tqdm(total=len(inputs_tensor), desc=f"Calculating SHAP values for cluster {cluster_id}") as pbar:
            for i in range(0, len(inputs_tensor), self.BATCH_SIZE):
                batch = inputs_tensor[i:i + self.BATCH_SIZE].clone().detach().requires_grad_(True)
                batch_values = explainer.shap_values(batch)
                
                if isinstance(batch_values, list):
                    batch_values = batch_values[0]
                    
                if torch.is_tensor(batch_values):
                    batch_values = batch_values.cpu().numpy()
                    
                shap_values_batches.append(batch_values)
                pbar.update(len(batch))
                
                if i % (self.BATCH_SIZE * 10) == 0:
                    torch.cuda.empty_cache()
        
        shap_values = np.concatenate(shap_values_batches, axis=0)
        print((f"SHAP values shape: {shap_values.shape}"))

        inputs = {
            "x_d": x_d,   # original dynamic inputs: [N, T, D]
            "x_s": x_s    # original static inputs: [N, S]
        }

        return shap_values, inputs
    
    def _get_cluster_shap_values_from_precomputed(self, cluster_id: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Retrieve SHAP values and corresponding inputs for a given cluster using the precomputed values.
    
        Args:
            cluster_id (int): Cluster ID.
    
        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]:
                shap_values (np.ndarray): SHAP values for the cluster.
                inputs (dict): A dictionary containing:
                    - "x_d" (np.ndarray): Dynamic input values, shape [N, T, D].
                    - "x_s" (np.ndarray): Static input values, shape [N, S].
        """
        # Obtain the global indices of samples belonging to the given cluster
        global_cluster_indices = np.where(self.clusterer.cluster_labels == cluster_id)[0]
        if len(global_cluster_indices) == 0:
            raise ValueError(f"No samples found for cluster {cluster_id}.")

        # Use the loaded sample indices (global indices for which SHAP was computed) to identify the intersection
        mask = np.isin(self.loaded_sample_indices, global_cluster_indices, assume_unique=True)
        if np.sum(mask) == 0:
            raise ValueError(f"No precomputed SHAP values available for cluster {cluster_id}.")
        
        shap_values = self.loaded_shap_values[mask]
        x_d = self.loaded_inputs["x_d"][mask]
        x_s = self.loaded_inputs["x_s"][mask]
        logging.info(f"Cluster {cluster_id}: {shap_values.shape[0]} samples with precomputed SHAP values.")
        return shap_values, {"x_d": x_d, "x_s": x_s}

    def _plot_shap_summary(self, combined_shap: np.ndarray, combined_inputs: np.ndarray, 
                           feature_names: list, cluster_id: int) -> None:
        """Generates and saves a SHAP summary plot for the given cluster.

        This function creates a summary plot to visualize the SHAP values for dynamic and
        static features. The dynamic SHAP values are assumed to have been summed over the time
        dimension prior to plotting.

        Args:
            combined_shap (np.ndarray): Combined SHAP values for dynamic and static features,
                with shape [N, D + S].
            combined_inputs (np.ndarray): Combined input values for dynamic and static features,
                with shape [N, D + S].
            feature_names (list): List of feature names corresponding to the columns in the arrays,
                with shape [D + S].
            cluster_id (int): Identifier for the cluster.
        """
        plt.figure(figsize=(10, 12))
        shap.summary_plot(
            combined_shap,
            combined_inputs,
            feature_names=feature_names,
            max_display=len(feature_names),
            show=False
        )
        plt.title(f"SHAP Summary for Cluster {cluster_id}")
        summary_path = self.results_folder / f"shap_summary_plot_cluster_{cluster_id}.png"
        plt.savefig(summary_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Summary plot saved to {summary_path}")

    def _plot_feature_importance_bar(self, dynamic_shap: np.ndarray, static_shap: np.ndarray, 
                                     feature_names: list, cluster_id: int) -> None:
        """Generates and saves a horizontal bar plot of median feature importance for the given cluster.

        This function calculates both the signed median SHAP value and the median absolute SHAP value 
        across samples for each feature. For dynamic features, the SHAP values are assumed to have been 
        summed over the time dimension prior to aggregation.

        Args:
            dynamic_shap_sum (np.ndarray): Dynamic SHAP values summed over time, with shape [N, D].
            static_shap (np.ndarray): Static SHAP values, with shape [N, S].
            feature_names (list): List of feature names, with dynamic features followed by static features, with shape [D + S].
            cluster_id (int): Identifier for the cluster.
        """
        # Compute signed medians for dynamic and static features
        dynamic_median = np.median(dynamic_shap, axis=0)
        static_median = np.median(static_shap, axis=0)
        median_profile = np.concatenate([dynamic_median, static_median], axis=0)
        
        # Compute median absolute values for dynamic and static features
        dynamic_median_abs = np.median(np.abs(dynamic_shap), axis=0)
        static_median_abs = np.median(np.abs(static_shap), axis=0)
        median_profile_abs = np.concatenate([dynamic_median_abs, static_median_abs], axis=0)
        
        num_features = len(feature_names)
        y = np.arange(num_features)
        bar_height = 0.35

        plt.figure(figsize=(10, 8))
        plt.barh(y - bar_height/2, median_profile, height=bar_height, color='skyblue', label='Signed Median')
        plt.barh(y + bar_height/2, median_profile_abs, height=bar_height, color='salmon', alpha=0.3, label='Absolute Median')
        plt.yticks(y, feature_names, fontsize=6)
        plt.gca().invert_yaxis()
        plt.xlabel("Aggregated SHAP Value")
        plt.title(f"Feature Importance for Cluster {cluster_id}")
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        barplot_path = self.results_folder / f"bar_plot_cluster_{cluster_id}.png"
        plt.savefig(barplot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Bar plot saved to {barplot_path}")

    def _generate_cluster_shap_plots(self, shap_values: np.ndarray, inputs: Dict[str, np.ndarray], 
                                     cluster_id: int) -> None:
        """Processes raw SHAP values and generates both a summary and bar plot for a cluster.

        This function separates the dynamic and static SHAP values from the input, reshapes the
        dynamic part to sum over the time dimension, and then creates:
        - A SHAP summary plot using the combined dynamic and static SHAP values.
        - A bar plot showing the median SHAP value for each feature across samples.

        Args:
            shap_values (np.ndarray): Raw SHAP values for all features, with shape
                [N, T * D + S] = [N, seq_length * n_dynamic + n_static].
            inputs (Dict[str, np.ndarray]): Dictionary containing:
                - "x_d": Dynamic inputs, shape [N, T, D].
                - "x_s": Static inputs, shape [N, S].
            cluster_id (int): Identifier for the cluster.
        """
        shap_values = shap_values.squeeze(-1)
        x_d = inputs["x_d"]  # shape: [N, T, D]
        x_s = inputs["x_s"]  # shape: [N, S]
        n_samples = shap_values.shape[0]
        n_dynamic = len(self.dynamic_features)

        # Split SHAP values into dynamic and static parts
        dynamic_shap = shap_values[:, :self.seq_length * n_dynamic].reshape(n_samples, self.seq_length, n_dynamic)
        static_shap = shap_values[:, self.seq_length * n_dynamic:]

        # Sum dynamic features across time
        dynamic_shap = dynamic_shap.sum(axis=1)
        x_d = x_d.sum(axis=1)

        combined_shap = np.concatenate([dynamic_shap, static_shap], axis=1) # shape: [N, D + S]
        combined_inputs = np.concatenate([x_d, x_s], axis=1) # shape: [N, D + S]
        feature_names = self.dynamic_features + self.static_features

        # Compute global clipping thresholds:
        # - Global lower_clip is the minimum of the 0.01st percentiles across all features
        # - Global upper_clip is the maximum of the 99.99th percentiles across all features
        # This removes the most extreme 0.01% from each tail
        lower_clip = min([np.percentile(combined_shap[:, i], 0.01) for i in range(combined_shap.shape[1])])
        upper_clip = max([np.percentile(combined_shap[:, i], 99.99) for i in range(combined_shap.shape[1])])
        combined_shap_clipped = np.clip(combined_shap, lower_clip, upper_clip)

        self._plot_shap_summary(combined_shap_clipped, combined_inputs, feature_names, cluster_id)
        self._plot_feature_importance_bar(dynamic_shap, static_shap, feature_names, cluster_id)
    
    def run_all_clusters(self) -> Dict[int, np.ndarray]:
        """
        Run SHAP analysis for all clusters.
                
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping cluster IDs to SHAP values.
        """
        if self.clusterer.cluster_labels is None:
            self.clusterer.perform_clustering()
        
        if self.inputs is None and not self.reuse_shap:
            self._get_inputs()
        
        results = {}        
        for cluster_id in range(self.clusterer.n_clusters):
            try:
                if self.reuse_shap:
                    # Get precomputed SHAP values for the cluster
                    shap_values, inputs = self._get_cluster_shap_values_from_precomputed(cluster_id)
                else:
                    # Compute SHAP values on the fly for this cluster
                    shap_values, inputs = self.run_shap_for_cluster(cluster_id=cluster_id)
                self._generate_cluster_shap_plots(shap_values, inputs, cluster_id)
                results[cluster_id] = shap_values

                del shap_values, inputs
                gc.collect()
                torch.cuda.empty_cache()

            except ValueError as e:
                print(f"Error processing cluster {cluster_id}: {str(e)}")
        
        return results


if __name__ == "__main__":
    run_dir = Path("/sci/labs/efratmorin/eli.levinkopf/batch_runs/runs/train_lstm_rs_22_1503_194719/")
    shap_cluster_analyzer = ShapClusterAnalyzer(run_dir=run_dir, epoch=25, n_clusters=10, reuse_shap=False)
    # shap_cluster_analyzer.clusterer.evaluate(output_path=shap_cluster_analyzer.results_folder)
    results = shap_cluster_analyzer.run_all_clusters()
