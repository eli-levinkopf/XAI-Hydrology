from typing import Dict, Tuple, Optional, Union
import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import logging
import shap
import argparse

from ShapAnalysis import SHAPAnalysis 

BACKGROUND_SIZE = 1024
BATCH_SIZE = 256

def extreme_filter_fn(y, percentile: float = 90):
    """
    Select sequences where the target streamflow (from y) is above the given percentile.
    
    Args:
        y: Array or tensor of shape [n_sequences, 1].
        percentile: Percentile cutoff (default 90 for top 10%).
        
    Returns:
        A boolean mask of shape [n_sequences].
    """
    targets = y.squeeze(1)
    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    threshold = np.percentile(targets_np, percentile)
    return targets >= threshold

def median_filter_fn(y, lower_percentile: float = 45, upper_percentile: float = 55):
    """
    Select sequences where the target streamflow (from y) is between the lower and upper percentiles.
    
    Args:
        y: Array or tensor of shape [n_sequences, 1].
        lower_percentile: Lower percentile cutoff (default 45).
        upper_percentile: Upper percentile cutoff (default 55).
        
    Returns:
        A boolean mask of shape [n_sequences].
    """
    targets = y.squeeze(1)
    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    lower = np.percentile(targets_np, lower_percentile)
    upper = np.percentile(targets_np, upper_percentile)
    return (targets >= lower) & (targets <= upper)

class ExtremeMedianSHAPAnalysis(SHAPAnalysis):
    def __init__(self, run_dir: str, epoch: int, num_samples: int, filter_type: str = "extreme", use_embedding: bool = False) -> None:
        """
        Initialize ExtremeMedianSHAPAnalysis.

        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number to load the model.
            num_samples (int): Total number of samples for SHAP analysis.
            filter_type (str): Either "extreme" (default) or "median". Determines which filter to use.
            use_embedding (bool): If True, run SHAP on the embedding outputs.
        """
        self.filter_type = filter_type.lower()
        if self.filter_type not in ["extreme", "median"]:
            raise ValueError("Invalid filter_type. Choose 'extreme' or 'median'.")
        if self.filter_type == "extreme":
            self.filter_fn = lambda y: extreme_filter_fn(y, percentile=90)
        else:
            self.filter_fn = lambda y: median_filter_fn(y, lower_percentile=45, upper_percentile=55)
        
        super().__init__(run_dir, epoch, num_samples, analysis_name=f"shap_{self.filter_type}", use_embedding=use_embedding)

    def _random_sample_from_file(self):
        """
        Load each basin's data from the validation output file and apply the filter function on the target y.
        Records basin IDs for each selected sample.
        
        Returns:
            final_x_d (np.ndarray): Dynamic features of shape [n_filtered_samples, seq_length, n_dynamic].
            final_x_s (np.ndarray): Static features of shape [n_filtered_samples, n_static].
            basin_ids_all (np.ndarray): 1D array of basin IDs corresponding to each sample.
        """
        validation_output_path = os.path.join(
            self.run_dir,
            "validation",
            f"model_epoch{self.epoch:03d}",
            "validation_all_output.p"
        )
        if not os.path.exists(validation_output_path):
            raise FileNotFoundError(f"Validation output file not found at {validation_output_path}")

        with open(validation_output_path, "rb") as f:
            data = pickle.load(f)

        sampled_x_d_list, sampled_x_s_list, basin_ids_list = [], [], []

        for basin_id, basin_data in tqdm(data.items(), desc="Filtering basins"):
            x_d = torch.tensor(basin_data["x_d"], dtype=torch.float32)  # [n_seq, seq_length, n_dynamic]
            x_s = torch.tensor(basin_data["x_s"], dtype=torch.float32)  # [1, n_static]
            y   = torch.tensor(basin_data["y"], dtype=torch.float32)    # [n_seq, 1]
            if x_s.ndim == 1:
                x_s = x_s.unsqueeze(0)
            if x_s.shape[0] == 1 and x_d.shape[0] > 1:
                x_s = x_s.repeat(x_d.shape[0], 1)
            # Preprocess to remove rows with NaNs.
            x_d, x_s = self._preprocess_basin_data(x_d, x_s)
            valid_y = ~torch.isnan(y).squeeze(1)
            x_d = x_d[valid_y]
            x_s = x_s[valid_y]
            y   = y[valid_y]
            # Apply the filter function on y.
            mask = self.filter_fn(y)
            x_d = x_d[mask]
            x_s = x_s[mask]
            y   = y[mask]
            # Record basin IDs for each selected sample.
            basin_ids = np.array([basin_id] * x_d.shape[0])
            sampled_x_d_list.append(x_d)
            sampled_x_s_list.append(x_s)
            basin_ids_list.append(basin_ids)

        if not sampled_x_d_list:
            raise ValueError("No samples selected after applying the filter.")

        final_x_d = np.concatenate(sampled_x_d_list, axis=0)
        final_x_s = np.concatenate(sampled_x_s_list, axis=0)
        basin_ids_all = np.concatenate(basin_ids_list, axis=0)

        # Shuffle the combined data while keeping basin_ids aligned.
        indices = np.arange(len(final_x_d))
        np.random.shuffle(indices)
        final_x_d = final_x_d[indices]
        final_x_s = final_x_s[indices]
        basin_ids_all = basin_ids_all[indices]

        return final_x_d, final_x_s, basin_ids_all

    def run_shap(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Run SHAP analysis on the filtered samples.

        Returns:
            Tuple containing (shap_values, inputs, basin_ids):
              - shap_values: SHAP values computed for each sample
              - inputs: A dict with keys {"x_d": final_x_d, "x_s": final_x_s}
              - basin_ids: An array of basin IDs corresponding to each sample
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        final_x_d, final_x_s, basin_ids = self._random_sample_from_file()
        if self.use_embedding:
            shap_inputs_array = self._get_embedding_outputs(final_x_d, final_x_s)
            combined_inputs_tensor = torch.tensor(shap_inputs_array, dtype=torch.float32).to(device)
            wrapped_model = self._wrap_model_embedding()
        else:
            combined_inputs = np.hstack([final_x_d.reshape(len(final_x_d), -1), final_x_s])
            combined_inputs_tensor = torch.tensor(combined_inputs, dtype=torch.float32).to(device)
            wrapped_model = self._wrap_model()

        logging.info(f"Combined inputs shape: {combined_inputs_tensor.shape}")
        
        background_indices = np.random.choice(
            range(len(combined_inputs_tensor)),
            size=min(BACKGROUND_SIZE, len(combined_inputs_tensor)),
            replace=False
        )
        background_tensor = combined_inputs_tensor[background_indices].clone().detach().requires_grad_(True)
        explainer = shap.GradientExplainer(wrapped_model, background_tensor, batch_size=BATCH_SIZE)

        shap_values_batches = []
        with tqdm(total=len(combined_inputs_tensor), desc="Calculating SHAP values...") as pbar:
            for i in range(0, len(combined_inputs_tensor), BATCH_SIZE):
                batch = combined_inputs_tensor[i:i+BATCH_SIZE].clone().detach().requires_grad_(True)
                batch_values = explainer.shap_values(batch)
                if isinstance(batch_values, list):
                    batch_values = batch_values[0]
                if torch.is_tensor(batch_values):
                    batch_values = batch_values.cpu().numpy()
                shap_values_batches.append(batch_values)
                pbar.update(len(batch))
                if i % (BATCH_SIZE * 10) == 0:
                    torch.cuda.empty_cache()

        shap_values = np.concatenate(shap_values_batches, axis=0)
        logging.info(f"SHAP values shape: {shap_values.shape}")

        fname_prefix = f"{'shap_values_embedding_' if self.use_embedding else 'shap_values_'}{self.filter_type}"
        np.save(os.path.join(self.results_folder, f"{fname_prefix}.npy"), shap_values)
        fname_prefix2 = f"{'inputs_embedding_' if self.use_embedding else 'inputs_'}{self.filter_type}"
        np.savez(os.path.join(self.results_folder, f"{fname_prefix2}.npz"), x_d=final_x_d, x_s=final_x_s, basin_ids=basin_ids)

        torch.cuda.empty_cache()
        return shap_values, {"x_d": final_x_d, "x_s": final_x_s}, basin_ids

    def aggregate_shap_by_basin(self, shap_values: np.ndarray, basin_ids: np.ndarray, aggregation: str = "median") -> Dict[str, np.ndarray]:
        """
        Aggregate SHAP values for each basin to produce one feature importance vector per basin.
        The aggregated dictionary is saved to disk.

        Args:
            shap_values (np.ndarray): Array of shape [n_samples, n_features] with SHAP values.
            basin_ids (np.ndarray): Array of shape [n_samples] with basin IDs.
            aggregation (str): "median" (default) or "mean".
            
        Returns:
            dict: Mapping from basin_id to an aggregated importance vector of shape [n_features].
        """
        unique_basins = np.unique(basin_ids)
        aggregated = {}
        for basin in unique_basins:
            indices = np.where(basin_ids == basin)[0]
            basin_shap = shap_values[indices]
            if aggregation == "median":
                aggregated_value = np.median(basin_shap, axis=0)
            elif aggregation == "mean":
                aggregated_value = np.mean(basin_shap, axis=0)
            else:
                raise ValueError("Invalid aggregation method. Choose 'median' or 'mean'.")
            aggregated[basin] = aggregated_value

        logging.info(f"Aggregated SHAP values shape (for one basin): {aggregated[unique_basins[0]].shape}")
        
        agg_fname = os.path.join(self.results_folder, f"aggregated_shap_{'embedding_' if self.use_embedding else ''}{self.filter_type}.p")
        with open(agg_fname, "wb") as f:
            pickle.dump(aggregated, f)
        logging.info(f"Aggregated SHAP values saved to {agg_fname}")
        return aggregated

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run SHAP analysis.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory.')
    parser.add_argument('--epoch', type=int, required=True, help='Which epoch checkpoint to load.')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples to use for SHAP analysis.')
    parser.add_argument('--reuse_shap', action='store_true', help='If set, reuse existing SHAP results.')
    parser.add_argument('--use_embedding', action='store_true', help='If set, run SHAP on embedding outputs.')
    args = parser.parse_args()

    analysis = ExtremeMedianSHAPAnalysis(args.run_dir, args.epoch, args.num_samples)
    shap_values, _, basin_ids = analysis.run_shap()
    # Aggregate per basin for clustering
    aggregated_importance = analysis.aggregate_shap_by_basin(shap_values, basin_ids, aggregation="median")
    # aggregated_importance is a dict mapping each basin_id to a feature importance vector