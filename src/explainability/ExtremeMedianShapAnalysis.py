from typing import Dict, Tuple, Union
import os
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
import pickle
import logging
import shap
import argparse

from ShapAnalysis import SHAPAnalysis 

BACKGROUND_SIZE = 1024
BATCH_SIZE = 256

def extreme_filter_fn(y, percentile: float = 90) -> np.ndarray:
    """
    Select sequences where the target streamflow (from y) is above the given percentile.
    
    Args:
        y: Array or tensor of shape [n_sequences, 1]
        percentile: Percentile cutoff (default 90 for top 10%)
        
    Returns:
        A boolean mask of shape [n_sequences]
    """
    targets = y.squeeze(1)
    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    # If there are no target values, return an empty mask
    if targets_np.size == 0:
        return np.array([], dtype=bool)
    threshold = np.percentile(targets_np, percentile)
    return targets >= threshold

def median_filter_fn(y, lower_percentile: float = 45, upper_percentile: float = 55) -> np.ndarray:
    """
    Select sequences where the target streamflow (from y) is between the lower and upper percentiles.
    
    Args:
        y: Array or tensor of shape [n_sequences, 1]
        lower_percentile: Lower percentile cutoff (default 45)
        upper_percentile: Upper percentile cutoff (default 55)
        
    Returns:
        A boolean mask of shape [n_sequences]
    """
    targets = y.squeeze(1)
    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    # If there are no target values, return an empty mask
    if targets_np.size == 0:
        return np.array([], dtype=bool)
    lower = np.percentile(targets_np, lower_percentile)
    upper = np.percentile(targets_np, upper_percentile)
    return (targets >= lower) & (targets <= upper)

class ExtremeMedianSHAPAnalysis(SHAPAnalysis):
    def __init__(self, 
                 run_dir: str, 
                 epoch: int, 
                 num_samples: int = 100000, 
                 period: str = "test",
                 filter_type: str = "extreme", 
                 use_embedding: bool = False) -> None:
        """
        Initialize ExtremeMedianSHAPAnalysis

        Args:
            run_dir (str): Path to the run directory
            epoch (int): Epoch number to load the model
            num_samples (int): Total number of samples for SHAP analysis
            period (str): The period to run the analysis on ("train", "validation", or "test")
            filter_type (str): Either "extreme" (default) or "median". Determines which filter to use
            use_embedding (bool): If True, run SHAP on the embedding outputs
        """
        self.filter_type = filter_type.lower()
        if self.filter_type not in ["extreme", "median"]:
            raise ValueError("Invalid filter_type. Choose 'extreme' or 'median'")
        if self.filter_type == "extreme":
            self.filter_fn = lambda y: extreme_filter_fn(y, percentile=90)
        else:
            self.filter_fn = lambda y: median_filter_fn(y, lower_percentile=45, upper_percentile=55)
        
        super().__init__(run_dir, epoch, num_samples, period=period, use_embedding=use_embedding)

    def _randomly_sample_basin_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Use ModelAnalyzer's get_inputs to obtain all samples, then use the dataset's lookup table to obtain basin IDs. 
        Apply filtering based on the target (last time step) and subsample an equal number of samples per basin.
        
        Returns:
            sampled_x_d (np.ndarray): Dynamic features of shape [n_filtered_samples, seq_length, n_dynamic]
            sampled_x_s (np.ndarray): Static features of shape [n_filtered_samples, n_static]
            sampled_ids (np.ndarray): 1D array of basin IDs corresponding to each sample
        """
        inputs = self.model_analyzer.get_inputs(fetch_target=True)
        x_d, x_s, y = inputs["x_d"], inputs["x_s"], inputs["y"]
        
        # Retrieve basin IDs from the dataset's lookup table
        # The lookup table maps each sample index to a tuple: (basin_id, [indices])
        lookup_table = self.model_analyzer.dataset.lookup_table
        basin_ids = np.array([lookup_table[i][0] for i in range(len(lookup_table))])
        
        # Remove samples with NaN targets (y has shape [num_samples, 1])
        valid_mask = ~np.isnan(y.squeeze(1))
        x_d = x_d[valid_mask]
        x_s = x_s[valid_mask]
        y   = y[valid_mask]
        basin_ids = basin_ids[valid_mask]
        
        # Apply filtering based on y values
        mask = self.filter_fn(torch.tensor(y, dtype=torch.float32))
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        x_d = x_d[mask]
        x_s = x_s[mask]
        basin_ids_filtered = basin_ids[mask]
        
        # Compute per-basin sampling targets
        unique_basins, counts = np.unique(basin_ids_filtered, return_counts=True)
        basin_counts = dict(zip(unique_basins, counts))
        basin_targets = self._compute_sampling_targets(basin_counts, self.num_samples)
        
        # Subsample an equal number of samples per basin
        sampled_x_d, sampled_x_s, sampled_basin_ids = [], [], []
        for basin in unique_basins:
            indices = np.where(basin_ids_filtered == basin)[0]
            target = basin_targets.get(basin, 0)
            if len(indices) > target:
                indices = np.random.choice(indices, size=target, replace=False)
            sampled_x_d.append(x_d[indices])
            sampled_x_s.append(x_s[indices])
            sampled_basin_ids.append(np.array([basin] * len(indices)))
        
        sampled_x_d = np.concatenate(sampled_x_d, axis=0)
        sampled_x_s = np.concatenate(sampled_x_s, axis=0)
        sampled_basin_ids = np.concatenate(sampled_basin_ids, axis=0)
        
        # Shuffle the final sampled data while keeping basin IDs aligned
        perm = np.random.permutation(len(sampled_x_d))
        sampled_x_d = sampled_x_d[perm]
        sampled_x_s = sampled_x_s[perm]
        sampled_basin_ids = sampled_basin_ids[perm]

        logging.info(f"Filtered samples: x_d shape {sampled_x_d.shape}, x_s shape {sampled_x_s.shape}, basin_ids shape {sampled_basin_ids.shape}")
        
        return sampled_x_d, sampled_x_s, sampled_basin_ids

    def _get_stratified_background(self, combined_inputs_tensor: Tensor, basin_ids: np.ndarray) -> Tensor:
        """
        Create a stratified background tensor ensuring every basin is represented, 
        but then subsample so that the total number of background samples does not exceed BACKGROUND_SIZE
        
        Args:
            combined_inputs_tensor (Tensor): Combined input tensor of shape [n_samples, total_features]
            basin_ids (np.ndarray): Array of shape [n_samples] with basin IDs
        
        Returns:
            Tensor: A background tensor with gradients enabled
        """
        unique_basins = np.unique(basin_ids)
        # Determine k = number of samples to pick per basin; at least one sample per basin
        k = max(1, BACKGROUND_SIZE // len(unique_basins))
        background_indices = []
        for b in unique_basins:
            basin_indices = np.where(basin_ids == b)[0]
            if len(basin_indices) <= k:
                background_indices.extend(basin_indices.tolist())
            else:
                background_indices.extend(np.random.choice(basin_indices, size=k, replace=False).tolist())

        background_indices = np.array(background_indices)
        
        # If more background samples than desired, subsample to BACKGROUND_SIZE
        if len(background_indices) > BACKGROUND_SIZE:
            background_indices = np.random.choice(background_indices, size=BACKGROUND_SIZE, replace=False)
        
        background_tensor = combined_inputs_tensor[background_indices].clone().detach().requires_grad_(True)
        return background_tensor

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
        final_x_d, final_x_s, basin_ids = self._randomly_sample_basin_data()
        if self.use_embedding:
            shap_inputs_array = self._get_embedding_outputs(final_x_d, final_x_s)
            combined_inputs_tensor = torch.tensor(shap_inputs_array, dtype=torch.float32).to(device)
            wrapped_model = self._wrap_model_embedding()
        else:
            combined_inputs = np.hstack([final_x_d.reshape(len(final_x_d), -1), final_x_s])
            combined_inputs_tensor = torch.tensor(combined_inputs, dtype=torch.float32).to(device)
            wrapped_model = self._wrap_model()

        logging.info(f"Combined inputs shape: {combined_inputs_tensor.shape}")
        
        background_tensor = self._get_stratified_background(combined_inputs_tensor, basin_ids)
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

    def aggregate_shap_by_basin(self, shap_values: np.ndarray, basin_ids: np.ndarray) -> Dict[str, Union[Dict[str, np.ndarray], list]]:
        """
        Aggregate SHAP values for each basin to produce two feature importance vectors per basin:
        one that is the median of the signed values and one that is the median of the absolute values.
        The input shap_values are assumed to be of shape [n_samples, total_features] where:
        total_features = (seq_length * n_dynamic) + n_static.
        For each sample, the dynamic part (first seq_length*n_dynamic features) is reshaped to [seq_length, n_dynamic]
        and then summed over time to produce a vector of shape [n_dynamic]. This is concatenated with the static part 
        (n_static features) to yield a combined vector of shape [n_dynamic+n_static].
        Then the median is computed over all samples for a basin, and both the signed and absolute medians are saved.
        
        Args:
            shap_values (np.ndarray): Array of shape [n_samples, total_features] with SHAP values.
            basin_ids (np.ndarray): Array of shape [n_samples] with basin IDs.
            
        Returns:
            dict: A dictionary with two keys:
                - "aggregated": mapping from basin_id to a dict with keys "signed" and "absolute" each holding a
                vector of shape [n_dynamic+n_static]
                - "feature_names": list of feature names (length = n_dynamic+n_static)
        """
        n_dynamic = len(self.dynamic_features)
        unique_basins = np.unique(basin_ids)
        aggregated = {}
        
        for basin in unique_basins:
            indices = np.where(basin_ids == basin)[0]
            basin_shap = shap_values[indices]  # shape: [n_samples_basin, total_features]
            combined_samples = []
            for sample in basin_shap:
                # Dynamic part: reshape and sum over time to aggregate to one value per dynamic feature
                dyn_part = sample[:self.seq_length * n_dynamic].reshape(self.seq_length, n_dynamic)
                dyn_agg = dyn_part.sum(axis=0)  # shape: (n_dynamic,)
                # Static part: slice out and squeeze to ensure 1D
                stat_part = sample[self.seq_length * n_dynamic:]
                if stat_part.ndim > 1:
                    stat_part = np.squeeze(stat_part, axis=-1)
                # Concatenate dynamic and static parts
                combined = np.concatenate([dyn_agg, stat_part])
                combined_samples.append(combined)
            combined_samples = np.array(combined_samples)  # shape: [n_samples_basin, n_dynamic+n_static]
            # Compute median aggregation: both for signed and absolute values
            signed_agg = np.median(combined_samples, axis=0)
            absolute_agg = np.median(np.abs(combined_samples), axis=0)
            aggregated[basin] = {"signed": signed_agg, "absolute": absolute_agg}

        out = {
            "aggregated": aggregated,
            "feature_names": self.dynamic_features + self.static_features
        }
        agg_fname = os.path.join(self.results_folder, f"aggregated_shap_{'embedding_' if self.use_embedding else ''}{self.filter_type}.p")
        with open(agg_fname, "wb") as f:
            pickle.dump(out, f)
        logging.info(f"Aggregated SHAP values and feature names saved to {agg_fname}")
        return out

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Run SHAP analysis.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory.')
    parser.add_argument('--epoch', type=int, required=True, help='Which epoch checkpoint to load.')
    parser.add_argument('--num_samples', type=int, default=100000, help='Total number of samples to use for SHAP analysis.')
    parser.add_argument('--period', type=str, default="test", help='Period to load data from (train/validation/test).')
    parser.add_argument('--filter_type', type=str, default="extreme", help='Filter type: "extreme" or "median".')
    parser.add_argument('--reuse_shap', action='store_true', help='If set, reuse existing SHAP results.')
    parser.add_argument('--use_embedding', action='store_true', help='If set, run SHAP on embedding outputs.')
    args = parser.parse_args()

    analysis = ExtremeMedianSHAPAnalysis(
        run_dir=args.run_dir,
        epoch=args.epoch,
        num_samples=args.num_samples,
        period=args.period,
        filter_type=args.filter_type,
        use_embedding=args.use_embedding
    )
    
    if args.reuse_shap:
        fname_prefix = f"{'shap_values_embedding_' if args.use_embedding else 'shap_values_'}{args.filter_type}"
        fname_prefix2 = f"{'inputs_embedding_' if args.use_embedding else 'inputs_'}{args.filter_type}"
        shap_values_path = os.path.join(analysis.results_folder, f"{fname_prefix}.npy")
        inputs_path = os.path.join(analysis.results_folder, f"{fname_prefix2}.npz")
        if os.path.exists(shap_values_path) and os.path.exists(inputs_path):
            logging.info("Reusing existing SHAP files. Loading from disk...")
            shap_values = np.load(shap_values_path)
            inputs_data = np.load(inputs_path)
            basin_ids = inputs_data["basin_ids"]
        else:
            raise FileNotFoundError("Reuse flag set, but at least one of the SHAP files (values or inputs) is missing.")
    else:
        shap_values, inputs, basin_ids = analysis.run_shap()
    
    analysis.aggregate_shap_by_basin(shap_values, basin_ids)