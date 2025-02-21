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
        y: Array or tensor of shape [n_sequences, 1]
        percentile: Percentile cutoff (default 90 for top 10%)
        
    Returns:
        A boolean mask of shape [n_sequences]
    """
    targets = y.squeeze(1)
    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    threshold = np.percentile(targets_np, percentile)
    return targets >= threshold

def median_filter_fn(y, lower_percentile: float = 45, upper_percentile: float = 55):
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

    def _random_sample_from_file(self):
        """
        Load each basin's data from the period-specific output file and apply the filter function on the target y.
        Records basin IDs for each selected sample. Then, sample an equal number of samples from each basin.
        
        Returns:
            final_x_d (np.ndarray): Dynamic features of shape [n_filtered_samples, seq_length, n_dynamic]
            final_x_s (np.ndarray): Static features of shape [n_filtered_samples, n_static]
            basin_ids_all (np.ndarray): 1D array of basin IDs corresponding to each sample
        """
        file_path = os.path.join(
            self.run_dir,
            self.period,
            f"model_epoch{self.epoch:03d}",
            f"{self.period}_all_output.p"
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Output file not found at {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Collect filtered samples per basin
        per_basin_x_d = {}
        per_basin_x_s = {}
        for basin_id, basin_data in tqdm(data.items(), desc="Filtering basins"):
            x_d = torch.tensor(basin_data["x_d"], dtype=torch.float32)  # [n_seq, seq_length, n_dynamic]
            x_s = torch.tensor(basin_data["x_s"], dtype=torch.float32)  # [1, n_static]
            y   = torch.tensor(basin_data["y"], dtype=torch.float32)    # [n_seq, 1]
            if x_s.ndim == 1:
                x_s = x_s.unsqueeze(0)
            if x_s.shape[0] == 1 and x_d.shape[0] > 1:
                x_s = x_s.repeat(x_d.shape[0], 1)
            # Preprocess to remove rows with NaNs
            x_d, x_s = self._preprocess_basin_data(x_d, x_s)
            # Compute valid_y mask from y and convert it to a NumPy array
            valid_y = (~torch.isnan(y).squeeze(1)).cpu().numpy()
            x_d = x_d[valid_y]
            x_s = x_s[valid_y]
            y   = y[valid_y]  # y remains as torch tensor
            # Apply the filter function on y
            mask = self.filter_fn(y)
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            x_d = x_d[mask]
            x_s = x_s[mask]
            y   = y[mask]
            if x_d.shape[0] > 0:
                per_basin_x_d[basin_id] = x_d
                per_basin_x_s[basin_id] = x_s

        if not per_basin_x_d:
            raise ValueError("No samples selected after applying the filter.")

        # Determine how many basins have samples
        basin_ids_list = list(per_basin_x_d.keys())
        n_basins = len(basin_ids_list)
        # Compute per-basin target sample count
        per_basin_target = self.num_samples // n_basins
        logging.info(f"Sampling up to {per_basin_target} samples per basin from {n_basins} basins.")

        equal_x_d_list, equal_x_s_list, equal_basin_ids_list = [], [], []
        for basin_id in basin_ids_list:
            basin_x_d = per_basin_x_d[basin_id]
            basin_x_s = per_basin_x_s[basin_id]
            n_basin = basin_x_d.shape[0]
            if n_basin > per_basin_target:
                chosen_indices = np.random.choice(n_basin, size=per_basin_target, replace=False)
                basin_x_d = basin_x_d[chosen_indices]
                basin_x_s = basin_x_s[chosen_indices]
            # Else, take all available samples
            equal_x_d_list.append(basin_x_d)
            equal_x_s_list.append(basin_x_s)
            equal_basin_ids_list.append(np.array([basin_id] * basin_x_d.shape[0]))

        final_x_d = np.concatenate(equal_x_d_list, axis=0)
        final_x_s = np.concatenate(equal_x_s_list, axis=0)
        basin_ids_all = np.concatenate(equal_basin_ids_list, axis=0)

        # Shuffle the combined data while keeping basin_ids aligned
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

    def aggregate_shap_by_basin(self, shap_values: np.ndarray, basin_ids: np.ndarray, aggregation: str = "median") -> Dict[str, Union[Dict[str, np.ndarray], list]]:
        """
        Aggregate SHAP values for each basin to produce one feature importance vector per basin.
        The input shap_values are assumed to be of shape [n_samples, total_features] where:
          total_features = (seq_length * n_dynamic) + n_static.
        For each sample, the dynamic part (first seq_length*n_dynamic features) is reshaped to [seq_length, n_dynamic]
        and then summed over time to produce a vector of shape [n_dynamic]. This is concatenated with the static part 
        (n_static features) to yield a combined vector of shape [n_dynamic+n_static].
        Then, aggregation (median or mean) is computed over all samples for a basin.
        
        Both the aggregated values and the feature names are saved together to disk.

        Args:
            shap_values (np.ndarray): Array of shape [n_samples, total_features] with SHAP values
            basin_ids (np.ndarray): Array of shape [n_samples] with basin IDs
            aggregation (str): "median" (default) or "mean"
            
        Returns:
            dict: A dictionary with two keys:
                - "aggregated": mapping from basin_id to an aggregated importance vector of shape [n_dynamic+n_static]
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
                # Dynamic part: reshape and sum over time to aggregate to one value per dynamic feature.
                dyn_part = sample[:self.seq_length * n_dynamic].reshape(self.seq_length, n_dynamic)
                dyn_agg = dyn_part.sum(axis=0)  # shape: (n_dynamic,)
                # Static part: slice out and squeeze to ensure 1D.
                stat_part = sample[self.seq_length * n_dynamic:]
                if stat_part.ndim > 1:
                    stat_part = np.squeeze(stat_part, axis=-1)
                # Concatenate dynamic and static parts.
                combined = np.concatenate([dyn_agg, stat_part])
                combined_samples.append(combined)
            combined_samples = np.array(combined_samples)  # shape: [n_samples_basin, n_dynamic+n_static]
            if aggregation == "median":
                aggregated_value = np.median(combined_samples, axis=0)
            elif aggregation == "mean":
                aggregated_value = np.mean(combined_samples, axis=0)
            else:
                raise ValueError("Invalid aggregation method. Choose 'median' or 'mean'.")
            aggregated[basin] = aggregated_value

        logging.info(f"Aggregated SHAP values shape (for one basin): {aggregated[unique_basins[0]].shape}")

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
    shap_values, inputs, basin_ids = analysis.run_shap()
    aggregated_importance = analysis.aggregate_shap_by_basin(shap_values, basin_ids, aggregation="median")
