import os
from pathlib import Path
import yaml
import torch
from torch import Tensor, nn
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import logging
from typing import Any

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.cudalstm import CudaLSTM

from model.model_analyzer import ModelAnalyzer

BATCH_SIZE = 256

class ExplainabilityBase:
    def __init__(self, run_dir: str, epoch: int, num_samples: int, analysis_name: str, period: str = "test") -> None:
        """
        Base class for any explainability analysis on a NeuralHydrology model.

        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number to load the model.
            num_samples (int): Number of samples to use for analysis.
            analysis_name (str): A short name for the analysis (e.g., "shap", "integrated_gradients").
            period (str): The period to load data from (e.g., "train", "validation", "test").
        """
        self.run_dir = run_dir
        self.epoch = epoch
        self.num_samples = num_samples
        self.analysis_name = analysis_name
        self.period = period.lower()

        self.cfg = Config(Path(run_dir) / "config.yml")
        self.seq_length = self.cfg.seq_length
        self.dynamic_features = self.cfg.dynamic_inputs
        self.static_features = sorted(self.cfg.static_attributes) # get_dataset returns static features in alphabetical order
        self.num_dynamic_features = len(self.dynamic_features)

        self.model = self._load_model()
        self.results_folder = self._setup_results_folder()
        
        self.model_analyzer = ModelAnalyzer(run_dir=Path(self.run_dir), epoch=self.epoch, period=self.period)

    def _load_config(self) -> dict[str, Any]:
        """Load the 'config.yml' into a Python dict."""
        config_path = os.path.join(self.run_dir, 'config.yml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find 'config.yml' at {config_path}")
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _load_model(self) -> nn.Module:
        """
        Load the trained model checkpoint (epoch) specified in the constructor.
        Returns:
            nn.Module: The loaded model in eval mode on the appropriate device.
        """
        # config = Config(self.cfg)
        model_class_name = self.cfg.model.lower()
        if model_class_name == 'cudalstm':
            model = CudaLSTM(cfg=self.cfg)
        else:
            raise ValueError(f"Model '{model_class_name}' is not supported by this script.")

        checkpoint_path = os.path.join(self.run_dir, f"model_epoch{self.epoch:03d}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(map_location)
        model.eval()
        return model

    def _setup_results_folder(self) -> str:
        """
        Set up the folder structure for saving results under:
        run_dir/<period>/model_epoch<epoch>/<analysis_name>/
        """
        results_folder = os.path.join(self.run_dir, self.period, f"model_epoch{self.epoch:03d}", self.analysis_name)
        os.makedirs(results_folder, exist_ok=True)
        return results_folder

    def _preprocess_basin_data(self, x_d: Tensor, x_s: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a single basin's dynamic and static data by aligning x_s and filtering out rows with NaNs
        
        Args:
            x_d (Tensor): Dynamic features of shape [n_samples, seq_length, n_dynamic_features]
            x_s (Tensor): Static features of shape [n_samples, n_static_features] or [1, n_static_features]

        Returns:
            Tuple of np.ndarray (x_d, x_s) with only valid (non-NaN) samples
        """
        
        # Align static along the sample dimension features if they are a single row but x_d contains multiple samples
        if x_s.ndim == 2 and x_s.shape[0] == 1 and x_d.shape[0] > 1:
            x_s = x_s.repeat(x_d.shape[0], 1)
        
        nan_mask_dynamic = ~torch.isnan(x_d).any(dim=(1, 2))
        nan_mask_static = ~torch.isnan(x_s).any(dim=1)
        valid_mask = nan_mask_dynamic & nan_mask_static

        x_d = x_d[valid_mask]
        x_s = x_s[valid_mask]

        return x_d.cpu().numpy(), x_s.cpu().numpy()

    def reconstruct_sliding_windows(self, tensor: Tensor) -> Tensor: 
        """ 
        Given a 2D tensor with shape [T, n_features] (e.g. T days of data), 
        reconstruct a 3D tensor of sliding-window sequences of length self.seq_length. 
        Reconstruct sliding-window sequences: for each day i from seq_length-1 to T-1, 
        the sequence is from day (i - seq_length + 1) to day i

        Args:
            tensor (Tensor): A 2D tensor with shape [T, n_features], where T is the number of time steps

        Returns:
            Tensor or None: A 3D tensor with shape [T - seq_length + 1, seq_length, n_features]
            Returns None if there are not enough time steps to form a full sequence
        """
        T = tensor.shape[0]

        if T < self.seq_length or tensor.ndim != 2:
            return None
        
        sequences = tensor.unfold(0, self.seq_length, 1).permute(0, 2, 1)
        return sequences
    
    def _compute_sampling_targets(self, basin_counts: dict[int, int], num_samples: int) -> dict[int, int]:
        """
        Compute the number of samples to draw from each basin using a largest-remainder method
        
        Args:
            basin_counts (dict): Dictionary mapping basin_id to the valid sample count (int)
            num_samples (int): The total number of samples desired
        
        Returns:
            dict: A dictionary mapping basin_id to the number of samples to draw
        """
        total_valid_samples = sum(basin_counts.values())
        
        # Compute ideal counts for each basin
        ideal_counts = {
            basin_id: num_samples * (count / total_valid_samples)
            for basin_id, count in basin_counts.items()
        }
        
        # Assign the floor of each ideal count
        floor_counts = {basin_id: int(ideal) for basin_id, ideal in ideal_counts.items()}
        total_assigned = sum(floor_counts.values())
        deficit = num_samples - total_assigned

        # Distribute remaining samples based on the fractional remainders
        remainders = {
            basin_id: ideal_counts[basin_id] - floor_counts[basin_id]
            for basin_id in ideal_counts
        }
        
        for basin_id, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True):
            if deficit <= 0:
                break
            # Only add if it doesn't exceed the basin's available valid samples
            if floor_counts[basin_id] < basin_counts[basin_id]:
                floor_counts[basin_id] += 1
                deficit -= 1

        return floor_counts

    def _randomly_sample_basin_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Efficiently sample data from the period-specific output .p file without caching all processed
        data into memory at once. This is done in two passes:
        1. First pass: iterate over basins to compute the valid sample count per basin
        2. Second pass: re-process each basin and sample the desired number of examples based on 
            per-basin targets computed using a largest-remainder method
        
        Returns:
            (final_x_d, final_x_s) as np.ndarray:
                - final_x_d: shape [num_samples, seq_length, n_dynamic_features]
                - final_x_s: shape [num_samples, n_static_features]
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

        # First Pass: Compute valid counts per basin
        basin_counts = {}
        for basin_id, basin_data in tqdm(data.items(), desc="Loading basins"):
            x_d = torch.tensor(basin_data["x_d"], dtype=torch.float32)
            x_s = torch.tensor(basin_data["x_s"], dtype=torch.float32)

            x_d = self.reconstruct_sliding_windows(x_d)
            if x_d is None:
                continue 

            # Preprocess to filter out samples with NaNs
            x_d, x_s = self._preprocess_basin_data(x_d, x_s)
            
            valid_count = x_d.shape[0]
            if valid_count > 0:
                basin_counts[basin_id] = valid_count

        # Compute per-basin sampling targets
        basin_targets = self._compute_sampling_targets(basin_counts, self.num_samples)

        # Second Pass: Re-process and sample each basin individually
        sampled_x_d, sampled_x_s = [], []
        for basin_id, basin_data in tqdm(data.items(), desc="Sampling basins"):
            target = basin_targets.get(basin_id, 0)
            if target <= 0:
                continue

            x_d = torch.tensor(basin_data["x_d"], dtype=torch.float32)
            x_s = torch.tensor(basin_data["x_s"], dtype=torch.float32)

            x_d = self.reconstruct_sliding_windows(x_d)
            if x_d is None:
                continue

            x_d, x_s = self._preprocess_basin_data(x_d, x_s)

            # If more valid samples exist than required, randomly choose the target number
            if len(x_d) > target:
                indices = np.random.choice(len(x_d), size=target, replace=False)
                x_d = x_d[indices]
                x_s = x_s[indices]

            sampled_x_d.append(x_d)
            sampled_x_s.append(x_s)

        final_x_d = np.concatenate(sampled_x_d, axis=0)
        final_x_s = np.concatenate(sampled_x_s, axis=0)

        # Shuffle the combined data to avoid any basin order bias
        indices = np.arange(len(final_x_d))
        np.random.shuffle(indices)
        final_x_d = final_x_d[indices]
        final_x_s = final_x_s[indices]

        return final_x_d, final_x_s

    def load_and_sample_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly sample a subset (of size self.num_samples) of the full inputs.

        Returns:
            final_x_d: np.ndarray of shape [num_samples, seq_length, n_dynamic]
            final_x_s: np.ndarray of shape [num_samples, n_static]
            sample_indices: np.ndarray of shape [num_samples] containing the global indices of the selected samples.
        """
        inputs = self.model_analyzer.get_inputs()
        x_d, x_s = inputs["x_d"], inputs["x_s"]
        logging.info(f"Before sampling: x_d shape {x_d.shape}, x_s shape {x_s.shape}") # Debugging
        total_samples = x_d.shape[0]
        if self.num_samples < total_samples:
            indices = np.random.choice(total_samples, size=self.num_samples, replace=False)
            final_x_d = x_d[indices]
            final_x_s = x_s[indices]
        else:
            final_x_d = x_d
            final_x_s = x_s
            indices = np.arange(total_samples)
        logging.info(f"After sampling: final_x_d shape {final_x_d.shape}, final_x_s shape {final_x_s.shape}") # Debugging
        return final_x_d, final_x_s, indices
    
    def _aggregate_static_features(self, values: np.ndarray, x_s: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Aggregate static feature importance values (SHAP, IG, etc.)

        Args:
            values (np.ndarray): Importance values to aggregate, shape [n_samples, n_features]
            x_s (np.ndarray, optional): Corresponding input values, shape [n_samples, n_features]

        Returns:
            tuple:
                - np.ndarray: Aggregated values, shape [n_samples, n_aggregated_features]
                - np.ndarray or None: Aggregated inputs if x_s is provided
                - list: Names of aggregated features
        """
        feature_groups = {
            'glc_pc': 'glc_pc_aggregated',
            'pnv_pc': 'pnv_pc_aggregated'
        }

        aggregated_data = {
            'values': [],
            'inputs': [] if x_s is not None else None,
            'names': []
        }

        # Find feature indices for each group
        group_indices = {
            prefix: [i for i, name in enumerate(self.static_features) if prefix in name]
            for prefix in feature_groups
        }

        grouped_indices = set().union(*group_indices.values())

        # Process individual features
        for i, name in enumerate(self.static_features):
            if i not in grouped_indices:
                aggregated_data['values'].append(values[:, i])
                if x_s is not None:
                    aggregated_data['inputs'].append(x_s[:, i])
                aggregated_data['names'].append(name)

        # Process grouped features
        for prefix, indices in group_indices.items():
            if indices:
                aggregated_data['values'].append(np.mean(values[:, indices], axis=1))
                if x_s is not None:
                    aggregated_data['inputs'].append(np.mean(x_s[:, indices], axis=1))
                aggregated_data['names'].append(feature_groups[prefix])

        combined_values = np.column_stack(aggregated_data['values'])
        combined_inputs = np.column_stack(aggregated_data['inputs']) if x_s is not None else None

        return (combined_values, combined_inputs, aggregated_data['names']) if x_s is not None else (
            combined_values, aggregated_data['names'])

    def _wrap_model(self) -> nn.Module:
        """
        Create a wrapper so we can pass a single input tensor of shape
        [batch, seq_length * num_dynamic + num_static] directly to self.model

        Returns:
            nn.Module: A wrapped model that reshapes the inputs and calls the original model
        """
        class WrappedModel(nn.Module):
            def __init__(self, original_model: nn.Module, seq_length: int, num_dynamic: int, num_static: int) -> None:
                super().__init__()
                self.original_model = original_model
                self.seq_length = seq_length
                self.num_dynamic = num_dynamic
                self.num_static = num_static

            def forward(self, inputs: Tensor) -> Tensor:
                """
                Args:
                    inputs (Tensor): Shape [batch, seq_length*num_dynamic + num_static]
                Returns:
                    Tensor: Model outputs, shape [batch, 1]
                """
                x_d_flat = inputs[:, : self.seq_length * self.num_dynamic]
                x_s = inputs[:, self.seq_length * self.num_dynamic:]
                x_d = x_d_flat.view(-1, self.seq_length, self.num_dynamic)

                model_inputs = {"x_d": x_d, "x_s": x_s}
                out_dict = self.original_model(model_inputs)
                # For example, return the last time step, first output dimension:
                return out_dict["y_hat"][:, -1, 0].unsqueeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return WrappedModel(
            self.model, 
            seq_length=self.seq_length,
            num_dynamic=len(self.dynamic_features),
            num_static=len(self.static_features)
        ).to(device).eval()
