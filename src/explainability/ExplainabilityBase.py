import os
import yaml
import torch
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.cudalstm import CudaLSTM


class ExplainabilityBase:
    def __init__(self, run_dir, epoch, num_samples, analysis_name):
        """
        Base class for any explainability analysis on a NeuralHydrology model.

        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number to load the model.
            num_samples (int): Number of samples to use for analysis.
            analysis_name (str): A short name for the analysis (e.g., "shap", "integrated_gradients").
        """
        self.run_dir = run_dir
        self.epoch = epoch
        self.num_samples = num_samples
        self.analysis_name = analysis_name

        self.cfg = self._load_config()
        self.dynamic_features = self.cfg["dynamic_inputs"]
        self.static_features = self.cfg["static_attributes"]
        self.seq_length = self.cfg["seq_length"]

        self.model = self._load_model()
        self.results_folder = self._setup_results_folder()

    def _load_config(self):
        """Load the 'config.yml' into a Python dict."""
        config_path = os.path.join(self.run_dir, 'config.yml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find 'config.yml' at {config_path}")
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _load_model(self):
        """
        Load the trained model checkpoint (epoch) specified in the constructor.
        Returns:
            torch.nn.Module: The loaded model in eval mode on the appropriate device.
        """
        config = Config(self.cfg)
        model_class_name = config.model.lower()
        if model_class_name == 'cudalstm':
            model = CudaLSTM(cfg=config)
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

    def _setup_results_folder(self):
        """Set up the folder structure for saving results (analysis_name_results/model_epochXXX)."""
        results_folder = os.path.join(self.run_dir, f"{self.analysis_name}_results")
        epoch_folder = os.path.join(results_folder, f"model_epoch{self.epoch:03d}")
        os.makedirs(epoch_folder, exist_ok=True)
        return epoch_folder

    def _preprocess_basin_data(self, x_d, x_s):
        """
        Preprocess a single basin's dynamic and static data by filtering out rows with NaNs.
        
        Args:
            x_d (torch.Tensor): Dynamic features of shape [n_samples, seq_length, n_dynamic_features].
            x_s (torch.Tensor): Static features of shape [n_samples, n_static_features].

        Returns:
            Tuple of np.ndarray (x_d, x_s) with only valid (non-NaN) samples.
        """
        nan_mask_dynamic = ~torch.isnan(x_d).any(dim=(1, 2))
        nan_mask_static = ~torch.isnan(x_s).any(dim=1)
        valid_mask = nan_mask_dynamic & nan_mask_static

        x_d = x_d[valid_mask]
        x_s = x_s[valid_mask]

        return x_d.cpu().numpy(), x_s.cpu().numpy()

    def _random_sample_from_file(self):
        """
        Efficiently sample data from the validation .p file without loading everything into memory.

        Returns:
            (final_x_d, final_x_s) as np.ndarray:
                - final_x_d: shape [num_samples, seq_length, n_dynamic_features]
                - final_x_s: shape [num_samples, n_static_features]
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

        # First pass: count valid samples per basin
        basin_sample_counts = defaultdict(int)
        total_valid_samples = 0

        for basin_id, basin_data in data.items():
            x_d = torch.tensor(basin_data["x_d"], dtype=torch.float32)
            x_s = torch.tensor(basin_data["x_s"], dtype=torch.float32)
            
            # Check if static features are a single copy
            if x_s.ndim == 1:
                n_samples = x_d.shape[0]  # number of dynamic samples
                # Duplicate x_s so its shape becomes (n_samples, n_static)
                x_s = x_s.unsqueeze(0).repeat(n_samples, 1)
            
            # Process the data (this function expects both to be 2D in the sample dimension)
            x_d, x_s = self._preprocess_basin_data(x_d, x_s)
            valid_count = x_d.shape[0]
            basin_sample_counts[basin_id] = valid_count
            total_valid_samples += valid_count

        # Calculate sampling probabilities per basin
        sampling_probs = {
            basin_id: count / total_valid_samples
            for basin_id, count in basin_sample_counts.items()
        }

        # Allocate samples per basin
        basin_targets = {}
        for basin_id, prob in sampling_probs.items():
            count = basin_sample_counts[basin_id]
            # Round to nearest int
            basin_targets[basin_id] = min(int(self.num_samples * prob + 0.5), count)

        # Second pass: collect stratified samples
        sampled_x_d, sampled_x_s = [], []
        for basin_id, basin_data in tqdm(data.items(), desc="Sampling basins"):
            target = basin_targets[basin_id]
            if target == 0:
                continue

            x_d = torch.tensor(basin_data["x_d"], dtype=torch.float32)
            x_s = torch.tensor(basin_data["x_s"], dtype=torch.float32)
            
            # Duplicate static features if necessary
            if x_s.ndim == 1:
                n_samples = x_d.shape[0]
                x_s = x_s.unsqueeze(0).repeat(n_samples, 1)
            
            # Preprocess the data (this removes any samples with NaNs, etc.)
            x_d, x_s = self._preprocess_basin_data(x_d, x_s)

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
    
    def _aggregate_static_features(self, values, x_s=None):
        """
        Aggregate static feature importance values (SHAP, IG, etc.)

        Args:
            values (np.ndarray): Importance values to aggregate, shape [n_samples, n_features].
            x_s (np.ndarray, optional): Corresponding input values, shape [n_samples, n_features].

        Returns:
            tuple:
                - np.ndarray: Aggregated values, shape [n_samples, n_aggregated_features].
                - np.ndarray or None: Aggregated inputs if x_s is provided.
                - list: Names of aggregated features.
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
            combined_values, aggregated_data['names']
        )

    def _wrap_model(self):
        """
        Create a wrapper so we can pass a single input tensor of shape
        [batch, seq_length * num_dynamic + num_static] directly to self.model.

        Returns:
            torch.nn.Module: A wrapped model that reshapes the inputs and calls the original model.
        """
        class WrappedModel(torch.nn.Module):
            def __init__(self, original_model, seq_length, num_dynamic, num_static):
                super().__init__()
                self.original_model = original_model
                self.seq_length = seq_length
                self.num_dynamic = num_dynamic
                self.num_static = num_static

            def forward(self, inputs):
                """
                Args:
                    inputs (torch.Tensor): Shape [batch, seq_length*num_dynamic + num_static].
                Returns:
                    torch.Tensor: Model outputs, shape [batch, 1].
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
