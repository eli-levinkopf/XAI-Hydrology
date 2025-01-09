import os
import argparse
import torch
import numpy as np
import yaml
import pickle
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import shap

# neuralhydrology imports
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.cudalstm import CudaLSTM

BACKGROUND_SIZE=1024
BATCH_SIZE=256


class SHAPAnalysis:
    def __init__(self, run_dir, epoch, num_samples):
        """
        Initialize SHAPAnalysis for a trained NeuralHydrology model.
        
        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number to load the model.
            num_samples (int): Number of samples for SHAP analysis.
        """
        self.run_dir = run_dir
        self.epoch = epoch
        self.num_samples = num_samples
        self.cfg = self._load_config()
        self.dynamic_features = self.cfg["dynamic_inputs"]
        self.static_features = self.cfg["static_attributes"]
        self.seq_length = self.cfg["seq_length"]
        self.model = self._load_model()
        self.results_folder = self._setup_results_folder()

    def _load_config(self):
        """Load the config.yml into a Python dict."""
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
        """Set up the folder structure for saving results."""
        results_folder = os.path.join(self.run_dir, "shap_results")
        epoch_folder = os.path.join(results_folder, f"model_epoch{self.epoch:03d}")
        os.makedirs(epoch_folder, exist_ok=True)
        return epoch_folder

    def _preprocess_basin_data(self, x_d, x_s):
        """
        Preprocess a single basin's dynamic and static data.
        Filters out samples with NaNs and ensures the correct shapes.
        """
        nan_mask_dynamic = ~torch.isnan(x_d).any(dim=(1, 2))
        nan_mask_static = ~torch.isnan(x_s).any(dim=1)
        valid_mask = nan_mask_dynamic & nan_mask_static

        x_d = x_d[valid_mask]
        x_s = x_s[valid_mask]

        return x_d.cpu().numpy(), x_s.cpu().numpy()

    def _random_sample_from_file(self):
        """
        Efficiently sample data without loading everything into memory.
        Returns:
            np.ndarray, np.ndarray: Arrays for x_d (dynamic inputs) and x_s (static inputs).
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

    def run_shap(self):
        """
        Main method to compute and save SHAP values for the model.
        
        Returns:
            shap_values (np.ndarray): The computed SHAP values.
            final_inputs (dict): {"x_d": final_x_d, "x_s": final_x_s}
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Sample dynamic and static inputs
        final_x_d, final_x_s = self._random_sample_from_file()

        # Combine x_d and x_s into a single tensor for SHAP
        combined_inputs = np.hstack([
            final_x_d.reshape(len(final_x_d), -1),
            final_x_s
        ])
        combined_inputs_tensor = torch.tensor(combined_inputs, dtype=torch.float32).to(device)

        # Define a wrapper so SHAP can interpret our model
        class WrappedModel(torch.nn.Module):
            def __init__(self, original_model, seq_length, num_dynamic, num_static):
                super().__init__()
                self.original_model = original_model
                self.seq_length = seq_length
                self.num_dynamic = num_dynamic
                self.num_static = num_static

            def forward(self, inputs):
                x_d_flat = inputs[:, :self.seq_length * self.num_dynamic]
                x_s = inputs[:, self.seq_length * self.num_dynamic:]
                x_d = x_d_flat.view(-1, self.seq_length, self.num_dynamic)

                model_inputs = {"x_d": x_d, "x_s": x_s}
                out_dict = self.original_model(model_inputs)
                # Output streamflow prediction for the last day
                return out_dict["y_hat"][:, -1, 0].unsqueeze(1)

        wrapped_model = WrappedModel(
            self.model,
            seq_length=self.seq_length,
            num_dynamic=len(self.dynamic_features),
            num_static=len(self.static_features)
        ).to(device)
        wrapped_model.eval()

        # Create background (reference) data
        background_indices = np.random.choice(
            range(len(combined_inputs_tensor)),
            size=min(BACKGROUND_SIZE, len(combined_inputs_tensor)),
            replace=False,
        )
        background_tensor = combined_inputs_tensor[background_indices].clone().detach().requires_grad_(True)

        explainer = shap.GradientExplainer(wrapped_model, background_tensor, batch_size=BATCH_SIZE)

        shap_values_batches = []
        with tqdm(total=len(combined_inputs_tensor), desc="Calculating SHAP values...") as pbar:
            for i in range(0, len(combined_inputs_tensor), BATCH_SIZE):
                batch = combined_inputs_tensor[i:i + BATCH_SIZE].clone().detach().requires_grad_(True)
                batch_values = explainer.shap_values(batch)
                if isinstance(batch_values, list):
                    batch_values = batch_values[0]
                if torch.is_tensor(batch_values):
                    batch_values = batch_values.cpu().numpy()

                shap_values_batches.append(batch_values)
                pbar.update(len(batch))

                # Clear CUDA cache periodically
                if i % (BATCH_SIZE * 10) == 0:
                    torch.cuda.empty_cache()

        shap_values = np.concatenate(shap_values_batches, axis=0)

        # Save results for reuse
        np.save(os.path.join(self.results_folder, "shap_values.npy"), shap_values)
        np.savez(os.path.join(self.results_folder, "inputs.npz"), x_d=final_x_d, x_s=final_x_s)

        # Clean up
        torch.cuda.empty_cache()

        # Example of saving a default visualization
        self._plot_shap_summary(shap_values, {"x_d": final_x_d, "x_s": final_x_s})

        return shap_values, {"x_d": final_x_d, "x_s": final_x_s}

    def _aggregate_shap_values(self, shap_values, x_s=None):
        """
        Aggregate SHAP values for static features, combining related features into groups.
        
        Args:
            shap_values (np.ndarray): Static SHAP values to aggregate, shape [n_samples, n_features]
            x_s (np.ndarray, optional): Static input values to aggregate, shape [n_samples, n_features]
        
        Returns:
            tuple: Contains:
                - np.ndarray: Aggregated SHAP values, shape [n_samples, n_aggregated_features]
                - np.ndarray or None: Aggregated inputs if x_s provided, shape [n_samples, n_aggregated_features]
                - list: Names of aggregated features
        """
        # Define feature groups with their prefixes
        feature_groups = {
            'glc_pc': 'glc_pc_aggregated',
            'pnv_pc': 'pnv_pc_aggregated'
        }
        
        # Initialize lists for aggregated data
        aggregated_data = {
            'shap_values': [],
            'inputs': [] if x_s is not None else None,
            'names': []
        }
        
        # Create index mapping for each feature group
        group_indices = {
            prefix: [i for i, name in enumerate(self.static_features) if prefix in name]
            for prefix in feature_groups
        }
        
        # Process regular features first
        grouped_indices = set().union(*group_indices.values())
        for i, name in enumerate(self.static_features):
            if i not in grouped_indices:
                aggregated_data['shap_values'].append(shap_values[:, i])
                if x_s is not None:
                    aggregated_data['inputs'].append(x_s[:, i])
                aggregated_data['names'].append(name)
        
        # Process feature groups
        for prefix, indices in group_indices.items():
            if indices:  # Only process if group has features
                aggregated_data['shap_values'].append(np.mean(shap_values[:, indices], axis=1))
                if x_s is not None:
                    aggregated_data['inputs'].append(np.mean(x_s[:, indices], axis=1))
                aggregated_data['names'].append(feature_groups[prefix])
        
        # Combine results
        combined_shap = np.column_stack(aggregated_data['shap_values'])
        combined_inputs = (np.column_stack(aggregated_data['inputs']) 
                        if x_s is not None else None)
        
        return (combined_shap, combined_inputs, aggregated_data['names']) if x_s is not None else (
            combined_shap, aggregated_data['names'])

    def _plot_shap_summary(self, shap_values, inputs):
        """Example SHAP summary visualization with separate x_d and x_s."""
        shap_values = shap_values.squeeze(-1)
        
        x_d = inputs["x_d"]  # [n_samples, seq_length, n_dynamic]
        x_s = inputs["x_s"]  # [n_samples, n_static]

        n_samples = shap_values.shape[0]
        n_dynamic = len(self.dynamic_features)
        n_static = len(self.static_features)

        # Split SHAP values back into dynamic and static parts
        dynamic_vals = shap_values[:, :self.seq_length * n_dynamic].reshape(n_samples, self.seq_length, n_dynamic)
        static_vals = shap_values[:, self.seq_length * n_dynamic:]

        # Aggregate static features
        combined_static_shap, combined_static_inputs, agg_static_names = self._aggregate_shap_values(static_vals, x_s)

        # Sum dynamic features across time for a simple summary
        dynamic_shap = dynamic_vals.sum(axis=1)
        dynamic_inputs = x_d.sum(axis=1)

        combined_shap = np.concatenate([dynamic_shap, combined_static_shap], axis=1)
        combined_inputs = np.concatenate([dynamic_inputs, combined_static_inputs], axis=1)
        feature_names = self.dynamic_features + agg_static_names

        # Plot
        plt.figure(figsize=(10, 12))
        shap.summary_plot(
            combined_shap,
            combined_inputs,
            feature_names=feature_names,
            max_display=len(feature_names),
            show=False
        )
        plt.xlim([np.min(combined_shap), np.max(combined_shap)])
        summary_plot_path = os.path.join(self.results_folder, "shap_summary_plot.png")
        plt.savefig(summary_plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Summary plot saved to {summary_plot_path}")

    def _plot_shap_contributions_over_time(self, shap_values):
        """
        Generate a plot showing the overall contribution of each dynamic feature to the prediction over time.

        Args:
            shap_values (np.ndarray): SHAP values array of shape [n_samples, seq_length * num_dynamic + num_static].
            inputs (dict): Dictionary with keys "x_d" (dynamic inputs) and "x_s" (static inputs).
        Returns:
            None
        """

        n_dynamic = len(self.dynamic_features)

        # Extract dynamic SHAP values
        dynamic_shap_values = shap_values[:, :self.seq_length * n_dynamic].reshape(-1, self.seq_length, n_dynamic)

        # Calculate the median absolute SHAP value for each feature over all samples
        median_shap_values = np.median(np.abs(dynamic_shap_values), axis=0)  # Shape: [seq_length, n_dynamic]

        # Create a new subfolder for these plots
        overall_contrib_folder = os.path.join(self.results_folder, "shap_overall_contrib")
        os.makedirs(overall_contrib_folder, exist_ok=True)

        for feature_idx, feature_name in enumerate(self.dynamic_features):
            plt.figure(figsize=(15, 6))
            
            # Plot the mean SHAP values over time for the current feature
            plt.plot(
                np.arange(-self.seq_length, 0),
                median_shap_values[:, feature_idx], 
                label=f"Overall SHAP Contribution for {feature_name}",
                color='b'
            )

            plt.xticks(np.arange(-self.seq_length, 1, 50))
            plt.title(f"Overall Contribution of {feature_name} to Prediction Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Mean Absolute SHAP Value")
            plt.legend(loc="upper left")
            plt.grid(True)

            # Save plot
            plot_path = os.path.join(overall_contrib_folder, f"overall_shap_contribution_{feature_name}.png")
            plt.savefig(plot_path, bbox_inches="tight", dpi=300)
            plt.close()
        print(f"Saved overall contribution plots for to {overall_contrib_folder}")

    def _plot_shap_individual_samples(self, shap_values, n_samples=5, seed=42):
        """
        Generate plots showing individual sample SHAP values for each dynamic feature over time.
        
        Args:
            shap_values (np.ndarray): SHAP values array of shape [n_samples, seq_length * num_dynamic + num_static]
            n_samples (int): Number of random samples to plot for each feature
            seed (int): Random seed for reproducibility
        """
        n_dynamic = len(self.dynamic_features)
        
        # Extract dynamic SHAP values
        dynamic_shap_values = shap_values[:, :self.seq_length * n_dynamic].reshape(-1, self.seq_length, n_dynamic)
        
        # Create a new subfolder for individual sample plots
        individual_samples_folder = os.path.join(self.results_folder, "shap_individual_samples")
        os.makedirs(individual_samples_folder, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Generate random sample indices
        total_samples = dynamic_shap_values.shape[0]
        random_indices = np.random.choice(total_samples, n_samples, replace=False)
        
        # Plot each feature
        for feature_idx, feature_name in enumerate(self.dynamic_features):
            plt.figure(figsize=(15, 8))
            
            # Plot individual samples
            for idx, sample_idx in enumerate(random_indices):
                sample_values = dynamic_shap_values[sample_idx, :, feature_idx]
                plt.plot(
                    np.arange(-self.seq_length, 0),
                    sample_values,
                    label=f'Sample {idx + 1}',
                    alpha=0.7,
                    linewidth=2
                )
                
            # Add median line for reference
            median_values = np.median(dynamic_shap_values[:, :, feature_idx], axis=0)
            plt.plot(
                np.arange(-self.seq_length, 0),
                median_values,
                label='Median',
                color='black',
                linestyle='--',
                linewidth=2
            )
                
            plt.xticks(np.arange(-self.seq_length, 1, 50))
            plt.title(f"Individual Sample SHAP Values for {feature_name}")
            plt.xlabel("Time Step")
            plt.ylabel("SHAP Value")
            plt.legend(loc="upper left")
            plt.grid(True)
            
            # Save plot
            plot_path = os.path.join(individual_samples_folder, f"individual_samples_{feature_name}.png")
            plt.savefig(plot_path, bbox_inches="tight", dpi=300)
            plt.close()
            
        print(f"Saved individual sample plots to {individual_samples_folder}")

    def _plot_shap_summary_bar(self, shap_values):
        """
        Generate a SHAP Summary Bar Plot showing the average absolute SHAP values for each feature.

        Args:
            shap_values (np.ndarray): SHAP values array of shape [n_samples, seq_length * num_dynamic + num_static].
        """

        # Extract dynamic and static SHAP values
        num_dynamic = len(self.dynamic_features)
        dynamic_shap_values = shap_values[:, :self.seq_length * num_dynamic].reshape(-1, self.seq_length, num_dynamic)
        static_shap_values = shap_values[:, self.seq_length * num_dynamic:] 

        # Sum SHAP values across time steps and then average across samples for dynamic features
        dynamic_summed_shap = np.sum(np.abs(dynamic_shap_values), axis=1)  # Shape: [n_samples, num_dynamic]
        dynamic_mean_shap = np.mean(dynamic_summed_shap, axis=0)  # Shape: [num_dynamic]


        # Aggregate static features
        combined_static_shap, agg_static_names = self._aggregate_shap_values(static_shap_values)
        static_mean_shap = np.mean(np.abs(combined_static_shap), axis=0).squeeze()  # Shape: [num_static]


        # Combine feature names and mean SHAP values
        feature_names = self.dynamic_features + agg_static_names
        mean_shap_values = np.concatenate([dynamic_mean_shap, static_mean_shap])

        # Sort features by importance
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_shap_values = mean_shap_values[sorted_indices]

        # Create bar plot
        plt.figure(figsize=(12, 8))
        plt.barh(sorted_features, sorted_shap_values, color='skyblue')
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title("SHAP Summary Bar Plot")
        plt.gca().invert_yaxis()  # Invert y-axis for better readability

        # Save plot
        plot_path = os.path.join(self.results_folder, "shap_summary_bar_plot.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"SHAP Summary Bar Plot saved to {plot_path}")

    def run_shap_visualizations(self, shap_values, inputs):
        self._plot_shap_summary(shap_values, {"x_d": inputs["x_d"], "x_s": inputs["x_s"]})
#        self._plot_shap_contributions_over_time(shap_values)
#        self._plot_shap_individual_samples(shap_values)
        self._plot_shap_summary_bar(shap_values)



def main():
    parser = argparse.ArgumentParser(description="Run SHAP analysis.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory.')
    parser.add_argument('--epoch', type=int, required=True, help='Which epoch checkpoint to load.')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples to use for SHAP analysis.')
    parser.add_argument('--reuse-shap', action='store_true',
                        help='If set, reuse shap_values.npy and inputs.npz if they exist, skipping new computation.')

    args = parser.parse_args()

    # Initialize the SHAP analysis object
    analysis = SHAPAnalysis(args.run_dir, args.epoch, args.num_samples)

    shap_values_path = os.path.join(analysis.results_folder, "shap_values.npy")
    inputs_path = os.path.join(analysis.results_folder, "inputs.npz")

    if args.reuse_shap:
        if os.path.exists(shap_values_path) and os.path.exists(inputs_path):
            print("Reusing existing SHAP files. Loading from disk...")
            shap_values = np.load(shap_values_path)
            inputs = np.load(inputs_path)
        else: 
            raise FileNotFoundError("Reuse flag set, but SHAP files not found. Please run SHAP analysis from scratch.")

    else:
        # Run SHAP analysis from scratch
        shap_values, inputs = analysis.run_shap()

    # Run visualizations
    analysis.run_shap_visualizations(shap_values, inputs)

if __name__ == "__main__":
    main()

