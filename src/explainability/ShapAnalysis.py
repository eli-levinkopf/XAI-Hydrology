import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import shap
import logging
torch.backends.cudnn.enabled = False

from ExplainabilityBase import ExplainabilityBase 


BACKGROUND_SIZE = 1024
BATCH_SIZE = 256


class SHAPAnalysis(ExplainabilityBase):
    def __init__(self, run_dir, epoch, num_samples):
        """
        Initialize SHAPAnalysis for a trained NeuralHydrology model.
        
        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number to load the model.
            num_samples (int): Number of samples for SHAP analysis.
        """
        super().__init__(run_dir, epoch, num_samples, analysis_name="shap")

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

        wrapped_model = self._wrap_model()

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

        return shap_values, {"x_d": final_x_d, "x_s": final_x_s}

    def _plot_shap_summary(self, shap_values, inputs):
        """
        Generate and save a SHAP summary plot, separating dynamic and static feature contributions.

        This function visualizes the SHAP values for both dynamic and static features. 
        It aggregates static feature attributions and sums dynamic feature attributions over time.
        The final SHAP summary plot is saved as a PNG file.

        Args:
            shap_values (np.ndarray): SHAP values for all features, shape [n_samples, n_features].
            inputs (dict): A dictionary containing:
                - "x_d" (np.ndarray): Dynamic input values, shape [n_samples, seq_length, n_dynamic].
                - "x_s" (np.ndarray): Static input values, shape [n_samples, n_static].

        Returns:
            None: The function saves the SHAP summary plot as an image file.
        """
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
        combined_static_shap, combined_static_inputs, agg_static_names = self._aggregate_static_features(static_vals, x_s)

        # Sum dynamic features across time
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
        logging.info(f"Summary plot saved to {summary_plot_path}")

    def _plot_shap_contributions_over_time(self, shap_values, last_n_days=None):
        """
        Generate a plot showing the overall contribution of each dynamic feature to the prediction over time.

        Args:
            shap_values (np.ndarray): SHAP values array of shape [n_samples, seq_length * num_dynamic + num_static].
            last_n_days (int, optional): Number of last days to display in the plot. Defaults to seq_length.
        """
        if last_n_days is None:
            last_n_days = self.seq_length

        n_dynamic = len(self.dynamic_features)
        dynamic_shap_values = shap_values[:, :self.seq_length * n_dynamic].reshape(-1, self.seq_length, n_dynamic)
        median_shap_values = np.median(np.abs(dynamic_shap_values), axis=0)  # [seq_length, n_dynamic]

        # Slice the last `last_n_days`
        days_to_plot = min(last_n_days, self.seq_length)
        median_shap_values = median_shap_values[-days_to_plot:]  # Take the last `days_to_plot` days
        time_steps = np.arange(-days_to_plot, 0)  # Adjust x-axis accordingly

        plt.figure(figsize=(15, 8))
        for feature_idx, feature_name in enumerate(self.dynamic_features):
            plt.plot(
                time_steps,
                median_shap_values[:, feature_idx],
                label=f"{feature_name}"
            )

        plt.xticks(np.arange(-days_to_plot, 1, max(10, days_to_plot // 7)))  # Dynamically adjust tick spacing
        plt.title(f"Overall Contribution of Dynamic Features to Prediction Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Mean Absolute SHAP Value")
        plt.legend(loc="upper left", fontsize=10, frameon=True)
        plt.grid(True)

        plot_path = os.path.join(self.results_folder, "shap_overall_contribution_combined.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Saved overall contribution plot to {plot_path}")

    def _plot_shap_summary_bar(self, shap_values):
        """
        Generate a SHAP Summary Bar Plot showing the average absolute SHAP values for each feature.

        Args:
            shap_values (np.ndarray): SHAP values array of shape [n_samples, seq_length * num_dynamic + num_static].
        """
        num_dynamic = len(self.dynamic_features)
        dynamic_shap_values = shap_values[:, :self.seq_length * num_dynamic].reshape(-1, self.seq_length, num_dynamic)
        static_shap_values = shap_values[:, self.seq_length * num_dynamic:]

        # Sum SHAP values across time steps and then average across samples for dynamic features
        dynamic_summed_shap = np.sum(np.abs(dynamic_shap_values), axis=1) # Shape: [n_samples, num_dynamic]
        dynamic_mean_shap = np.mean(dynamic_summed_shap, axis=0) # Shape: [num_dynamic]

        combined_static_shap, agg_static_names = self._aggregate_static_features(static_shap_values)
        static_mean_shap = np.mean(np.abs(combined_static_shap), axis=0).squeeze()

        feature_names = self.dynamic_features + agg_static_names
        mean_shap_values = np.concatenate([dynamic_mean_shap, static_mean_shap])

        # Sort and plot
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_shap_values = mean_shap_values[sorted_indices]

        # Define colors: dynamic features in one color, static features in another.
        dynamic_color = 'skyblue'
        static_color = 'blue'
        colors = [dynamic_color if i < num_dynamic else static_color for i in sorted_indices]

        # Plot the bar chart with individual bar colors
        plt.figure(figsize=(12, 8))
        plt.barh(sorted_features, sorted_shap_values, color=colors)
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title("SHAP Summary Bar Plot")
        plt.gca().invert_yaxis()

        dynamic_patch = mpatches.Patch(color=dynamic_color, label='Dynamic Features')
        static_patch = mpatches.Patch(color=static_color, label='Static Features')
        plt.legend(handles=[dynamic_patch, static_patch], loc='lower right')
        
        plot_path = os.path.join(self.results_folder, "shap_summary_bar_plot.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"SHAP Summary Bar Plot saved to {plot_path}")

    def _plot_shap_summary_bar1(self, shap_values):
        """
        Generate a SHAP Summary Bar Plot showing the average absolute SHAP values for each feature - using shap.plots.bar.
        Args:
            shap_values (np.ndarray): SHAP values array of shape [n_samples, seq_length * num_dynamic + num_static].
        """
        num_dynamic = len(self.dynamic_features)

        # Split into dynamic vs. static
        dynamic_shap_values = shap_values[:, :self.seq_length * num_dynamic].reshape(-1, self.seq_length, num_dynamic)
        static_shap_values = shap_values[:, self.seq_length * num_dynamic:]

        # Sum SHAP values across time steps and then average across samples for dynamic features
        dynamic_summed_shap = np.sum(np.abs(dynamic_shap_values), axis=1)  # [n_samples, num_dynamic]
        dynamic_mean_shap = np.mean(dynamic_summed_shap, axis=0)  # [num_dynamic]

        # Aggregate static features
        combined_static_shap, agg_static_names = self._aggregate_static_features(static_shap_values)
        static_mean_shap = np.mean(np.abs(combined_static_shap), axis=0).squeeze()

        # Create final vectors
        feature_names = self.dynamic_features + agg_static_names
        mean_shap_values = np.concatenate([dynamic_mean_shap, static_mean_shap])  # shape: [num_features]
        
        # Build an Explanation object do:
        aggregated_shap = np.tile(mean_shap_values, (1, 1))  # shape: (1, n_features)
        explanation = shap.Explanation(
            values=aggregated_shap,
            feature_names=feature_names
        )

        plt.figure(figsize=(12, 8))
        shap.plots.bar(explanation, max_display=len(feature_names), show=False)

        plot_path = os.path.join(self.results_folder, "shap_summary_bar_plot.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"SHAP Summary Bar Plot saved to {plot_path}")

    def run_shap_visualizations(self, shap_values, inputs):
        self._plot_shap_summary(shap_values, inputs)
        self._plot_shap_contributions_over_time(shap_values)
        self._plot_shap_summary_bar(shap_values)
        # self._plot_shap_individual_samples(shap_values)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run SHAP analysis.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory.')
    parser.add_argument('--epoch', type=int, required=True, help='Which epoch checkpoint to load.')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples to use for SHAP analysis.')
    parser.add_argument('--reuse_shap', action='store_true',
                        help='If set, reuse shap_values.npy and inputs.npz if they exist, skipping new computation.')

    args = parser.parse_args()
    analysis = SHAPAnalysis(args.run_dir, args.epoch, args.num_samples)

    shap_values_path = os.path.join(analysis.results_folder, "shap_values.npy")
    inputs_path = os.path.join(analysis.results_folder, "inputs.npz")

    if args.reuse_shap:
        if os.path.exists(shap_values_path) and os.path.exists(inputs_path):
            logging.info("Reusing existing SHAP files. Loading from disk...")
            shap_values = np.load(shap_values_path)
            inputs = np.load(inputs_path)
        else:
            raise FileNotFoundError("Reuse flag set, but SHAP files not found. Please run SHAP analysis from scratch.")
    else:
        shap_values, inputs = analysis.run_shap()

    analysis.run_shap_visualizations(shap_values, inputs)


if __name__ == "__main__":
    main()
