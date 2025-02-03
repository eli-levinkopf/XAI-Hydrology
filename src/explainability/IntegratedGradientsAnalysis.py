import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
torch.backends.cudnn.enabled = False

from ExplainabilityBase import ExplainabilityBase 


BACKGROUND_SIZE = 8192
BATCH_SIZE = 256
IG_STEPS = 100 


class IntegratedGradientsAnalysis(ExplainabilityBase):
    def __init__(self, run_dir, epoch, num_samples):
        """
        Initialize IntegratedGradientsAnalysis for a trained NeuralHydrology model.
        
        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number to load the model.
            num_samples (int): Number of samples to use for analysis.
        """
        super().__init__(run_dir, epoch, num_samples, analysis_name="integrated_gradients")

    @staticmethod
    def integrated_gradients(model, baseline, inputs, steps=50):
        """
        Compute Integrated Gradients for a batch of inputs.

        Args:
            model (torch.nn.Module): The wrapped model accepting shape [batch, num_features].
            baseline (torch.Tensor): The baseline input of shape [1, num_features] or [batch, num_features].
            inputs (torch.Tensor): The original inputs of shape [batch, num_features].
            steps (int): Number of interpolation steps.

        Returns:
            torch.Tensor: Integrated gradients w.r.t. each input feature, shape [batch, num_features].
        """
        # 1) If there's a single baseline row but multiple inputs, repeat the baseline
        if baseline.shape[0] == 1 and inputs.shape[0] > 1:
            baseline = baseline.repeat(inputs.shape[0], 1)

        # 2) Prepare a tensor to store gradients at each step and create a list of interpolation alphas from 0 to 1
        grads = torch.zeros(steps, *inputs.shape, device=inputs.device)
        alphas = torch.linspace(0.0, 1.0, steps, device=inputs.device)

        # 3) Loop over these alphas
        for i, alpha in enumerate(alphas):
            # 3a) Interpolate input: x(α) = x0 + α(x - x0)
            interpolated_input = baseline + alpha * (inputs - baseline)
            interpolated_input.requires_grad_(True)

            # 3b) Compute model output for this interpolated input
            output = model(interpolated_input).sum()

            # 3c) Compute gradients of output w.r.t. input
            grad = torch.autograd.grad(
                outputs=output,
                inputs=interpolated_input,
                grad_outputs=torch.ones_like(output),
                create_graph=False
            )[0]
            grads[i] = grad

        # 4) Average gradients across all steps
        avg_grads = grads.mean(dim=0)
        
        # 5) Compute the final IG by multiplying avg grads by (inputs - baseline)
        ig = (inputs - baseline) * avg_grads
        
        return ig

    def run_integrated_gradients(self):
        """
        Main method to compute and save IG values for the model.
        
        Returns:
            ig_values (np.ndarray): The computed IG values of shape [n_samples, n_features].
            final_inputs (dict): {"x_d": final_x_d, "x_s": final_x_s}
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        final_x_d, final_x_s = self._random_sample_from_file()
        combined_inputs = np.hstack([
            final_x_d.reshape(len(final_x_d), -1),
            final_x_s
        ])
        combined_inputs_tensor = torch.tensor(combined_inputs, dtype=torch.float32).to(device)

        wrapped_model = self._wrap_model()

        # Create a baseline by averaging over a random subset
        baseline_indices = np.random.choice(
            range(len(combined_inputs_tensor)), 
            size=min(BACKGROUND_SIZE, len(combined_inputs_tensor)), 
            replace=False
        )
        baseline = combined_inputs_tensor[baseline_indices].mean(dim=0, keepdim=True)

        ig_batches = []
        with tqdm(total=len(combined_inputs_tensor), desc="Calculating Integrated Gradients") as pbar:
            for i in range(0, len(combined_inputs_tensor), BATCH_SIZE):
                batch = combined_inputs_tensor[i : i + BATCH_SIZE]
                ig = self.integrated_gradients(
                    model=wrapped_model,
                    baseline=baseline,
                    inputs=batch,
                    steps=IG_STEPS
                )
                ig_batches.append(ig.detach().cpu().numpy())
                pbar.update(len(batch))

                if i % (BATCH_SIZE * 10) == 0:
                    torch.cuda.empty_cache()

        ig_values = np.concatenate(ig_batches, axis=0)

        # Save results
        np.save(os.path.join(self.results_folder, "ig_values.npy"), ig_values)
        np.savez(os.path.join(self.results_folder, "ig_inputs.npz"), x_d=final_x_d, x_s=final_x_s)

        torch.cuda.empty_cache()
        return ig_values, {"x_d": final_x_d, "x_s": final_x_s}

    def _sum_ig_dynamic_over_time(self, ig_dyn):
        """
        Sum dynamic IG values across the time dimension.
        ig_dyn is shape [n_samples, seq_length * num_dynamic].
        Returns shape [n_samples, num_dynamic].
        """
        n_samples = ig_dyn.shape[0]
        num_dynamic = len(self.dynamic_features)
        ig_dyn_reshaped = ig_dyn.reshape(n_samples, self.seq_length, num_dynamic)
        return ig_dyn_reshaped.sum(axis=1)  # sum over seq_length

    def _plot_ig_bar(self, ig_values, feature_names, title="Overall Feature Contributions", use_abs=True):
        """
        Plot a horizontal bar chart of average Integrated Gradients across samples.

        Args:
            ig_values (np.ndarray): [n_samples, n_features]
            feature_names (list of str): Feature names of length n_features
            title (str): Title for the plot
            use_abs (bool): If True, plot mean absolute IG; else plot mean signed IG.
        """
        if use_abs:
            scores = np.mean(np.abs(ig_values), axis=0)
            x_label = "Mean |IG|"
            plot_suffix = "absolute"
        else:
            scores = np.mean(ig_values, axis=0)
            x_label = "Mean Signed IG"
            plot_suffix = "signed"

        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_idx]
        sorted_feature_names = [feature_names[i] for i in sorted_idx]
        
        if use_abs:
            for feature, value in zip(sorted_feature_names, sorted_scores):
                logging.info(f"{feature}: |{value:.2f}|")

        plt.figure(figsize=(8, 6))
        plt.barh(range(len(sorted_scores)), sorted_scores, color='skyblue')
        plt.yticks(range(len(sorted_scores)), sorted_feature_names)
        plt.xlabel(x_label)
        plt.title(title)
        plt.tight_layout()
        plt.gca().invert_yaxis()

        plot_path = os.path.join(self.results_folder, f"IG_summary_bar_plot_{plot_suffix}.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"IG Summary Bar Plot ({plot_suffix}) saved to {plot_path}")

    def run_ig_visualizations(self, ig_values, inputs):
        """
        Produce an overall bar chart of dynamic + static feature contributions.
        1) Separate dynamic vs. static
        2) Sum dynamic IG across time
        3) Aggregate static features
        4) Plot bar charts
        """
        num_dyn_features = len(self.dynamic_features)
        ig_dyn = ig_values[:, : self.seq_length * num_dyn_features]
        ig_stat = ig_values[:, self.seq_length * num_dyn_features :]

        # Sum dynamic across time
        summed_ig_dyn = self._sum_ig_dynamic_over_time(ig_dyn)

        # Aggregate static
        aggregated_ig_stat, agg_stat_names = self._aggregate_static_features(ig_stat)

        # Combine
        combined_ig = np.hstack([summed_ig_dyn, aggregated_ig_stat])
        combined_feature_names = self.dynamic_features + agg_stat_names

        # Plot bar chart of mean absolute IG
        self._plot_ig_bar(
            ig_values=combined_ig,
            feature_names=combined_feature_names,
            title="Overall Feature Contributions (IG) - absolute",
            use_abs=True
        )

        # Plot bar chart of mean signed IG
        self._plot_ig_bar(
            ig_values=combined_ig,
            feature_names=combined_feature_names,
            title="Overall Feature Contributions (IG) - signed",
            use_abs=False
        )

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run Integrated Gradients analysis.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory.')
    parser.add_argument('--epoch', type=int, required=True, help='Which epoch checkpoint to load.')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples to use for IG analysis.')
    parser.add_argument('--reuse_ig', action='store_true',
                        help='If set, reuse ig_values.npy and ig_inputs.npz if they exist.')

    args = parser.parse_args()
    analysis = IntegratedGradientsAnalysis(args.run_dir, args.epoch, args.num_samples)

    ig_values_path = os.path.join(analysis.results_folder, "ig_values.npy")
    inputs_path = os.path.join(analysis.results_folder, "ig_inputs.npz")

    if args.reuse_ig:
        if os.path.exists(ig_values_path) and os.path.exists(inputs_path):
            logging.info("Reusing existing IG files. Loading from disk...")
            ig_values = np.load(ig_values_path)
            inputs = np.load(inputs_path)
        else:
            raise FileNotFoundError("reuse-ig flag set, but IG files not found. "
                                    "Please run IG analysis from scratch.")
    else:
        ig_values, inputs = analysis.run_integrated_gradients()

    analysis.run_ig_visualizations(ig_values, inputs)


if __name__ == "__main__":
    main()
