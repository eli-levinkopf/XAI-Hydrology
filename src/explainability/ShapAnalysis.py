from typing import Dict, Tuple, Optional, Union
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import shap
import logging
from torch import Tensor, nn
from pathlib import Path
import matplotlib.cm as cm
from itertools import cycle

torch.backends.cudnn.enabled = False

from ExplainabilityBase import ExplainabilityBase 

BACKGROUND_SIZE = 1024
BATCH_SIZE = 256


class SHAPAnalysis(ExplainabilityBase):
    def __init__(self, run_dir: Union[str, Path], epoch: int, num_samples: int = 100000, period: str = "test", use_embedding: bool = False) -> None:
        """
        Initialize SHAPAnalysis for a trained NeuralHydrology model.
        
        Args:
            run_dir: Path to the run directory.
            epoch: Epoch number to load the model.
            num_samples: Number of samples for SHAP analysis.
            use_embedding: If True, run SHAP on the embedding outputs.
            period: The period to load data from ("train", "validation", or "test").
        """
        super().__init__(run_dir, epoch, num_samples, analysis_name="shap", period=period)
        self.use_embedding = use_embedding

    def _get_embedding_outputs(self, final_x_d: np.ndarray, final_x_s: np.ndarray) -> np.ndarray:
        """
        Run a forward pass with a hook on the embedding network (embedding_net) in batches to extract 
        the embedding outputs for each sample. Then, process the raw output to obtain a 
        flattened representation.

        Expected raw output from embedding_net: [seq_length, num_samples, dynamic_emb_dim + static_emb_dim],
        where:
          - the first dynamic_emb_dim features come from the dynamic embedding (one per time step)
          - the next static_emb_dim features come from the static embedding (repeated over time)

        We convert this into a flattened representation of shape:
          [num_samples, (seq_length * dynamic_emb_dim) + static_emb_dim]

       Args:
            final_x_d: Dynamic input features
            final_x_s: Static input features
        
        Returns:
            np.ndarray: Final flattened embedding representation.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Recombine inputs as in run_shap:
        combined_inputs = np.hstack([
            final_x_d.reshape(len(final_x_d), -1),
            final_x_s
        ])
        combined_inputs_tensor = torch.tensor(combined_inputs, dtype=torch.float32).to(device)
        
        num_dynamic = len(self.dynamic_features)
        seq_length = self.seq_length
        x_d_flat = combined_inputs_tensor[:, : seq_length * num_dynamic]
        x_s = combined_inputs_tensor[:, seq_length * num_dynamic:]
        x_d = x_d_flat.view(-1, seq_length, num_dynamic)

        # Prepare to collect embedding outputs from each batch
        embedding_outputs_list = []

        def hook_fn(module: nn.Module, input: Tuple[Tensor], output: Tensor) -> Tensor:
            # Append the output from this batch
            embedding_outputs_list.append(output.detach())
            return output

        # Attach hook to the embedding_net
        hook_handle = self.model.embedding_net.register_forward_hook(hook_fn)

        # Process the inputs in batches to avoid a huge allocation
        num_samples = x_d.shape[0]
        for i in range(0, num_samples, BATCH_SIZE):
            batch_x_d = x_d[i: i + BATCH_SIZE]
            batch_x_s = x_s[i: i + BATCH_SIZE]
            inputs = {"x_d": batch_x_d, "x_s": batch_x_s}
            with torch.no_grad():
                _ = self.model(inputs)
            # Optionally clear the cache after each batch:
            torch.cuda.empty_cache()

        hook_handle.remove()

        # Each element in embedding_outputs_list is of shape: [seq_length, batch, emb_dim]
        # Concatenate along the batch dimension (dim=1)
        combined_embedding = torch.cat(embedding_outputs_list, dim=1)

        # Process the combined embeddings
        emb_out = combined_embedding.cpu()
        logging.info(f"Captured raw embedding output shape: {emb_out.shape}")

        # Permute to shape [num_samples, seq_length, emb_dim]
        emb_out = emb_out.permute(1, 0, 2)
        logging.info(f"After permuting, embedding shape: {emb_out.shape}")

        # Retrieve embedding dimensions from the model configuration
        dyn_emb_dim = self.model.embedding_net.dynamics_output_size
        stat_emb_dim = self.model.embedding_net.statics_output_size

        # Split dynamic and static parts:
        dynamic_emb = emb_out[:, :, :dyn_emb_dim]    # shape: [num_samples, seq_length, dyn_emb_dim]
        static_emb = emb_out[:, 0, dyn_emb_dim:]     # shape: [num_samples, stat_emb_dim]
        logging.info(f"Dynamic part shape: {dynamic_emb.shape}, Static part shape: {static_emb.shape}")

        # Flatten the dynamic part over time: [num_samples, seq_length * dyn_emb_dim]
        dynamic_flat = dynamic_emb.reshape(dynamic_emb.shape[0], -1)
        logging.info(f"Flattened dynamic part shape: {dynamic_flat.shape}")

        # Concatenate to get final embedding representation: [num_samples, (seq_length * dyn_emb_dim) + stat_emb_dim]
        final_emb = torch.cat([dynamic_flat, static_emb], dim=1)
        logging.info(f"Final embedding representation shape: {final_emb.shape}")
        logging.debug(f"Sample final embedding values (first 2 samples): {final_emb[:2].numpy()}")
    
        return final_emb.numpy()

    def _wrap_model_embedding(self)-> nn.Module:
        """
        Create a wrapped model that accepts the flattened embedding output as input and maps 
        it to the final prediction. We override the embedding network's output using a hook.
        
        Since the rest of the model (e.g., the LSTM) expects the embedding_net output to be of shape
        [seq_length, batch, (dyn_emb_dim + stat_emb_dim)], we need to reverse the flattening transformation:
          - Split the flattened input into a dynamic part of shape [batch, seq_length, dyn_emb_dim] and 
            a static part of shape [batch, stat_emb_dim].
          - Replicate the static part along the time dimension and concatenate with the dynamic part.
          - Permute the result to get [seq_length, batch, (dyn_emb_dim + stat_emb_dim)].
        
        Returns:
            torch.nn.Module: Wrapped model that takes an input of shape [batch, flattened_dim].
        """
        class WrappedModelEmbedding(torch.nn.Module):
            def __init__(self, original_model: nn.Module, seq_length: int, num_dynamic: int, num_static: int) -> None:
                super().__init__()
                self.original_model = original_model
                self.seq_length = seq_length
                self.num_dynamic = num_dynamic    # original number of dynamic features (used only for dummy inputs)
                self.num_static = num_static      # original number of static features (used only for dummy inputs)
                self.dyn_emb_dim = self.original_model.embedding_net.dynamics_output_size
                self.stat_emb_dim = self.original_model.embedding_net.statics_output_size

            def forward(self, embedding_input: Tensor) -> Tensor:
                """
                Args:
                    embedding_input (torch.Tensor): Tensor of shape [batch, flattened_dim],
                      where flattened_dim = (seq_length * dyn_emb_dim) + stat_emb_dim.
                Returns:
                    torch.Tensor: Final prediction of shape [batch, 1].
                """
                batch_size = embedding_input.size(0)
                seq_length = self.seq_length

                # Split dynamic vs static
                dynamic_flat = embedding_input[:, :seq_length * self.dyn_emb_dim] # [batch, seq_length * dyn_emb_dim]
                static_part = embedding_input[:, seq_length * self.dyn_emb_dim:] #[batch, stat_emb_dim]

                # Reshape dynamic => [batch, seq_length, dyn_emb_dim]
                dynamic_emb = dynamic_flat.reshape(batch_size, seq_length, self.dyn_emb_dim)
                # Repeat static along the time axis => [batch, 1, stat_emb_dim] -> [batch, seq_length, stat_emb_dim]
                static_emb = static_part.unsqueeze(1).repeat(1, seq_length, 1)

                # Concatenate => [batch, seq_length, dyn_emb_dim + stat_emb_dim]
                embedding_reconstructed = torch.cat([dynamic_emb, static_emb], dim=2)
                # Permute => [seq_length, batch, dyn_emb_dim + stat_emb_dim]
                embedding_reconstructed = embedding_reconstructed.permute(1, 0, 2)

                # Define a hook that overrides the embedding_net output with the reconstructed embedding
                def hook_fn(module: nn.Module, input: Tuple[Tensor], output: Tensor) -> Tensor:
                    return embedding_reconstructed
                
                # Attach the hook
                hook_handle = self.original_model.embedding_net.register_forward_hook(hook_fn)

                # Create dummy inputs (their values will be ignored due to the hook)
                dummy_x_d = torch.zeros(batch_size, self.seq_length, self.num_dynamic, device=embedding_input.device)
                dummy_x_s = torch.zeros(batch_size, self.num_static, device=embedding_input.device)
                inputs = {"x_d": dummy_x_d, "x_s": dummy_x_s}
                out_dict = self.original_model(inputs)
                hook_handle.remove()

                # Assume the model output is in out_dict["y_hat"]; take the last time step's prediction.
                return out_dict["y_hat"][:, -1, 0].unsqueeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return WrappedModelEmbedding(
            self.model, 
            seq_length=self.seq_length,
            num_dynamic=len(self.dynamic_features),
            num_static=len(self.static_features)
        ).to(device).eval()

    def run_shap(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Main method to compute and save SHAP values for the model.
        Depending on self.use_embedding, SHAP is run either on the original combined inputs
        or on the embedding outputs.
        
        Returns:
            shap_values (np.ndarray): The computed SHAP values.
            final_inputs (dict): {"x_d": final_x_d, "x_s": final_x_s}
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Sample dynamic and static inputs
        final_x_d, final_x_s = self._random_sample_from_file()

        if self.use_embedding:
            # Get the embedding outputs from embedding_net, then process to flatten.
            shap_inputs_array = self._get_embedding_outputs(final_x_d, final_x_s)
            # Now shap_inputs_array is of shape [num_samples, (seq_length * dyn_emb_dim) + stat_emb_dim]
            combined_inputs_tensor = torch.tensor(shap_inputs_array, dtype=torch.float32).to(device)
            wrapped_model = self._wrap_model_embedding()
        else:
            # Combine x_d and x_s into a single tensor for SHAP
            combined_inputs = np.hstack([
                final_x_d.reshape(len(final_x_d), -1),
                final_x_s
            ])
            combined_inputs_tensor = torch.tensor(combined_inputs, dtype=torch.float32).to(device)
            wrapped_model = self._wrap_model()

        logging.info(f"Combined inputs shape: {combined_inputs_tensor.shape}")
        
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
        logging.info(f"SHAP values shape: {shap_values.shape}")

        # Save results for reuse (naming files differently if embedding is used)
        np.save(os.path.join(self.results_folder, f"{'shap_values_embedding' if self.use_embedding else 'shap_values'}.npy"), shap_values)
        np.savez(os.path.join(self.results_folder, f"{'inputs_embedding' if self.use_embedding else 'inputs'}.npz"), x_d=final_x_d, x_s=final_x_s)

        # Clean up
        torch.cuda.empty_cache()

        return shap_values, {"x_d": final_x_d, "x_s": final_x_s}

    def _plot_shap_summary(self, shap_values: np.ndarray, inputs: Dict[str, np.ndarray]) -> None:
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

    def _plot_shap_contributions_over_time(self, shap_values: np.ndarray, last_n_days: Optional[int] = None) -> None:
        """
        Generate a plot showing the overall contribution of each dynamic feature to the prediction over time.
        If use_embedding is True, the dynamic features correspond to the dynamic embedding dimensions.
        
        Args:
            shap_values (np.ndarray): SHAP values array.
            last_n_days (int, optional): Number of last days to display in the plot. Defaults to seq_length.
        """
        if last_n_days is None:
            last_n_days = self.seq_length

        if self.use_embedding:
            # In embedding mode, the dynamic part is given by the dynamic embedding output.
            dyn_emb_dim = self.model.embedding_net.dynamics_output_size
            dynamic_shap_values = shap_values[:, :self.seq_length * dyn_emb_dim].reshape(-1, self.seq_length, dyn_emb_dim)
            median_shap_values = np.median(np.abs(dynamic_shap_values), axis=0)  # [seq_length, dyn_emb_dim]
            feature_names = [f"Embed {i+1}" for i in range(dyn_emb_dim)]
        else:
            # Use original dynamic features.
            n_dynamic = len(self.dynamic_features)
            dynamic_shap_values = shap_values[:, :self.seq_length * n_dynamic].reshape(-1, self.seq_length, n_dynamic)
            median_shap_values = np.median(np.abs(dynamic_shap_values), axis=0)  # [seq_length, n_dynamic]
            feature_names = self.dynamic_features

        # Slice to get only the last 'last_n_days'
        days_to_plot = min(last_n_days, self.seq_length)
        median_shap_values = median_shap_values[-days_to_plot:]  # Take the last days_to_plot days
        time_steps = np.arange(-days_to_plot, 0)

        custom_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
            "#ffbb78", "#8c564b", "#98df8a", "#7f7f7f", 
            "#bcbd22", "#17becf", "#aec7e8", "#9467bd", 
            "#e377c2", "#f7b6d2", "#c5b0d5", "#c49c94"
        ]
        line_styles = ["-", "--"]

        plt.figure(figsize=(12, 10))
        for feature_idx, feature_name in enumerate(feature_names):
            color = custom_colors[feature_idx % 16]
            line_style = line_styles[(feature_idx // 16) % 2]

            plt.plot(time_steps, median_shap_values[:, feature_idx], label=feature_name, color=color, linestyle=line_style)

        plt.xticks(np.arange(-days_to_plot, 1, max(10, days_to_plot // 7)))
        plt.title("Overall Contribution of {} Features to Prediction Over Time".format(
            "Embedding Dynamic" if self.use_embedding else "Dynamic"))
        plt.xlabel("Time Step")
        plt.ylabel("Mean Absolute SHAP Value")
        plt.legend(loc="upper left", fontsize=10, frameon=True)
        plt.grid(True)

        plot_path = os.path.join(self.results_folder, "shap_overall_contribution_combined{}".format(
            "_Embedding.png" if self.use_embedding else ".png"))
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Saved overall contribution plot to {plot_path}")

    def _plot_shap_summary_bar(self, shap_values: np.ndarray) -> None:
        """
        Generate a SHAP Summary Bar Plot showing the average absolute SHAP values for each feature.
        Supports both original inputs and embedding outputs.
        
        Args:
            shap_values (np.ndarray): SHAP values array.
        """
        if self.use_embedding:
            # In embedding mode, the flattened input dimensions are:
            # [seq_length * dyn_emb_dim, stat_emb_dim]
            dyn_emb_dim = self.model.embedding_net.dynamics_output_size
            stat_emb_dim = self.model.embedding_net.statics_output_size

            dynamic_shap_values = shap_values[:, :self.seq_length * dyn_emb_dim].reshape(-1, self.seq_length, dyn_emb_dim)
            static_shap_values = shap_values[:, self.seq_length * dyn_emb_dim:]

            # Sum dynamic contributions over time and average over samples
            dynamic_summed_shap = np.sum(np.abs(dynamic_shap_values), axis=1)  # [n_samples, dyn_emb_dim]
            dynamic_mean_shap = np.mean(dynamic_summed_shap, axis=0)  # [dyn_emb_dim]
            static_mean_shap = np.mean(np.abs(static_shap_values), axis=0).squeeze()  # [stat_emb_dim]

            feature_names = [f"Embed Dyn {i+1}" for i in range(dyn_emb_dim)] + \
                            [f"Embed Stat {i+1}" for i in range(stat_emb_dim)]
            mean_shap_values = np.concatenate([dynamic_mean_shap, static_mean_shap])

            dynamic_color = 'skyblue'
            static_color = 'blue'
            colors = [dynamic_color] * dyn_emb_dim + [static_color] * stat_emb_dim
        else:
            num_dynamic = len(self.dynamic_features)
            dynamic_shap_values = shap_values[:, :self.seq_length * num_dynamic].reshape(-1, self.seq_length, num_dynamic)
            static_shap_values = shap_values[:, self.seq_length * num_dynamic:]

            # Sum dynamic contributions across time and average over samples
            dynamic_summed_shap = np.sum(np.abs(dynamic_shap_values), axis=1)  # [n_samples, num_dynamic]
            dynamic_mean_shap = np.mean(dynamic_summed_shap, axis=0)  # [num_dynamic]

            combined_static_shap, agg_static_names = self._aggregate_static_features(static_shap_values)
            static_mean_shap = np.mean(np.abs(combined_static_shap), axis=0).squeeze()

            feature_names = self.dynamic_features + agg_static_names
            mean_shap_values = np.concatenate([dynamic_mean_shap, static_mean_shap])

            # Colors for original features.
            dynamic_color = 'skyblue'
            static_color = 'blue'
            colors = [dynamic_color] * len(self.dynamic_features) + [static_color] * len(agg_static_names)

        # Sort features by mean absolute SHAP value
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_shap_values = mean_shap_values[sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]

        plt.figure(figsize=(10, 12))
        plt.barh(sorted_features, sorted_shap_values, color=sorted_colors)
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title("SHAP Summary Bar Plot")
        plt.gca().invert_yaxis()

        dynamic_patch = mpatches.Patch(color=dynamic_color, label='Embedding Dynamic' if self.use_embedding else 'Dynamic')
        static_patch = mpatches.Patch(color=static_color, label='Embedding Static' if self.use_embedding else 'Static')
        plt.legend(handles=[dynamic_patch, static_patch], loc='lower right')

        plot_path = os.path.join(self.results_folder, "shap_summary_bar_plot{}".format(
            "_Embedding.png" if self.use_embedding else ".png"))
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"SHAP Summary Bar Plot saved to {plot_path}")

    def run_shap_visualizations(self, shap_values: np.ndarray, inputs: Dict[str, np.ndarray]) -> None:
        if not self.use_embedding:
            self._plot_shap_summary(shap_values, inputs)
        self._plot_shap_contributions_over_time(shap_values)
        self._plot_shap_summary_bar(shap_values)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run SHAP analysis.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory.')
    parser.add_argument('--epoch', type=int, required=True, help='Which epoch checkpoint to load.')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples to use for SHAP analysis.')
    parser.add_argument('--period', type=str, default="test", help='Period to load data from (train/validation/test).')
    parser.add_argument('--reuse_shap', action='store_true', help='If set, reuse existing SHAP results.')
    parser.add_argument('--use_embedding', action='store_true', help='If set, run SHAP on embedding outputs.')

    args = parser.parse_args()
    analysis = SHAPAnalysis(args.run_dir, args.epoch, args.num_samples, period=args.period, use_embedding=args.use_embedding)

    if args.reuse_shap:
        shap_values_path = os.path.join(analysis.results_folder, f"{'shap_values_embedding' if args.use_embedding else 'shap_values'}.npy")
        inputs_path = os.path.join(analysis.results_folder, f"{'inputs_embedding' if args.use_embedding else 'inputs'}.npz")
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
