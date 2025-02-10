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
    def __init__(self, run_dir, epoch, num_samples, use_embedding=False):
        """
        Initialize SHAPAnalysis for a trained NeuralHydrology model.
        
        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number to load the model.
            num_samples (int): Number of samples for SHAP analysis.
            use_embedding (bool): If True, run SHAP on the embedding output.
        """
        super().__init__(run_dir, epoch, num_samples, analysis_name="shap")
        self.use_embedding = use_embedding

    def _get_embedding_outputs(self, final_x_d, final_x_s):
        """
        Run a forward pass with a hook on the embedding network (embedding_net) to extract 
        the embedding outputs for each sample. Then, process the raw output to obtain a 
        flattened representation.

        Expected raw output from embedding_net: [seq_length, batch, dynamic_emb_dim + static_emb_dim],
        where:
          - the first dynamic_emb_dim features come from the dynamic embedding (one per time step)
          - the next static_emb_dim features come from the static embedding (repeated over time)

        We convert this into a flattened representation of shape:
          [batch, (seq_length * dynamic_emb_dim) + static_emb_dim]
        
        Returns:
            np.ndarray: Final flattened embedding representation.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Recombine inputs
        combined_inputs = np.hstack([
            final_x_d.reshape(len(final_x_d), -1),
            final_x_s
        ])
        combined_inputs_tensor = torch.tensor(combined_inputs, dtype=torch.float32).to(device)
        # Split combined_inputs_tensor into x_d and x_s
        num_dynamic = len(self.dynamic_features)
        seq_length = self.seq_length
        x_d_flat = combined_inputs_tensor[:, : seq_length * num_dynamic]
        x_s = combined_inputs_tensor[:, seq_length * num_dynamic:]
        x_d = x_d_flat.view(-1, seq_length, num_dynamic)
        inputs = {"x_d": x_d, "x_s": x_s}

        embedding_outputs = []

        def hook_fn(module, input, output):
            embedding_outputs.append(output.detach())
            return output

        # Attach hook to embedding_net
        hook_handle = self.model.embedding_net.register_forward_hook(hook_fn)
        _ = self.model(inputs)  # Run one forward pass so the hook is called.
        hook_handle.remove()

        # embedding_outputs[0] has shape [seq_length, batch, total_emb_dim]
        emb_out = embedding_outputs[0].cpu()
        logging.info(f"Captured raw embedding output shape: {emb_out.shape}")

        # Permute to [batch, seq_length, total_emb_dim]
        emb_out = emb_out.permute(1, 0, 2)
        logging.info(f"After permuting, embedding shape: {emb_out.shape}")

        # Split dynamic and static parts:
        #   dynamic -> first dynamic_emb_dim features
        #   static  -> last static_emb_dim features
        dynamic_emb = emb_out[:, :, : self.dynamic_emb_dim]   # [batch, seq_length, dynamic_emb_dim]
        static_emb = emb_out[:, 0, self.dynamic_emb_dim:]     # [batch, static_emb_dim]
        logging.info(f"Dynamic part shape: {dynamic_emb.shape}, Static part shape: {static_emb.shape}")

        # Flatten dynamic part over time:
        dynamic_flat = dynamic_emb.reshape(dynamic_emb.shape[0], -1)  # [batch, seq_length*dynamic_emb_dim]

        final_emb = torch.cat([dynamic_flat, static_emb], dim=1)  # [batch, seq_length*dynamic_emb_dim + static_emb_dim]
        logging.info(f"Final embedding representation shape: {final_emb.shape}")

        return final_emb.numpy()

    def _wrap_model_embedding(self):
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
            def __init__(self, original_model, seq_length, dynamic_emb_dim, static_emb_dim):
                super().__init__()
                self.original_model = original_model
                self.seq_length = seq_length
                self.dynamic_emb_dim = dynamic_emb_dim
                self.static_emb_dim = static_emb_dim

            def forward(self, embedding_input):
                """
                Args:
                    embedding_input (torch.Tensor): Tensor of shape [batch, flattened_dim],
                      where flattened_dim = (seq_length * dyn_emb_dim) + stat_emb_dim.
                Returns:
                    torch.Tensor: Final prediction of shape [batch, 1].
                """
                batch_size = embedding_input.size(0)

                # 1) Split dynamic vs static
                dynamic_flat = embedding_input[:, :self.seq_length * self.dynamic_emb_dim]   # shape [batch, seq_length * dynamic_emb_dim]
                static_part = embedding_input[:, self.seq_length * self.dynamic_emb_dim:]    # shape [batch, static_emb_dim]

                # 2) Reshape dynamic => [batch, seq_length, dynamic_emb_dim]
                dynamic_emb = dynamic_flat.reshape(batch_size, self.seq_length, self.dynamic_emb_dim)

                # 3) Replicate static_part along the time axis => [batch, 1, static_emb_dim] -> [batch, seq_length, static_emb_dim]
                static_emb = static_part.unsqueeze(1).repeat(1, self.seq_length, 1)
                
                # 4) Concatenate along the feature dimension => [batch, seq_length, total_emb_dim]
                #    Permute => [seq_length, batch, total_emb_dim]
                embedding_reconstructed = torch.cat([dynamic_emb, static_emb], dim=2)
                embedding_reconstructed = embedding_reconstructed.permute(1, 0, 2)

                def hook_fn(module, input, output):
                    return embedding_reconstructed
                
                # Attach the hook
                hook_handle = self.original_model.embedding_net.register_forward_hook(hook_fn)

                # Dummy inputs (their values will be ignored due to the hook)
                dummy_x_d = torch.zeros(batch_size, self.seq_length, 1, device=embedding_input.device)
                dummy_x_s = torch.zeros(batch_size, 1, device=embedding_input.device)
                out_dict = self.original_model({"x_d": dummy_x_d, "x_s": dummy_x_s})
                
                hook_handle.remove()
                return out_dict["y_hat"][:, -1, 0].unsqueeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return WrappedModelEmbedding(
            self.model, 
            seq_length=self.seq_length,
            dynamic_emb_dim=self.dynamic_emb_dim,
            static_emb_dim=self.static_emb_dim
        ).to(device).eval()

    def run_shap(self):
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

        # logging.info("Model's modules:")
        # for name, module in self.model.named_modules():
        #     logging.info(f"Module: {name} --> {module}")

        # Sample dynamic and static inputs
        final_x_d, final_x_s = self._random_sample_from_file()

        if self.use_embedding:
            # Get the embedding outputs from embedding_net, then process to flatten
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

        # Reference (background) data for SHAP
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

        # Save results
        fn_prefix = "embedding" if self.use_embedding else ""
        # np.save(os.path.join(self.results_folder, f"shap_values_{fn_prefix}.npy"), shap_values)
        # np.savez(os.path.join(self.results_folder, f"inputs_{fn_prefix}.npz"), x_d=final_x_d, x_s=final_x_s)

        # Clean up
        torch.cuda.empty_cache()

        return shap_values, {"x_d": final_x_d, "x_s": final_x_s}

    def _plot_shap_summary_embedding(self, shap_values):
        """
        If `use_embedding` is True, shap_values has shape:
          [n_samples, seq_length*dynamic_emb_dim + static_emb_dim]
        We'll treat each embedding dimension as a "feature."
        
        We can replicate the same idea: 
          - dynamic part => [n_samples, seq_length*dynamic_emb_dim] => reshape => [n_samples, seq_length, dynamic_emb_dim]
          - static part => [n_samples, static_emb_dim]
        
        Then, for a "summary," we might sum over time for the dynamic part, resulting in shape [n_samples, dynamic_emb_dim].
        We'll label them as 'dyn_emb_0, ..., dyn_emb_dim-1' for dynamic and 'stat_emb_0, ..., stat_emb_dim-1' for static.
        """
        shap_values.squeeze(-1)
        n_samples = shap_values.shape[0]
        seq_length = self.seq_length
        dyn_dim = self.dynamic_emb_dim
        stat_dim = self.static_emb_dim

        dyn_size = seq_length * dyn_dim

        # 1) Split
        dynamic_vals = shap_values[:, :dyn_size]  # shape [n_samples, seq_length*dyn_dim]
        static_vals = shap_values[:, dyn_size:]   # shape [n_samples, stat_dim]

        # 2) Reshape dynamic part => [n_samples, seq_length, dyn_dim]
        dynamic_vals_3d = dynamic_vals.reshape(n_samples, seq_length, dyn_dim)

        # 3) Sum over time for the dynamic part
        dynamic_summed = dynamic_vals_3d.sum(axis=1)  # shape [n_samples, dyn_dim]

        # Combine
        combined_shap = np.concatenate([dynamic_summed, static_vals], axis=1)  # [n_samples, dyn_dim + stat_dim]

        # Build simple numeric feature names
        dyn_feature_names = [f"dyn_emb_{i}" for i in range(dyn_dim)]
        stat_feature_names = [f"stat_emb_{i}" for i in range(stat_dim)]
        feature_names = dyn_feature_names + stat_feature_names

        # For the data values themselves, we can't easily revert to original features.
        # But if you want a reference, we can build dummy inputs too. For the summary plot,
        # you can pass `None` or some placeholder. We'll pass zeros as a placeholder here:
        placeholder_data = np.zeros_like(combined_shap)

        # Plot
        plt.figure(figsize=(10, 12))
        shap.summary_plot(
            combined_shap,
            placeholder_data,
            feature_names=feature_names,
            max_display=len(feature_names),
            show=False
        )
        plt.xlim([np.min(combined_shap), np.max(combined_shap)])
        summary_plot_path = os.path.join(self.results_folder, "shap_summary_plot_embedding.png")
        plt.savefig(summary_plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Embedding-based SHAP summary plot saved to {summary_plot_path}")

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

        # Slice the last last_n_days
        days_to_plot = min(last_n_days, self.seq_length)
        median_shap_values = median_shap_values[-days_to_plot:]  # Take the last days_to_plot days
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
        if self.use_embedding:
            # We do a separate plotting approach for embedding-based SHAP.
            self._plot_shap_summary_embedding(shap_values)
            # You could add similar "over_time" or "summary_bar" variants specialized for embeddings.
        else:
            # Original approach for raw features
            self._plot_shap_summary(shap_values, inputs)
            self._plot_shap_contributions_over_time(shap_values)
            self._plot_shap_summary_bar(shap_values)
            # self._plot_shap_individual_samples(shap_values)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run SHAP analysis.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory.')
    parser.add_argument('--epoch', type=int, required=True, help='Which epoch checkpoint to load.')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples to use for SHAP analysis.')
    parser.add_argument('--reuse_shap', action='store_true', help='If set, reuse existing SHAP results.')
    parser.add_argument('--use_embedding', action='store_true', help='If set, run SHAP on embedding outputs.')

    args = parser.parse_args()
    analysis = SHAPAnalysis(args.run_dir, args.epoch, args.num_samples, use_embedding=args.use_embedding)

    if args.reuse_shap:
        fn_prefix = "embedding" if args.use_embedding else ""
        shap_values_path = os.path.join(analysis.results_folder, f"shap_values_{fn_prefix}.npy")
        inputs_path = os.path.join(analysis.results_folder, f"inputs_{fn_prefix}.npz")
        if os.path.exists(shap_values_path) and os.path.exists(inputs_path):
            logging.info("Reusing existing SHAP files...")
            shap_values = np.load(shap_values_path)
            inputs = np.load(inputs_path)
        else:
            raise FileNotFoundError("Reuse flag set, but files not found. Please run SHAP analysis first.")
    else:
        shap_values, inputs = analysis.run_shap()

    # analysis.run_shap_visualizations(shap_values, inputs)


if __name__ == "__main__":
    main()
