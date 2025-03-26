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

from ExplainabilityBase import ExplainabilityBase 

torch.backends.cudnn.enabled = False

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

    def _get_embedding_outputs(self, x_d: np.ndarray, x_s: np.ndarray) -> np.ndarray:
        """
        Run a forward pass with a hook on the embedding network (embedding_net) to extract
        the embedding outputs for each sample and then produce a flattened representation.
        
        Memory-optimized version that processes batches without storing all intermediate outputs.
        
        Args:
            x_d: Dynamic input features (np.ndarray of shape [num_samples, seq_length, n_dynamic])
            x_s: Static input features (np.ndarray of shape [num_samples, n_static])
        
        Returns:
            np.ndarray: Final flattened embedding representation.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_samples = x_d.shape[0]
        n_dynamic = len(self.dynamic_features)
        
        dynamic_embedding = self.cfg.dynamics_embedding
        static_embedding = self.cfg.statics_embedding
        dyn_emb_out_dim = dynamic_embedding["hiddens"][-1] if dynamic_embedding and dynamic_embedding.get("hiddens") else -1
        stat_emb_out_dim = static_embedding["hiddens"][-1] if static_embedding and static_embedding.get("hiddens") else -1
        
        # Preallocate the final output array to avoid memory spikes
        if dyn_emb_out_dim > 0:
            final_dim = (self.seq_length * dyn_emb_out_dim) + stat_emb_out_dim
        else:
            final_dim = (self.seq_length * n_dynamic) + stat_emb_out_dim
            
        final_outputs = np.zeros((num_samples, final_dim), dtype=np.float32)
        
        if dyn_emb_out_dim > 0:
            # --- Model with both dynamic and static embeddings ---
            # Process in batches
            for i in range(0, num_samples, BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, num_samples)
                batch_size = batch_end - i

                batch_x_d = torch.tensor(x_d[i:batch_end], dtype=torch.float32).to(device)
                batch_x_s = torch.tensor(x_s[i:batch_end], dtype=torch.float32).to(device)
                embedding_outputs_list = []
                
                def hook_fn(module: nn.Module, input: Tuple[Tensor], output: Tensor) -> Tensor:
                    embedding_outputs_list.append(output.detach().cpu())
                    return output
                    
                hook_handle = self.model.embedding_net.register_forward_hook(hook_fn)
                
                inputs = {"x_d": batch_x_d, "x_s": batch_x_s}
                with torch.no_grad():
                    _ = self.model(inputs)
                    
                hook_handle.remove()
                
                # Process the collected embeddings for this batch
                combined_embedding = torch.cat(embedding_outputs_list, dim=1)
                # Permute to [batch_size, seq_length, emb_dim]
                emb_out = combined_embedding.permute(1, 0, 2)
                
                # Split into dynamic and static parts
                dynamic_emb = emb_out[:, :, :dyn_emb_out_dim]    # [batch_size, seq_length, dyn_emb_out_dim]
                static_emb = emb_out[:, 0, dyn_emb_out_dim:]     # [batch_size, stat_emb_out_dim]
                
                dynamic_flat = dynamic_emb.reshape(batch_size, -1)
                batch_final_emb = torch.cat([dynamic_flat, static_emb], dim=1).numpy()
                
                # Store directly in the preallocated array
                final_outputs[i:batch_end] = batch_final_emb
                
                # Clear memory
                del batch_x_d, batch_x_s, combined_embedding, emb_out, dynamic_emb, static_emb, dynamic_flat, batch_final_emb
                embedding_outputs_list.clear()
                torch.cuda.empty_cache()
                
        else:
            # --- Model with only static embedding ---
            # In this case, the model does not embed x_d. The static branch processes x_s.
            # We still pass both x_d and x_s to the model, but use a dummy for x_d in the embedding branch.
            # Meanwhile, we want to preserve the original x_d (flattened) for the final representation.
            x_d_flat = x_d.reshape(num_samples, -1)  # Raw dynamic input, flattened

            for i in range(0, num_samples, BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, num_samples)
                batch_size_actual = batch_end - i
                dummy_x_d = torch.zeros(batch_size_actual, self.seq_length, n_dynamic, device=device)
                batch_x_s = torch.tensor(x_s[i: i + batch_size_actual], dtype=torch.float32, device=device)
                embedding_outputs_list = []
                
                def hook_fn(module: nn.Module, input: Tuple[Tensor], output: Tensor) -> Tensor:
                    embedding_outputs_list.append(output.detach().cpu())
                    return output
                    
                hook_handle = self.model.embedding_net.register_forward_hook(hook_fn)
                
                inputs = {"x_d": dummy_x_d, "x_s": batch_x_s}
                with torch.no_grad():
                    _ = self.model(inputs)
                    
                hook_handle.remove()
                
                # Process the collected embeddings for this batch
                combined_embedding = torch.cat(embedding_outputs_list, dim=0)  # shape: [T, batch, total_emb_dim]
                if combined_embedding.dim() == 3:
                    # combined_embedding is [T, batch, total_emb_dim] where total_emb_dim = dyn_emb_out_dim + stat_emb_out_dim
                    full_static = combined_embedding[0]  # take the first time step, shape: [batch, 46]
                    dyn_emb_out_dim = len(self.dynamic_features)
                    # Extract only the static portion: expected shape [batch, 32]
                    static_emb = full_static[:, dyn_emb_out_dim:]
                else:
                    static_emb = combined_embedding
                    
                batch_x_d_flat = torch.tensor(x_d_flat[i:batch_end], dtype=torch.float32)
                batch_final_emb = torch.cat([batch_x_d_flat, static_emb], dim=1).numpy()

                final_outputs[i:batch_end] = batch_final_emb
                
                # Clear memory
                del dummy_x_d, batch_x_s, static_emb, batch_x_d_flat, batch_final_emb
                embedding_outputs_list.clear()
                torch.cuda.empty_cache()

        logging.info(f"Final embedding representation shape: {final_outputs.shape}")
        return final_outputs

    def _wrap_model_embedding(self) -> nn.Module:
        """
        Create a wrapped model that accepts the flattened embedding output as input and maps 
        it to the final prediction. This wrapped model “reconstructs” an input for the downstream
        network by reversing the flattening applied during SHAP extraction.
        
        For models with both dynamic and static embeddings:
        - The flattened input is of shape [batch, (seq_length * dyn_emb_out_dim) + stat_emb_out_dim].
        - It is split into a dynamic part (reshaped to [batch, seq_length, dyn_emb_out_dim]) and a 
            static part (of shape [batch, stat_emb_out_dim]), with the static part repeated along the time axis.
        
        For models with only a static embedding:
        - The flattened input is of shape [batch, (seq_length * num_dynamic) + stat_emb_out_dim],
            where the first part (which would be the raw dynamic input) is ignored.
        - Only the static embedding is used. It is repeated along the time axis so that the 
            downstream network (which expects a time dimension) receives an input of shape 
            [seq_length, batch, stat_emb_out_dim].
        
        Returns:
            nn.Module: A wrapped model that takes an input of shape [batch, flattened_dim] and returns 
                    a final prediction of shape [batch, 1].
        """
        class WrappedModelEmbedding(nn.Module):
            def __init__(self, original_model: nn.Module, cfg) -> None:
                super().__init__()
                self.original_model = original_model
                self.cfg = cfg
                self.num_dynamic = len(self.cfg.dynamic_inputs)  # Number of dynamic (raw) features
                self.num_static = len(self.cfg.static_attributes)    # Number of static (raw) features

                dynamic_embedding = self.cfg.dynamics_embedding
                static_embedding = self.cfg.statics_embedding
                self.dyn_emb_out_dim = dynamic_embedding["hiddens"][-1] if dynamic_embedding and dynamic_embedding.get("hiddens") else -1
                self.stat_emb_out_dim = static_embedding["hiddens"][-1] if static_embedding and static_embedding.get("hiddens") else -1

            def forward(self, embedding_input: Tensor) -> Tensor:
                """
                Reconstruct the embedding input into the shape expected by the downstream network.
                
                Args:
                    embedding_input (Tensor): Flattened embedding representation.
                    - For both-embedding models: shape [batch, (seq_length * dyn_emb_out_dim) + stat_emb_out_dim].
                    - For static-only models: shape [batch, (seq_length * num_dynamic) + stat_emb_out_dim].
                
                Returns:
                    Tensor: Final prediction of shape [batch, 1].
                """
                batch_size = embedding_input.size(0)
                seq_length = self.cfg.seq_length

                if self.dyn_emb_out_dim > 0:
                    # --- Model with both dynamic and static embeddings ---
                    dynamic_flat = embedding_input[:, :seq_length * self.dyn_emb_out_dim]  # [batch, seq_length * dyn_emb_out_dim]
                    static_part = embedding_input[:, seq_length * self.dyn_emb_out_dim:]    # [batch, stat_emb_out_dim]

                    dynamic_emb = dynamic_flat.reshape(batch_size, seq_length, self.dyn_emb_out_dim)
                    # Repeat static embedding along the time axis
                    static_emb = static_part.unsqueeze(1).repeat(1, seq_length, 1)
                    embedding_reconstructed = torch.cat([dynamic_emb, static_emb], dim=2)
                else:
                    # --- Model with only static embedding ---
                    # The flattened input is [batch, (seq_length * num_dynamic) + stat_emb_out_dim]
                    # The first part is the raw dynamic input; the second part is the static embedding.
                    dynamic_part = embedding_input[:, : seq_length * self.num_dynamic]  # [batch, seq_length * num_dynamic]
                    dynamic_part = dynamic_part.reshape(batch_size, seq_length, self.num_dynamic)
                    static_part = embedding_input[:, seq_length * self.num_dynamic:]   # [batch, stat_emb_out_dim]
                    # Replicate the static embedding along the time axis
                    static_emb = static_part.unsqueeze(1).repeat(1, seq_length, 1)
                    # Concatenate the raw dynamic input with the static embedding to reconstruct the full embedding.
                    embedding_reconstructed = torch.cat([dynamic_part, static_emb], dim=2)

                # Permute to [seq_length, batch, features] as expected by the downstream network
                embedding_reconstructed = embedding_reconstructed.permute(1, 0, 2)

                # Override the embedding_net output using a hook
                def hook_fn(module: nn.Module, input: Tuple[Tensor], output: Tensor) -> Tensor:
                    return embedding_reconstructed

                hook_handle = self.original_model.embedding_net.register_forward_hook(hook_fn)

                # Create dummy inputs; their actual values are ignored due to the hook
                dummy_x_d = torch.zeros(batch_size, seq_length, self.num_dynamic, device=embedding_input.device)
                dummy_x_s = torch.zeros(batch_size, self.num_static, device=embedding_input.device)
                inputs = {"x_d": dummy_x_d, "x_s": dummy_x_s}
                out_dict = self.original_model(inputs)
                hook_handle.remove()

                # Return the last time step's prediction
                return out_dict["y_hat"][:, -1, 0].unsqueeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return WrappedModelEmbedding(
            self.model, 
            cfg=self.cfg
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

        # Sample dynamic and static inputs (and record the global sample indices)
        final_x_d, final_x_s, sample_indices = self.load_and_sample_inputs()

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

                if i % (BATCH_SIZE * 10) == 0:
                    torch.cuda.empty_cache()

        shap_values = np.concatenate(shap_values_batches, axis=0)
        logging.info(f"SHAP values shape: {shap_values.shape}")

        # Save results for reuse (naming files differently if embedding is used)
        shap_filename = f"{'shap_values_embedding' if self.use_embedding else 'shap_values'}.npy"
        inputs_filename = f"{'inputs_embedding' if self.use_embedding else 'inputs'}.npz"
        np.save(os.path.join(self.results_folder, shap_filename), shap_values)
        np.savez(os.path.join(self.results_folder, inputs_filename), x_d=final_x_d, x_s=final_x_s)
        np.save(os.path.join(self.results_folder, "sample_indices.npy"), sample_indices)

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

        # Split SHAP values back into dynamic and static parts
        dynamic_shap = shap_values[:, :self.seq_length * n_dynamic].reshape(n_samples, self.seq_length, n_dynamic)
        static_shap = shap_values[:, self.seq_length * n_dynamic:]

        # Sum dynamic features across time
        dynamic_shap = dynamic_shap.sum(axis=1)
        x_d = x_d.sum(axis=1)

        combined_shap = np.concatenate([dynamic_shap, static_shap], axis=1) # shape: [N, D + S]
        combined_inputs = np.concatenate([x_d, x_s], axis=1) # shape: [N, D + S]
        feature_names = self.dynamic_features + self.static_features

        # Compute global clipping thresholds:
        # - Global lower_clip is the minimum of the 0.01st percentiles across all features
        # - Global upper_clip is the maximum of the 99.99th percentiles across all features
        # This removes the most extreme 0.01% from each tail
        lower_clip = min([np.percentile(combined_shap[:, i], 0.01) for i in range(combined_shap.shape[1])])
        upper_clip = max([np.percentile(combined_shap[:, i], 99.99) for i in range(combined_shap.shape[1])])
        combined_shap_clipped = np.clip(combined_shap, lower_clip, upper_clip)

        # Plot
        plt.figure(figsize=(10, 12))
        shap.summary_plot(
            combined_shap_clipped,
            combined_inputs,
            feature_names=feature_names,
            max_display=len(feature_names),
            show=False
        )

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
            # In embedding mode, the flattened input dimensions are: [seq_length * dyn_emb_dim, stat_emb_dim]
            dyn_emb_dim = self.model.embedding_net.dynamics_output_size
            stat_emb_dim = self.model.embedding_net.statics_output_size
            logging.info(f"Embedding dimensions: {dyn_emb_dim} dynamic, {stat_emb_dim} static")
            
            dynamic_shap_values = shap_values[:, :self.seq_length * dyn_emb_dim].reshape(-1, self.seq_length, dyn_emb_dim)
            static_shap_values = shap_values[:, self.seq_length * dyn_emb_dim:]

            # Sum dynamic contributions over time and average over samples
            dynamic_summed_shap = np.sum(dynamic_shap_values, axis=1)  # [n_samples, dyn_emb_dim]
            dynamic_mean_shap = np.mean(np.abs(dynamic_summed_shap), axis=0)  # [dyn_emb_dim]
            static_mean_shap = np.mean(np.abs(static_shap_values), axis=0).squeeze()  # [stat_emb_dim]

            # If dyn_emb_dim == len(self.dynamic_features), then there's no real dynamic "embedding"
            if dyn_emb_dim == len(self.dynamic_features):
                dynamic_feature_names = self.dynamic_features
            else:
                dynamic_feature_names = [f"Embed Dyn {i+1}" for i in range(dyn_emb_dim)]

            static_feature_names = [f"Embed Stat {i+1}" for i in range(stat_emb_dim)]

            feature_names = dynamic_feature_names + static_feature_names
            mean_shap_values = np.concatenate([dynamic_mean_shap, static_mean_shap])

            colors = ['skyblue'] * dyn_emb_dim + ['blue'] * stat_emb_dim
        else:
            num_dynamic = len(self.dynamic_features)
            dynamic_shap_values = shap_values[:, :self.seq_length * num_dynamic].reshape(-1, self.seq_length, num_dynamic)
            static_shap_values = shap_values[:, self.seq_length * num_dynamic:]

            # Sum dynamic contributions across time and average over samples
            dynamic_summed_shap = np.sum(dynamic_shap_values, axis=1) # [n_samples, num_dynamic]
            dynamic_mean_shap = np.mean(np.abs(dynamic_summed_shap), axis=0) # [num_dynamic]
            static_mean_shap = np.mean(np.abs(static_shap_values), axis=0).squeeze()

            feature_names = self.dynamic_features + self.static_features
            mean_shap_values = np.concatenate([dynamic_mean_shap, static_mean_shap])

            colors = ['skyblue'] * len(self.dynamic_features) + ['blue'] * len(self.static_features)

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

        plt.legend(
            handles=[mpatches.Patch(color='skyblue', label='Dynamic' if not self.use_embedding else 'Embedding Dynamic'),
                     mpatches.Patch(color='blue', label='Static' if not self.use_embedding else 'Embedding Static')], 
                     loc='lower right')

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
        shap_filename = f"{'shap_values_embedding' if args.use_embedding else 'shap_values'}.npy"
        inputs_filename = f"{'inputs_embedding' if args.use_embedding else 'inputs'}.npz"
        sample_indices_path = os.path.join(analysis.results_folder, "sample_indices.npy")
        shap_values_path = os.path.join(analysis.results_folder, shap_filename)
        inputs_path = os.path.join(analysis.results_folder, inputs_filename)
        if os.path.exists(shap_values_path) and os.path.exists(inputs_path) and os.path.exists(sample_indices_path):
            logging.info("Reusing existing SHAP files. Loading from disk...")
            shap_values = np.load(shap_values_path, mmap_mode='r')
            inputs = np.load(inputs_path, mmap_mode='r')
            # sample_indices is saved for metadata but not used here directly.
            # sample_indices = np.load(sample_indices_path, mmap_mode='r')
        else:
            raise FileNotFoundError("Reuse flag set, but one or more SHAP files not found. Please run SHAP analysis from scratch.")
    else:
        shap_values, inputs = analysis.run_shap()

    analysis.run_shap_visualizations(shap_values, inputs)


if __name__ == "__main__":
    main()
