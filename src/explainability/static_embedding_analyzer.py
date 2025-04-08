import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from model.model_analyzer import ModelAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StaticEmbeddingAnalyzer:
    """
    Analyzes the static embedding layer of a NeuralHydrology model.

    Performs three main analyses:
    1. Visualizes the effective weights of the embedding layer (optionally phased).
    2. Visualizes the pairwise similarity between embedding dimensions based on weights.
    3. Visualizes the correlation between embedding outputs and original static features.
    """
    def __init__(self, run_dir: Path, epoch: int, period: str = "test"):
        self.run_dir = Path(run_dir)
        self.epoch = epoch
        self.period = period
        self.results_dir = self.run_dir / self.period / f"model_epoch{self.epoch:03d}" / "static_embedding_analysis"
        os.makedirs(self.results_dir, exist_ok=True)

        self.analyzer = ModelAnalyzer(run_dir=self.run_dir, epoch=self.epoch, period=self.period)

        self.cfg = self.analyzer.cfg
        self.static_features = sorted(self.cfg.static_attributes)
        self.n_static_features = len(self.static_features)

        self.W1_stat: torch.Tensor | None = None
        self.B1_stat: torch.Tensor | None = None
        self.W2_stat: torch.Tensor | None = None
        self.B2_stat: torch.Tensor | None = None
        self.embedding_layer: nn.Sequential | None = None
        self.df_effective_weights: pd.DataFrame | None = None
        self.n_embedding_dims: int | None = None

        self._load_weights_and_reconstruct_layer()

    def _load_weights_and_reconstruct_layer(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_state = torch.load(
                self.run_dir / f"model_epoch{self.epoch:03d}.pt",
                map_location=device,
                weights_only=True
            )
            w1_key = 'embedding_net.statics_embedding.net.0.weight'
            b1_key = 'embedding_net.statics_embedding.net.0.bias'
            w2_key = 'embedding_net.statics_embedding.net.3.weight'
            b2_key = 'embedding_net.statics_embedding.net.3.bias'
            required_keys = [w1_key, b1_key, w2_key, b2_key]
            if not all(key in model_state for key in required_keys):
                missing = [k for k in required_keys if k not in model_state]
                logging.error(f"Embedding keys missing: {missing}")
                raise KeyError(f"Missing keys: {missing}")

            self.W1_stat = model_state[w1_key].cpu()
            self.B1_stat = model_state[b1_key].cpu()
            self.W2_stat = model_state[w2_key].cpu()
            self.B2_stat = model_state[b2_key].cpu()
            input_size = self.W1_stat.shape[1]
            hidden_size = self.W1_stat.shape[0]
            output_size = self.W2_stat.shape[0]
            self.n_embedding_dims = output_size

            if input_size != self.n_static_features:
                 logging.warning(f"Weight matrix input size ({input_size}) != num static features ({self.n_static_features}).")

            self.embedding_layer = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, output_size), nn.Tanh()
            ).cpu()
            temp_state_dict = {
                '0.weight': self.W1_stat, '0.bias': self.B1_stat,
                '2.weight': self.W2_stat, '2.bias': self.B2_stat
            }
            self.embedding_layer.load_state_dict(temp_state_dict)
            self.embedding_layer.eval()

            W_eff_np = np.dot(self.W2_stat.numpy(), self.W1_stat.numpy())
            self.df_effective_weights = pd.DataFrame(
                W_eff_np,
                columns=self.static_features[:input_size],
                index=[f"EmbDim_{i+1}" for i in range(output_size)]
            )

        except Exception as e:
            logging.exception(f"Error loading weights/reconstructing layer: {e}")
            raise

    def _plot_generic_heatmap(self, data: pd.DataFrame, title: str, filename: str,
                              xlabel: str, ylabel: str,
                              xticklabels: list | bool = True, yticklabels: list | bool = True,
                              cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None,
                              na_color: str = 'lightgray', annotate: bool = False, fmt: str = ".2f",
                              **kwargs):
        """
        Internal helper function to generate and save a generic heatmap.

        Args:
            data (pd.DataFrame): Data to plot.
            title (str): Plot title.
            filename (str): Full path to save the plot file.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            xticklabels (list | bool, optional): Labels for x-axis ticks. Defaults to True.
            yticklabels (list | bool, optional): Labels for y-axis ticks. Defaults to True.
            cmap (str, optional): Colormap. Defaults to "viridis".
            vmin (float | None, optional): Minimum value for color scale. Defaults to None (auto).
            vmax (float | None, optional): Maximum value for color scale. Defaults to None (auto).
            na_color (str, optional): Color for NaN values. Defaults to 'lightgray'.
            annotate (bool, optional): Annotate cells with values. Defaults to False.
            fmt (str, optional): String format for annotations. Defaults to ".2f".
            **kwargs: Additional keyword arguments passed to sns.heatmap.
        """
        if data is None or data.empty:
            logging.warning(f"Skipping heatmap '{title}' due to missing data.")
            return

        if data.isnull().all().all():
            logging.warning(f"Skipping heatmap '{title}' because all data is masked (NaN).")
            return

        # Determine vmin/vmax from non-NaN values if not provided
        effective_vmin = vmin if vmin is not None else data.min(skipna=True).min(skipna=True)
        effective_vmax = vmax if vmax is not None else data.max(skipna=True).max(skipna=True)

        # Adjust if vmin/vmax are invalid or equal after masking
        if pd.isna(effective_vmin) or pd.isna(effective_vmax) or effective_vmin == effective_vmax:
             logging.warning(f"Adjusting vmin/vmax for '{title}' due to limited data range after masking.")
             vmin_adj = effective_vmin - 0.1 if not pd.isna(effective_vmin) else -0.1
             vmax_adj = effective_vmax + 0.1 if not pd.isna(effective_vmax) else 0.1
             if vmin_adj >= vmax_adj:
                vmin_adj, vmax_adj = -0.1, 0.1
             effective_vmin, effective_vmax = vmin_adj, vmax_adj

        n_rows, n_cols = data.shape

        width_mult = 0.4 if n_cols < 50 else 0.3
        height_mult = 0.3 if n_rows < 40 else 0.25
        figsize_width = max(10, n_cols * width_mult)
        figsize_height = max(8, n_rows * height_mult)

        plt.figure(figsize=(figsize_width, figsize_height))
        ax = plt.gca()
        ax.set_facecolor(na_color)

        sns.heatmap(
            data,
            cmap=cmap,
            annot=annotate,
            fmt=fmt,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            vmin=effective_vmin,
            vmax=effective_vmax,
            cbar=True,
            mask=data.isna(),
            ax=ax,
            linewidths=kwargs.pop('linewidths', 0.5 if annotate else 0),
            linecolor=kwargs.pop('linecolor', 'lightgray'),
            **kwargs
        )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

        if isinstance(xticklabels, (list, pd.Index, np.ndarray)) or xticklabels is True:
             plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        ax.set_title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_effective_weights(self, **kwargs):
        """
        Generates and saves a heatmap of the effective embedding weights (W2 * W1).

        Args:
            **kwargs: Additional keyword arguments passed to sns.heatmap via _plot_generic_heatmap.
        """
        if self.df_effective_weights is None:
            logging.error("Effective weights not calculated. Cannot plot.")
            return

        title = "Heatmap of Effective Static Embedding Weights (W2 * W1)"
        filepath = self.results_dir / "effective_weights_heatmap.png"

        self._plot_generic_heatmap(
            data=self.df_effective_weights,
            title=title,
            filename=filepath,
            xlabel="Static Input Features",
            ylabel="Effective Embedding Dimension",
            xticklabels=self.df_effective_weights.columns,
            yticklabels=self.df_effective_weights.index,
            cmap="rainbow",
            vmin=kwargs.pop('vmin', None), 
            vmax=kwargs.pop('vmax', None),
            **kwargs
        )
        logging.info(f"Effective weights heatmap saved to {filepath}")

    def plot_phased_effective_weights(self, thresholds: list, **kwargs):
        """
        Generates multiple weight heatmaps, thresholding by absolute weight value.

        Args:
            thresholds (list): List of absolute value thresholds (e.g., [0.5, 0.25, 0.1]).
            **kwargs: Additional keyword arguments for heatmap plotting (passed to helper).
        """
        if self.df_effective_weights is None:
            logging.error("Effective weights not calculated. Cannot plot phased heatmap.")
            return

        global_vmin = self.df_effective_weights.min().min()
        global_vmax = self.df_effective_weights.max().max()

        for i, threshold in enumerate(thresholds):
            masked_df = self.df_effective_weights.where(np.abs(self.df_effective_weights) >= threshold)

            phase_title = (f"Effective Static Embedding Weights (Phase {i+1}: |Weight| >= {threshold:.2f})")
            phase_filename = f"static_embeddings_weights_heatmap_phase{i+1}_thresh{threshold:.2f}.png"
            filepath = self.results_dir / phase_filename

            self._plot_generic_heatmap(
                data=masked_df,
                title=phase_title,
                filename=filepath,
                xlabel="Static Input Features",
                ylabel="Effective Embedding Dimension",
                xticklabels=self.df_effective_weights.columns,
                yticklabels=self.df_effective_weights.index,
                cmap="RdBu_r",
                vmin=global_vmin,
                vmax=global_vmax,
                na_color=kwargs.pop('na_color', 'lightgray'),
                **kwargs
            )
        logging.info(f"Phased effective weights heatmaps saved to {self.results_dir}")

    def analyze_dimension_similarity(self, metric='cosine', **kwargs):
        """
        Calculates and plots pairwise similarity between embedding dimensions based on effective weights.

        Args:
            metric (str, optional): Similarity metric ('cosine' or 'pearson'). Defaults to 'cosine'.
            **kwargs: Additional keyword arguments passed to sns.heatmap via _plot_generic_heatmap.

        Returns:
            pd.DataFrame | None: DataFrame containing the similarity matrix, or None on error.
        """
        if self.df_effective_weights is None:
            logging.error("Effective weights not calculated. Cannot compute similarity.")
            return None

        similarity_matrix = None
        metric_name = ""
        vmin, vmax = -1, 1
        df_weights = self.df_effective_weights

        if metric == 'cosine':
            similarity_matrix = cosine_similarity(df_weights.values)
            metric_name = "Cosine Similarity"
        elif metric == 'pearson':
            similarity_matrix = df_weights.T.corr(method='pearson').values
            metric_name = "Pearson Correlation"
        else:
            logging.error(f"Unknown similarity metric '{metric}'. Choose 'cosine' or 'pearson'.")
            return None

        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=df_weights.index,
            columns=df_weights.index
        )

        filename = f"embedding_dim_similarity_{metric}.png"
        filepath = self.results_dir / filename
        plot_title = f"Pairwise {metric_name} Between Embedding Dims"

        self._plot_generic_heatmap(
            data=similarity_df,
            title=plot_title,
            filename=filepath,
            xlabel="Embedding Dimension",
            ylabel="Embedding Dimension",
            xticklabels=similarity_df.columns,
            yticklabels=similarity_df.index,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            annotate=True,
            fmt=kwargs.pop('fmt', ".1f"),
            **kwargs
        )
        logging.info(f"Embedding dimension similarity heatmap saved to {filepath}")

        return similarity_df


    def analyze_output_feature_correlation(self, **kwargs):
        """
        Calculates and plots correlation between embedding outputs and original static features.

        Args:
            **kwargs: Additional keyword arguments passed to the internal plotting helper
                    (and ultimately to sns.heatmap).

        Returns:
            pd.DataFrame | None: DataFrame containing the correlation matrix, or None on error.
        """
        if self.embedding_layer is None:
            logging.error("Embedding layer not reconstructed. Cannot calculate outputs.")
            return None

        unique_inputs_dict = self.analyzer.get_unique_static_inputs()
        norm_x_s_unique = unique_inputs_dict['x_s_unique'] # Shape [num_basins, S]
        num_basins, _ = norm_x_s_unique.shape
        if num_basins == 0:
            logging.error("No unique basins found in the dataset.")
            return None

        attribute_means = self.analyzer.scaler.get('attribute_means', {})
        attribute_stds = self.analyzer.scaler.get('attribute_stds', {})

        stat_centers = np.zeros(len(self.static_features))
        stat_scales = np.ones(len(self.static_features))
        for k, feat_name in enumerate(self.static_features):
            stat_centers[k] = attribute_means.get(feat_name, 0.0)
            scale_val = attribute_stds.get(feat_name, 1.0)
            stat_scales[k] = scale_val if scale_val != 0 else 1.0
        
        original_static_features_np = (norm_x_s_unique * stat_scales) + stat_centers # norm_x_s [N, S], stat_scales [S], stat_centers [S]

        features_tensor = torch.from_numpy(norm_x_s_unique).float().cpu()

        # Get embedding outputs
        with torch.no_grad():
            embedding_outputs_tensor = self.embedding_layer(features_tensor)
        embedding_outputs_np = embedding_outputs_tensor.cpu().numpy()

        # Calculate correlations
        num_embeddings = embedding_outputs_np.shape[1]
        num_features = original_static_features_np.shape[1]
        correlation_matrix = np.zeros((num_embeddings, num_features))

        for i in range(num_embeddings): # Iterate through embedding dimensions
            emb_col = embedding_outputs_np[:, i]
            if np.isnan(emb_col).all():
                correlation_matrix[i, :] = np.nan
                continue
            for j in range(num_features): # Iterate through original static features
                feat_col = original_static_features_np[:, j]

                if np.allclose(np.std(feat_col[~np.isnan(feat_col)]), 0, atol=1e-9) or \
                np.isnan(feat_col).all() or \
                np.isnan(emb_col).all():
                    corr = np.nan
                else:
                    finite_mask = np.isfinite(emb_col) & np.isfinite(feat_col)
                    if np.sum(finite_mask) < 2 :
                        corr = np.nan
                    else:
                        corr, _ = pearsonr(emb_col[finite_mask], feat_col[finite_mask])

                correlation_matrix[i, j] = corr

        embedding_dim_names = [f"EmbDim_{k+1}" for k in range(num_embeddings)]
        correlation_df = pd.DataFrame(
            correlation_matrix,
            index=embedding_dim_names,
            columns=self.static_features
        )

        filepath = self.results_dir / "output_feature_correlation.png"
        plot_title = "Correlation: Embedding Outputs vs. Original Static Features"

        self._plot_generic_heatmap(
            data=correlation_df,
            title=plot_title,
            filename=filepath,
            xlabel="Original Static Features",
            ylabel="Embedding Dimension Output",
            xticklabels=correlation_df.columns,
            yticklabels=correlation_df.index,
            cmap="RdBu_r", 
            vmin=-1, vmax=1,
            annotate=False,
            fmt=kwargs.pop('fmt', ".1f"),
            na_color=kwargs.pop('na_color', 'lightgray'), 
            **kwargs
        )
        logging.info(f"Output vs. Feature correlation heatmap saved to {filepath}")

        return correlation_df


if __name__ == "__main__":
    run_dir = Path("/sci/labs/efratmorin/eli.levinkopf/batch_runs/runs/train_lstm_rs_22_1503_194719/")

    if not run_dir.exists():
        logging.error(f"ERROR: Model run directory not found at {run_dir}") 
    else:
        static_analyzer = StaticEmbeddingAnalyzer(run_dir=run_dir, epoch=25, period="test")
        static_analyzer.plot_phased_effective_weights(thresholds=[0.7, 0.5, 0.25, 0.0])
        similarity_df_cos = static_analyzer.analyze_dimension_similarity(metric='cosine')
        correlation_df = static_analyzer.analyze_output_feature_correlation()