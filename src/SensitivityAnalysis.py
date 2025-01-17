import os
import torch
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml
import re
import matplotlib.pyplot as plt
from adjustText import adjust_text
from SALib.sample.sobol import sample
from SALib.sample.morris import sample as morris_sample
from SALib.analyze import sobol, morris
from neuralhydrology.modelzoo.ealstm import EALSTM
from neuralhydrology.utils.config import Config

sys.path.append("/sci/labs/efratmorin/eli.levinkopf/neuralhydrology")
# sys.path.append("/sci/labs/efratmorin/omer_roi_cohen/neuralhydrology")

METADATA_PATH = "/sci/labs/efratmorin/eli.levinkopf/batch_runs/metadata"
BASE_SAMPLE_SIZE = 2048
NUM_LEVELS = 8
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

DATA_DIR = "/sci/labs/efratmorin/lab_share/FloodsML/data/Caravan/timeseries/csv"
ATTRIBUTES_DIR = "/sci/labs/efratmorin/lab_share/FloodsML/data/Caravan/attributes"

class SensitivityAnalysis:
    def __init__(self, model_path, analysis_type="both", method="morris"):
        self.model_path = model_path
        self.cfg = self._load_config()
        self.seed = self.cfg.get('seed', 42)
        self._set_seed(self.seed)
        self.dynamic_inputs = self.cfg['dynamic_inputs']
        self.static_attributes = self.cfg['static_attributes']
        self.region = self._extract_region()
        self.seq_length = self.cfg['seq_length']
        self.feature_range_path = f'{METADATA_PATH}/feature_range_{self.region}.csv'
        self.analysis_type = analysis_type
        self.method = method
        self.normalization_stats = {}
        self._load_normalization_statistics_from_scaler_file()
        self.static_baseline = self._create_static_baseline()
        self.dynamic_baseline = self._create_dynamic_baseline()
        self.problem = self._define_problem()

        if self.method == "sobol":
            self.param_values = sample(
                self.problem, 
                BASE_SAMPLE_SIZE, 
                seed=self.seed
            )
        elif self.method == "morris":
            self.param_values = morris_sample(
                self.problem, 
                BASE_SAMPLE_SIZE, 
                num_levels=NUM_LEVELS, 
                optimal_trajectories=32, 
                local_optimization=True, 
                seed=self.seed
            )
        else:
            raise ValueError("Unsupported method. Choose 'sobol' or 'morris'.")

        self.model = self._load_model()

    def _set_seed(self, seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(self.model_path), 'config.yml')
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _extract_region(self):
        train_basin_file = self.cfg['train_basin_file']
        regions = ['camels', 'camelsaus', 'camelsbr', 'camelscl', 'camelsgb', 'hysets', 'lamah']
        for region in regions:
            if region in train_basin_file:
                return region
        return 'all'

    def _define_problem(self):
        features, feature_bounds = self._extract_feature_bounds()

        problem = {
            'num_vars': len(features),
            'names': features,
            'bounds': feature_bounds.tolist()
        }

        return problem

    def _extract_feature_bounds(self):
        feature_ranges = pd.read_csv(self.feature_range_path)

        # Normalize feature names for comparison
        feature_ranges['Feature'] = feature_ranges['Feature'].str.strip().str.lower()
        normalized_static_attributes = [attr.strip().lower() for attr in self.static_attributes]
        normalized_dynamic_inputs = [attr.strip().lower() for attr in self.dynamic_inputs]

        if self.analysis_type == "dynamic":
            features = normalized_dynamic_inputs
        elif self.analysis_type == "static":
            features = normalized_static_attributes
        else:
            features = normalized_static_attributes + normalized_dynamic_inputs

        feature_bounds = []

        # Extract feature bounds
        for feature in features:
            range_data = feature_ranges[feature_ranges['Feature'] == feature]
            if not range_data.empty:
                min_val, max_val = range_data[['Min', 'Max']].iloc[0]
                feature_bounds.append([min_val, max_val])
            else:
                raise ValueError(f"Feature {feature} not found in range file {self.feature_range_path}")

        feature_bounds = np.array(feature_bounds)

        if feature_bounds.ndim != 2 or feature_bounds.shape[1] != 2:
            raise ValueError("Feature bounds array must be a 2D array with [min, max] pairs for each feature.")

        return features, feature_bounds

    def _load_model(self):
        config = Config(self.cfg)
        model_class_name = self.cfg['model'].lower()
        if model_class_name == 'cudalstm':
            from neuralhydrology.modelzoo.cudalstm import CudaLSTM
            model = CudaLSTM(cfg=config)
        elif model_class_name == 'ealstm':
            model = EALSTM(cfg=config)
        else:
            raise ValueError(f"Model {model_class_name} is not supported.")
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(self.model_path, map_location=map_location))
        model.to(map_location)
        model.eval()
        return model

    def _load_normalization_statistics_from_scaler_file(self):
        scaler_path = os.path.join(os.path.dirname(self.model_path), 'train_data/train_data_scaler.yml')
        with open(scaler_path, 'r') as f:
            scaler = yaml.safe_load(f)

        # Ensure all expected keys are present
        required_keys = ['attribute_means', 'attribute_stds', 'xarray_feature_center', 'xarray_feature_scale']
        for key in required_keys:
            if key not in scaler:
                raise KeyError(f"Expected key '{key}' not found in scaler file.")

        # Store everything in one dictionary
        for feature in scaler['attribute_means'].keys():
            self.normalization_stats[feature] = {
                'mean': scaler['attribute_means'][feature],
                'std': scaler['attribute_stds'][feature]
            }

        center_data = scaler['xarray_feature_center']['data_vars']
        scale_data = scaler['xarray_feature_scale']['data_vars']

        for feature in center_data:
            self.normalization_stats[feature] = {
                'center': center_data[feature]['data'],
                'scale': scale_data[feature]['data']
            }

    def _create_static_baseline(self):
        """Create a normalized baseline for static features using their mean and std."""
        static_baseline = []
        for feature in self.static_attributes:
            mean = self.normalization_stats[feature]['mean']
            std = self.normalization_stats[feature]['std']
            # normalized baseline value for this static feature
            static_val = (mean - mean) / std  # This should be 0.0 since (mean - mean)/std = 0
            # Alternatively, we can just use 0 here, since the normalized mean is always 0,
            # but this form makes the intention clear.
            static_baseline.append(static_val)
        return np.array(static_baseline, dtype=np.float32)

    def _create_dynamic_baseline(self):
        """Create a normalized baseline for dynamic features using their center and scale."""
        dynamic_baseline = []
        for feature in self.dynamic_inputs:
            center = self.normalization_stats[feature]['center']
            scale = self.normalization_stats[feature]['scale']
            # normalized baseline value for this dynamic feature
            # (center - center)/scale = 0, but again we keep the formula for clarity
            dyn_val = (center - center) / scale
            dynamic_baseline.append(dyn_val)
        return np.array(dynamic_baseline, dtype=np.float32)

    def _standardize_features(self, features):
        standardized_features = []
        for i, feature in enumerate(self.problem['names']):
            if feature not in self.normalization_stats:
                raise ValueError(f"Feature {feature} not found in normalization statistics.")

            if feature in self.static_attributes:
                mean = self.normalization_stats[feature]['mean']
                std = self.normalization_stats[feature]['std']
                standardized_feature = (features[:, i] - mean) / std
            elif feature in self.dynamic_inputs:
                center = self.normalization_stats[feature]['center']
                scale = self.normalization_stats[feature]['scale']
                standardized_feature = (features[:, i] - center) / scale
            else:
                raise ValueError(f"Normalization parameters missing for feature {feature}.")

            standardized_features.append(standardized_feature)

        return np.column_stack(standardized_features)

    def _predict_batch(self, input_features_batch):
        # input_features_batch are the features we vary (according to self.analysis_type)
        # We already assume these are in original scale and we standardize them below.

        # Standardize only the features that we're varying.
        input_features_batch = self._standardize_features(input_features_batch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_static_features = len(self.static_attributes)
        num_dynamic_features = len(self.dynamic_inputs)

        if self.analysis_type == "dynamic":
            # We vary dynamic features. Use static baseline for x_s and replaced dynamic features with standardized_features.
            # x_s is fixed at static_baseline (which is all zeros since normalized mean is zero).
            x_s_batch = np.tile(self.static_baseline, (input_features_batch.shape[0], 1))
            x_d_batch = input_features_batch  # dynamic features vary
            
            x_s_batch = torch.tensor(x_s_batch, dtype=torch.float32).to(device)
            # For dynamic inputs, we replicate the single-vector along the seq_length dimension
            x_d_batch = torch.tensor(x_d_batch, dtype=torch.float32).unsqueeze(1)
            x_d_batch = x_d_batch.repeat(1, self.seq_length, 1).to(device)

        elif self.analysis_type == "static":
            # We vary static features. Use dynamic baseline for x_d and replaced static features with standardized_features.
            # x_d is fixed at dynamic_baseline (normalized to zero).
            x_s_batch = input_features_batch
            x_d_batch = np.tile(self.dynamic_baseline, (input_features_batch.shape[0], 1))
            
            x_s_batch = torch.tensor(x_s_batch, dtype=torch.float32).to(device)
            # For dynamic inputs, replicate baseline along time dimension
            x_d_batch = torch.tensor(x_d_batch, dtype=torch.float32).unsqueeze(1)
            x_d_batch = x_d_batch.repeat(1, self.seq_length, 1).to(device)

        else:  # both dynamic and static
            # Split the input_features_batch into static and dynamic parts
            x_s_var = input_features_batch[:, :num_static_features]
            x_d_var = input_features_batch[:, num_static_features:]

            # For 'both' scenario, we might fix one group at baseline and vary the other or vary both.
            # If you want both sets to vary, you must define how you're sampling them.
            # Here is one approach: just use x_s_var and x_d_var as given.
            x_s_batch = x_s_var
            x_d_batch = x_d_var
            
            x_s_batch = torch.tensor(x_s_batch, dtype=torch.float32).to(device)
            x_d_batch = torch.tensor(x_d_batch, dtype=torch.float32).unsqueeze(1)
            x_d_batch = x_d_batch.repeat(1, self.seq_length, 1).to(device)

        inputs = {
            'x_s': x_s_batch,
            'x_d': x_d_batch
        }

        with torch.no_grad():
            outputs = self.model(inputs)
            return outputs['y_hat'][:, -1].cpu().numpy()


    def _collect_results(self):
        results = []
        total_samples = len(self.param_values)

        with tqdm(total=total_samples, desc="Running Sensitivity Analysis") as pbar:
            for i in range(0, total_samples, BATCH_SIZE):
                batch = self.param_values[i:i + BATCH_SIZE]
                batch_results = self._predict_batch(np.array(batch))
                results.extend(batch_results)
                pbar.update(len(batch))
        
        return np.array(results).flatten()

    def _analyze_results(self, results):
        if self.method == "sobol":
            Si = sobol.analyze(
                self.problem, 
                results, 
                print_to_console=False
            )
            feature_names = self.problem['names']
            data = {'Feature': feature_names, 'S1': Si['S1'], 'ST': Si['ST']}
        elif self.method == "morris":
            Si = morris.analyze(
                self.problem, 
                self.param_values, 
                results,
                num_resamples=1024, 
                num_levels=NUM_LEVELS, 
                seed=self.seed
            )
            feature_names = self.problem['names']
            data = {'Feature': feature_names, 'Mu*': Si['mu_star'], 'Mu': Si['mu'], 'Std Dev': Si['sigma']}
        else:
            raise ValueError("Unsupported sensitivity analysis method.")

        return pd.DataFrame(data)

    def _aggregate_features(self, df):
        glc_features = [f"glc_pc_s{i:02d}" for i in range(1, 23)]
        pnv_features = [f"pnv_pc_s{i:02d}" for i in range(1, 16)]

        df = self._aggregate_feature_group(df, glc_features, "glc_pc[01-22]")
        df = self._aggregate_feature_group(df, pnv_features, "pnv_pc[01-15]")
        
        return df.dropna(axis=1, how='all').sort_values(
            by='Mu*' if self.method == "morris" else 'S1', ascending=False
        ).reset_index(drop=True)

    def _aggregate_feature_group(self, df, features, group_name):
        if any(f in df['Feature'].values for f in features):
            feature_df = df[df['Feature'].isin(features)].copy()
            aggregated = pd.DataFrame({
                'Feature': [group_name],
                'S1': [feature_df['S1'].mean() if 'S1' in df.columns else None],
                'ST': [feature_df['ST'].mean() if 'ST' in df.columns else None],
                'Mu': [feature_df['Mu'].mean() if 'Mu' in df.columns else None],
                'Mu*': [feature_df['Mu*'].mean() if 'Mu*' in df.columns else None],
                'Std Dev': [feature_df['Std Dev'].mean() if 'Std Dev' in df.columns else None]
            })
            return pd.concat([df[~df['Feature'].isin(features)], aggregated], ignore_index=True)
        return df

    def _save_and_visualize_results(self, df):
        print(df.to_string(index=False, float_format='{:,.2f}'.format))
        output_dir = os.path.join(os.path.dirname(self.model_path), 'sensitivity_analysis_results')
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            f"sensitivity_results_{os.path.basename(self.model_path).split('.')[0]}_{self.method}_{self.analysis_type}.csv"
        )
        df.to_csv(output_file, index=False, float_format='%.2f')
        print(f"Sensitivity analysis results saved to: {output_file}")

        if self.method == "morris":
            df = df.sort_values(by='Mu*', ascending=False)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

            # Mu* Plot
            axes[0, 0].barh(df['Feature'], df['Mu*'], color='cornflowerblue', alpha=0.7)
            axes[0, 0].set_title("Mu* (Mean Absolute Effect)")
            axes[0, 0].invert_yaxis()

            # Mu Plot
            axes[0, 1].barh(df['Feature'], df['Mu'], color='cornflowerblue', alpha=0.7)
            axes[0, 1].set_title("Mu (Mean Effect)")
            axes[0, 1].invert_yaxis()

            # Sigma Plot 
            axes[1, 0].barh(df['Feature'], df['Std Dev'], color='cornflowerblue', alpha=0.7)
            axes[1, 0].set_title("Standard Deviation")
            axes[1, 0].invert_yaxis()

            fig.delaxes(axes[1, 1])

            for ax in axes.flat:
                ax.set_xlabel("Value")
                ax.grid(True, linestyle='--', alpha=0.5)
            
            morris_path = os.path.join(output_dir, f"morris_sensitivity_bar_{os.path.basename(self.model_path).split('.')[0]}_{self.analysis_type}.png")
            plt.savefig(morris_path, dpi=300, bbox_inches='tight')
            print(f"Horizontal bar plots saved to: {morris_path}")
            plt.show()

            # Scatter Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(df['Mu*'], df['Std Dev'], alpha=0.7, s=100)
            texts = [plt.text(x, y, label, fontsize=9) for x, y, label in zip(df['Mu*'], df['Std Dev'], df['Feature'])]
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', alpha=0.5))
            plt.xlabel('Mu* (Mean Absolute Effect)')
            plt.ylabel('Sigma (Std Dev)')
            plt.title("Morris Sensitivity Analysis")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            scatter_path = os.path.join(output_dir, f"morris_sensitivity_scatter_{os.path.basename(self.model_path).split('.')[0]}_{self.analysis_type}.png")
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.show()
        elif self.method == "sobol":
            # Sobol Bar Plots
            fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            axes[0].bar(df['Feature'], df['S1'], color='blue', alpha=0.7)
            axes[0].set_ylabel("S1 (First-order Sensitivity Index)")
            axes[0].set_title("Sobol Sensitivity Analysis - S1")

            axes[1].bar(df['Feature'], df['ST'], color='green', alpha=0.7)
            axes[1].set_ylabel("ST (Total Sensitivity Index)")
            axes[1].set_xlabel("Features")
            axes[1].set_title("Sobol Sensitivity Analysis - ST")

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            sobol_path = os.path.join(output_dir, f"sobol_sensitivity_analysis_{os.path.basename(self.model_path).split('.')[0]}_{self.analysis_type}.png")
            plt.savefig(sobol_path, dpi=300, bbox_inches='tight')
            print(f"Sobol sensitivity analysis plot saved to: {sobol_path}")
            plt.show()

    def run_sensitivity_analysis(self):
        results = self._collect_results()
        df = self._analyze_results(results)
        df = self._aggregate_features(df)
        self._save_and_visualize_results(df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis on EA-LSTM model.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file (e.g., model_epoch060.pt)")
    parser.add_argument("--analysis_type", type=str, choices=["dynamic", "static", "both"], default="both",
                        help="Type of sensitivity analysis to perform: 'dynamic', 'static', or 'both'")
    parser.add_argument("--method", type=str, choices=["sobol", "morris"], default="sobol",
                        help="Sensitivity analysis method: 'sobol' or 'morris'")
    args = parser.parse_args()

    sa = SensitivityAnalysis(model_path=args.model_path, analysis_type=args.analysis_type, method=args.method)
    sa.run_sensitivity_analysis()
