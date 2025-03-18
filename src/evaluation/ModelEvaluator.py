import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
import os
import re
import argparse
import logging

class ModelEvaluator:
    def __init__(self, run_dir, epoch):
        """
        Initializes the ModelEvaluator with the specified run directory and epoch.

        Args:
            run_dir (str): Path to the run directory.
            epoch (int): Epoch number to evaluate.
        """
        self.run_dir = run_dir
        self.epoch = epoch
        self.test_dir = os.path.join(run_dir, 'test', f'model_epoch{str(epoch).zfill(3)}')
        self.metrics_file = os.path.join(self.test_dir, 'test_metrics.csv')
        self.results_file = os.path.join(self.test_dir, 'test_results.p')
        self.output_dir = os.path.join(self.test_dir, 'test_results')
        self.training_loss_file = os.path.join(run_dir, 'output.log')
        self.test_start = self.get_test_start_date()
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics = None
        self.results = None

    def get_test_start_date(self):
        """Parses the test start date from the training log."""
        try:
            with open(self.training_loss_file, 'r') as f:
                for line in f:
                    if "test_start_date" in line:
                        match = re.search(r"test_start_date: (\d{4}-\d{2}-\d{2})", line)
                        if match:
                            return dt.datetime.strptime(match.group(1), "%Y-%m-%d")
        except FileNotFoundError:
            logging.info(f"Training loss file not found at {self.training_loss_file}. Please verify the file path.")
        return None

    def load_metrics(self):
        """Loads the test metrics from the metrics file."""
        logging.info("Loading metrics...")
        try:
            self.metrics = pd.read_csv(self.metrics_file)
            logging.info("Metrics loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Metrics file not found at {self.metrics_file}: {e}")
            raise

    def analyze_metrics(self):
        """Analyzes the metrics and generates descriptive statistics and plots."""
        if self.metrics is not None and {'NSE', 'MSE', 'KGE'}.issubset(self.metrics.columns):
            # Descriptive statistics for metrics
            metrics_summary = self.metrics[['NSE', 'MSE', 'KGE']].describe()
            metrics_summary.to_csv(os.path.join(self.output_dir, 'metrics_summary.csv'))
            logging.info(f"Metrics summary saved to {os.path.join(self.output_dir, 'metrics_summary.csv')}")

            # Plot FULL CDF of NSE values
            plt.figure(figsize=(10, 6))
            sorted_nse = np.sort(self.metrics['NSE'])
            cdf = np.arange(1, len(sorted_nse) + 1) / len(sorted_nse)
            plt.plot(sorted_nse, cdf, label='NSE CDF')
            
            negative_nse_percentage = (self.metrics['NSE'] < 0).mean() * 100
            median_nse = self.metrics['NSE'].median()
            
            plt.title(f'NSE CDF\n({negative_nse_percentage:.1f}% of NSEs < 0)\nMedian NSE: {median_nse:.3f}')
            plt.xlabel('NSE')
            plt.ylabel('CDF')
            plt.savefig(os.path.join(self.output_dir, 'nse_cdf.png'))
            plt.close()
            
            logging.info(f"NSE CDF plot saved to {os.path.join(self.output_dir, 'nse_cdf.png')}")

            # Plot ZOOMED-IN CDF (clip outliers) for NSE
            threshold = 0
            clipped_df = self.metrics[self.metrics['NSE'] > threshold]
            
            if not clipped_df.empty:
                plt.figure(figsize=(10, 6))
                sorted_nse_clipped = np.sort(clipped_df['NSE'])
                cdf_clipped = np.arange(1, len(sorted_nse_clipped) + 1) / len(sorted_nse_clipped)
                plt.plot(sorted_nse_clipped, cdf_clipped, label='NSE CDF')
                
                median_nse_clipped = clipped_df['NSE'].median()
                
                plt.title(f'Median NSE: {median_nse_clipped:.3f}')
                plt.xlabel('NSE')
                plt.ylabel('CDF')
                plt.savefig(os.path.join(self.output_dir, 'nse_cdf_clipped.png'))
                plt.close()
                
                logging.info(f"Zoomed-in NSE CDF plot saved to "
                            f"{os.path.join(self.output_dir, 'nse_cdf_clipped.png')}")
            else:
                logging.warning(f"No data points with NSE > {threshold}; zoomed-in plot not created.")

            # Plot FULL CDF of KGE values
            plt.figure(figsize=(10, 6))
            sorted_kge = np.sort(self.metrics['KGE'])
            cdf_kge = np.arange(1, len(sorted_kge) + 1) / len(sorted_kge)
            plt.plot(sorted_kge, cdf_kge, label='KGE CDF')
            
            negative_kge_percentage = (self.metrics['KGE'] < 0).mean() * 100
            median_kge = self.metrics['KGE'].median()
            
            plt.title(f'KGE CDF\n({negative_kge_percentage:.1f}% of KGEs < 0)\nMedian KGE: {median_kge:.3f}')
            plt.xlabel('KGE')
            plt.ylabel('CDF')
            plt.savefig(os.path.join(self.output_dir, 'kge_cdf.png'))
            plt.close()
            
            logging.info(f"KGE CDF plot saved to {os.path.join(self.output_dir, 'kge_cdf.png')}")

            # Plot ZOOMED-IN CDF (clip negatives) for KGE values
            threshold_kge = 0
            clipped_kge_df = self.metrics[self.metrics['KGE'] > threshold_kge]
            
            if not clipped_kge_df.empty:
                plt.figure(figsize=(10, 6))
                sorted_kge_clipped = np.sort(clipped_kge_df['KGE'])
                cdf_kge_clipped = np.arange(1, len(sorted_kge_clipped) + 1) / len(sorted_kge_clipped)
                plt.plot(sorted_kge_clipped, cdf_kge_clipped, label='KGE CDF')
                
                median_kge_clipped = clipped_kge_df['KGE'].median()
                
                plt.title(f'Median KGE: {median_kge_clipped:.3f}')
                plt.xlabel('KGE')
                plt.ylabel('CDF')
                plt.savefig(os.path.join(self.output_dir, 'kge_cdf_clipped.png'))
                plt.close()
                
                logging.info(f"Zoomed-in KGE CDF plot saved to {os.path.join(self.output_dir, 'kge_cdf_clipped.png')}")
            else:
                logging.warning(f"No data points with KGE > {threshold_kge}; zoomed-in plot not created.")

            # Identify outliers based on NSE and MSE
            outliers = self.metrics[
                (self.metrics['NSE'] < -1) | 
                (self.metrics['MSE'] > self.metrics['MSE'].quantile(0.95))
            ]
            outliers.to_csv(os.path.join(self.output_dir, 'outliers_summary.csv'))
            logging.info(f"Outliers saved to {os.path.join(self.output_dir, 'outliers_summary.csv')}")

    def load_results(self):
        """Loads the test results from the results file."""
        logging.info("Loading results...")
        try:
            with open(self.results_file, 'rb') as f:
                self.results = pickle.load(f)
                logging.info("Results loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Results file not found at {self.results_file}: {e}")
            raise

    def plot_observed_vs_predicted(self):
        """Plots observed vs. predicted streamflow for each basin in the results."""
        logging.info("Plotting observed vs predicted...")
        if self.results:
            for basin_id, data in self.results.items():
                basin_data = data['1D']
                if 'xr' in basin_data:
                    ds = basin_data['xr']
                    if 'streamflow_obs' in ds and 'streamflow_sim' in ds:
                        q_obs = ds['streamflow_obs'].values.flatten()
                        q_sim = ds['streamflow_sim'].values.flatten()

                        dates = [self.test_start + dt.timedelta(days=int(x)) for x in range(len(q_obs))]

                        observed_vs_predicted_dir = os.path.join(self.output_dir, 'observed_vs_predicted')
                        os.makedirs(observed_vs_predicted_dir, exist_ok=True)

                        # Plot observed vs. predicted
                        plt.figure(figsize=(10, 6))
                        plt.plot(dates, q_obs, label='Observed', linestyle='-', alpha=0.7)
                        plt.plot(dates, q_sim, label='Predicted', linestyle='--', alpha=0.7)
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
                        plt.xlabel('Year')
                        plt.ylabel('Streamflow')
                        plt.title(f'Observed vs Predicted Streamflow for Basin {basin_id}')
                        plt.legend()
                        plt.grid()
                        plt.savefig(os.path.join(observed_vs_predicted_dir, f'observed_vs_predicted_{basin_id}.png'))
                        plt.close()
        logging.info(f"Observed vs Predicted plots saved to {os.path.join(observed_vs_predicted_dir)}")

    def residual_analysis(self):
        """Performs residual analysis by plotting residuals and their histograms."""
        logging.info("Performing residual analysis...")
        if self.results is None:
            logging.warning("No results loaded. Cannot perform residual analysis.")
            return

        for basin_id, data in self.results.items():
            basin_data = data['1D']
            if 'xr' in basin_data:
                ds = basin_data['xr']
                if 'streamflow_obs' in ds and 'streamflow_sim' in ds:
                    obs = ds['streamflow_obs'].values.flatten()
                    sim = ds['streamflow_sim'].values.flatten()
                    residuals = obs - sim

                    # Remove NaN values
                    residuals = residuals[~np.isnan(residuals)]
                    
                    # Check if there are valid residuals
                    if len(residuals) == 0:
                        logging.warning(f"No valid residuals for Basin {basin_id}. Skipping.")
                        continue
                        
                    dates = [self.test_start + dt.timedelta(days=int(x)) for x in range(len(obs))]

                    residuals_dir = os.path.join(self.output_dir, 'residuals')
                    histograms_dir = os.path.join(self.output_dir, 'residual_histograms')
                    os.makedirs(residuals_dir, exist_ok=True)
                    os.makedirs(histograms_dir, exist_ok=True)

                    # Plot residuals with yearly x-axis
                    plt.figure(figsize=(10, 6))
                    plt.plot(dates[:len(residuals)], residuals, label='Residuals', color='purple', alpha=0.7)
                    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
                    plt.gca().xaxis.set_major_locator(mdates.YearLocator()) 
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    plt.xlabel('Year')
                    plt.ylabel('Residuals (Observed - Predicted)')
                    plt.title(f'Residual Analysis for Basin {basin_id}')
                    plt.legend()
                    plt.grid()
                    plt.savefig(os.path.join(residuals_dir, f'residuals_{basin_id}.png'))
                    plt.close()

                    # Histogram of residuals
                    plt.figure(figsize=(8, 5))
                    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
                    plt.xlabel('Residual Value')
                    plt.ylabel('Frequency')
                    plt.title(f'Residual Histogram for Basin {basin_id}')
                    plt.grid()
                    plt.savefig(os.path.join(histograms_dir, f'residual_histogram_{basin_id}.png'))
                    plt.close()
        
        logging.info(f"Residuals plots saved to {os.path.join(residuals_dir)}")
        logging.info(f"Residual histograms saved to {os.path.join(histograms_dir)}")

    def extreme_flow_analysis(self, threshold=0.95):
        """Analyzes extreme flow events based on a threshold percentile."""
        logging.info("Performing extreme flow analysis...")
        if self.results is None:
            logging.warning("No results loaded. Cannot perform extreme flow analysis.")
            return

        for basin_id, data in self.results.items():
            basin_data = data['1D']
            if 'xr' in basin_data:
                ds = basin_data['xr']
                if 'streamflow_obs' in ds and 'streamflow_sim' in ds:
                    obs = ds['streamflow_obs'].values.flatten()
                    sim = ds['streamflow_sim'].values.flatten()
                    high_flow_idx = np.where(obs >= np.percentile(obs, threshold * 100))[0]

                    dates = [self.test_start + dt.timedelta(days=int(x)) for x in range(len(obs))]

                    extreme_flow_analysis_dir = os.path.join(self.output_dir, 'extreme_flow_analysis')
                    os.makedirs(extreme_flow_analysis_dir, exist_ok=True)

                    # Plot extreme flows with dates
                    plt.figure(figsize=(10, 6))
                    plt.scatter([dates[i] for i in high_flow_idx], sim[high_flow_idx], color='red', alpha=0.6, label='Predicted Extreme Flows')
                    plt.scatter([dates[i] for i in high_flow_idx], obs[high_flow_idx], color='blue', alpha=0.6, label='Observed Extreme Flows')
                    plt.plot(dates, obs, color='blue', alpha=0.3, label='Observed')
                    plt.plot(dates, sim, color='red', alpha=0.3, label='Predicted')
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
                    plt.xlabel('Year')
                    plt.ylabel('Streamflow')
                    plt.title(f'Extreme Flow Analysis for Basin {basin_id}')
                    plt.legend()
                    plt.grid()
                    plt.savefig(os.path.join(extreme_flow_analysis_dir, f'extreme_flows_{basin_id}.png'))
                    plt.close()
        logging.info(f"Extreme flow analysis saved to {os.path.join(extreme_flow_analysis_dir)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Run analysis on a trained NeuralHydrology model.")
    parser.add_argument("--run-dir", required=True, help="Path to the run directory.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load the model.")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.run_dir, args.epoch)
    evaluator.load_metrics()
    evaluator.analyze_metrics()
    # evaluator.load_results()
    # evaluator.plot_observed_vs_predicted()
    # evaluator.residual_analysis()
    # evaluator.extreme_flow_analysis()
