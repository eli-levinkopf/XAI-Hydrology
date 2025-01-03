import argparse
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import yaml
import logging


class ValidationMetricsAnalyzer:
    def __init__(self, log_dir):
        """
        Initializes the ValidationMetricsAnalyzer.

        Args:
            log_dir (str): Path to the training directory containing output.log and config.yml.
        """
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "output.log")
        self.config_file = os.path.join(log_dir, "config.yml")
        self.output_dir = os.path.join(log_dir, "validation_metrics_analysis")

        self._create_directory(self.output_dir)

        self.output_csv = os.path.join(self.output_dir, "validation_metrics_summary.csv")
        self.stats_csv = os.path.join(self.output_dir, "validation_metrics_statistics.csv")
        self.loss_plot_path = os.path.join(self.output_dir, "loss_convergence_plot.png")

        self.metrics = self._load_metrics_from_config()
        self.validation_regex = self._construct_validation_regex()
        self.training_regex = r"Epoch (\d+) average loss: avg_loss: ([\d\.]+)"
        self.epochs = []
        self.avg_losses = []
        self.training_losses = []
        self.metric_values = {metric: [] for metric in self.metrics}

    def _create_directory(self, path):
        """Creates a directory if it doesn't exist."""
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError as e:
            raise RuntimeError(f"Failed to create directory {path}: {e}")

    def _load_metrics_from_config(self):
        """Loads the list of metrics from the config.yml file."""
        try:
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
            return config.get("metrics", [])
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {self.config_file}. Ensure the path is correct.")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse YAML config file: {e}")

    def _construct_validation_regex(self):
        """Constructs a regex pattern to extract validation metrics from the log file."""
        metrics_part = ", ".join([f"{metric}: ([\\d\\.-]+)" for metric in self.metrics])
        return (
            r"Epoch (\d+) average validation loss: ([\d\.]+) -- Median validation metrics: "
            r"avg_loss: ([\d\.]+), " + metrics_part
        )

    def _extract_training_and_validation_metrics(self, line):
        """Extracts training and validation metrics from a line of the log file."""
        val_match = re.search(self.validation_regex, line)
        train_match = re.search(self.training_regex, line)
        
        if val_match:
            epoch = int(val_match.group(1))
            self.epochs.append(epoch)
            self.avg_losses.append(float(val_match.group(2)))
            for i, metric in enumerate(self.metrics, start=4):
                self.metric_values[metric].append(float(val_match.group(i)))

        if train_match:
            epoch = int(train_match.group(1))
            loss = float(train_match.group(2))
            while len(self.training_losses) < epoch:
                self.training_losses.append(None)
            self.training_losses[epoch - 1] = loss

    def _extract_metrics(self):
        """Extracts metrics and training losses from the log file."""
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    self._extract_training_and_validation_metrics(line)

            max_epoch = max(self.epochs) if self.epochs else 0
            self._ensure_list_lengths(max_epoch)
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file not found at {self.log_file}. Ensure the path is correct.")

    def _ensure_list_lengths(self, max_epoch):
        """Ensures all lists are of equal length for consistent analysis."""
        while len(self.training_losses) < max_epoch:
            self.training_losses.append(None)
        while len(self.avg_losses) < max_epoch:
            self.avg_losses.append(None)
        for metric in self.metrics:
            while len(self.metric_values[metric]) < max_epoch:
                self.metric_values[metric].append(None)

    def _save_metrics_to_csv(self):
        """Saves extracted metrics to CSV files."""
        metrics_data = {"Epoch": self.epochs, **self.metric_values}
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(self.output_csv, index=False)

        stats_df = metrics_df.drop(columns=["Epoch"]).describe()
        stats_df.index.name = "Stat"
        stats_df.to_csv(self.stats_csv, index=True)

    def _plot_loss_convergence(self):
        """Plots training and validation loss convergence over epochs."""
        self._plot_series(
            [self.training_losses, self.avg_losses],
            ["Training Loss", "Validation Loss"],
            "Loss",
            self.loss_plot_path,
            "Training and Validation Loss over Epochs",
            colors=["blue", "red"]
        )

    def _plot_metric_progression(self):
        """Plots the progression of each metric over epochs."""
        for metric in self.metrics:
            plot_path = os.path.join(self.output_dir, f"{metric.lower()}_progress_plot.png")
            self._plot_series(
                [self.metric_values[metric]],
                [f"Validation {metric}"],
                metric,
                plot_path,
                f"{metric} Progression During Training"
            )

    def _plot_series(self, series_list, labels, ylabel, save_path, title, colors=None):
        """
        Plots one or more series with the same x-axis (epochs).
        
        Args:
            series_list (list of lists): List of series to plot.
            labels (list of str): Labels for the series.
            ylabel (str): Y-axis label.
            save_path (str): Path to save the plot.
            title (str): Plot title.
            colors (list of str, optional): List of colors for the series.
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(series_list[0]) + 1)

        for i, (series, label) in enumerate(zip(series_list, labels)):
            color = colors[i] if colors else None
            plt.plot(
                epochs,
                [x if x is not None else float('nan') for x in series],
                label=label,
                color=color
            )
        
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.close()

    def analyze(self):
        """Performs the complete analysis pipeline."""
        logging.info("Extracting metrics...")
        self._extract_metrics()

        logging.info("Saving metrics to CSV...")
        self._save_metrics_to_csv()

        logging.info("Plotting loss convergence...")
        self._plot_loss_convergence()

        logging.info("Plotting metrics progression...")
        self._plot_metric_progression()

        logging.info(f"Statistics, metrics, and plots have been successfully saved to {self.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Extract validation metrics from a log file.")
    parser.add_argument("log_dir", type=str, help="Path to the training directory containing output.log and config.yml")
    args = parser.parse_args()

    analyzer = ValidationMetricsAnalyzer(args.log_dir)
    analyzer.analyze()
