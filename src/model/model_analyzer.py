import gc
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralhydrology.datasetzoo import get_dataset

from model.base_model_loader import BaseModelLoader


class ModelAnalyzer(BaseModelLoader):
    """
    Analyzes neural hydrology models by extracting hidden states.
    """
    def __init__(self, run_dir: Path, epoch: int, period: str = "test"):
        super().__init__(run_dir, epoch)
        self.period = period
        self.dataset = None
        self.dataloader = None

    def get_dataset(self):
        """
        Lazy load the dataset and dataloader if not already loaded.
        """
        if self.dataset is None:
            self.dataset = get_dataset(
                self.cfg, 
                is_train=False, 
                period=self.period, 
                scaler=self.scaler
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self.dataset.collate_fn
            )
        return self.dataset, self.dataloader

    def get_hidden_states(self) -> torch.Tensor:
        """
        Extract the hidden states (h_n) of the model from the dataset.
        
        Returns:
            torch.Tensor: Concatenated hidden states tensor.
        """
        self.get_dataset()
        all_hidden_states = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.dataloader, desc="Loading hidden states"):
                batch = {
                    k: v.to(self.cfg.device)
                    for k, v in batch_data.items()
                    if not (isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.datetime64))
                }
                model_output = self.model(batch)
                all_hidden_states.append(model_output["h_n"].squeeze(1).cpu())

        return torch.cat(all_hidden_states)

    def get_inputs(self, fetch_target: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract and combine the dynamic and static input features from the dataset.
        Optionally also returns the target variable 'y' extracted from the last time step.
        
        Args:
            fetch_target (bool, optional): Whether to include the target variable 'y'. Defaults to False.

        Returns:
            dict: {
                "x_d": np.ndarray of shape [num_samples, T, D],
                "x_s": np.ndarray of shape [num_samples, S],
                "y": np.ndarray of shape [num_samples, 1]  (only if fetch_target is True)
            }
            where:
                - num_samples is the total number of samples,
                - T is the number of timesteps (sequence length),
                - D is the number of dynamic features,
                - S is the number of static features,
        """
        self.get_dataset()
        self.model.eval()

        first_batch = next(iter(self.dataloader))
        batch_x_d = first_batch['x_d'].detach().cpu().numpy()  # shape: [batch, T, D]
        batch_x_s = first_batch['x_s'].detach().cpu().numpy()  # shape: [batch, S]
        _, T, D = batch_x_d.shape
        _, S = batch_x_s.shape

        if fetch_target:
            batch_y = first_batch['y'].detach().cpu().numpy()  # shape: [batch, T, 1]
            # Extract target from the last time step, resulting in shape [batch, 1]
            batch_y_last = batch_y[:, -1, :]
            _, Y = batch_y_last.shape  # Y should be 1

        # Determine total number of samples
        num_samples = len(self.dataset)
        
        # Pre-allocate the output arrays
        x_d_full = np.empty((num_samples, T, D), dtype=batch_x_d.dtype)
        x_s_full = np.empty((num_samples, S), dtype=batch_x_s.dtype)
        if fetch_target:
            y_full = np.empty((num_samples, 1), dtype=batch_y_last.dtype)
        
        start_idx = 0
        with torch.no_grad():
            for batch_data in tqdm(self.dataloader, desc="Extracting inputs"):
                batch_x_d = batch_data['x_d'].detach().cpu().numpy()  # [batch, T, D]
                batch_x_s = batch_data['x_s'].detach().cpu().numpy()  # [batch, S]
                batch_size = batch_x_d.shape[0]
                
                x_d_full[start_idx:start_idx + batch_size] = batch_x_d
                x_s_full[start_idx:start_idx + batch_size] = batch_x_s
                
                if fetch_target:
                    batch_y = batch_data['y'].detach().cpu().numpy()  # [batch, T, 1]
                    # Take the last time step (target), which gives shape [batch, 1]
                    batch_y_last = batch_y[:, -1, :]
                    y_full[start_idx:start_idx + batch_size] = batch_y_last
                    
                start_idx += batch_size

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

        inputs = {"x_d": x_d_full, "x_s": x_s_full}
        if fetch_target:
            inputs["y"] = y_full

        logging.info(
            f"Extracted inputs: x_d shape {x_d_full.shape}, x_s shape {x_s_full.shape}" +
            (f", y shape {y_full.shape}" if fetch_target else "")
        )
        return inputs

    def get_unique_static_inputs(self) -> Dict[str, np.ndarray | pd.Index] | None:
        """
        Extracts unique static input features and corresponding basin IDs.

        Returns:
            dict: {
                "x_s_unique": np.ndarray of shape [num_basins, S],
                "basin_ids": pd.Index object containing unique basin IDs
            }
            Returns None if lookup_table is not available or data extraction fails.
        """
        dataset, _ = self.get_dataset()

        if not hasattr(dataset, 'lookup_table'):
            logging.error("Dataset object does not have the 'lookup_table' attribute needed to map samples to basin IDs.")
            return None

        inputs = self.get_inputs(fetch_target=False)

        x_s = inputs['x_s'] # Shape [N_samples, S]
        num_total_samples = x_s.shape[0]
        lookup_table = dataset.lookup_table

        try:
            # Extract basin IDs corresponding to each row in all_x_s
            basin_ids_per_sample = np.array([lookup_table[i][0] for i in range(num_total_samples)])
        except IndexError:
             logging.error("Index error while accessing lookup_table. Ensure it's correctly structured.")
             return None
        except Exception as e:
            logging.error(f"Error processing lookup_table: {e}")
            return None

        temp_df = pd.DataFrame(x_s)
        temp_df['basin_id'] = basin_ids_per_sample

        # Keep only the first occurrence of each basin_id
        unique_rows_df = temp_df.drop_duplicates(subset=['basin_id'], keep='first')
        basin_ids_unique = pd.Index(unique_rows_df['basin_id'])
        x_s_unique = unique_rows_df.drop(columns=['basin_id']).values
        num_unique_basins = x_s_unique.shape[0]

        logging.info(f"Found {num_unique_basins} unique basins.")
        logging.info(f"Extracted unique static data using lookup_table. Shape: {x_s_unique.shape}")

        # Perform a sanity check on the number of features
        sorted_static_features = sorted(self.cfg.static_attributes)
        if x_s_unique.shape[1] != len(sorted_static_features):
             logging.warning(f"Number of columns in unique static features ({x_s_unique.shape[1]}) "
                             f"does not match expected number ({len(sorted_static_features)}).")

        return {"x_s_unique": x_s_unique, "basin_ids": basin_ids_unique}
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dir = Path("/sci/labs/efratmorin/eli.levinkopf/batch_runs/runs/train_lstm_rs_22_1503_194719")
    epoch = 25
    analyzer = ModelAnalyzer(run_dir, epoch)
    inputs = analyzer.get_unique_static_inputs()