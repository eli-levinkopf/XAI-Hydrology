import gc
import logging
from pathlib import Path
from typing import Dict
import numpy as np
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

    def get_inputs(self) -> Dict[str, np.ndarray]:
        """
        Extract and combine the dynamic and static input features from the dataset.
        
        Returns:
            dict: {
                "x_d": np.ndarray of shape [num_samples, T, D],
                "x_s": np.ndarray of shape [num_samples, S]
            }
            where:
                - num_samples is the total number of samples
                - T is the number of timesteps (sequence length)
                - D is the number of dynamic features
                - S is the number of static features 
        """
        self.get_dataset()
        self.model.eval()

        first_batch = next(iter(self.dataloader))
        batch_x_d = first_batch['x_d'].detach().cpu().numpy()  # shape: [batch, T, D]
        batch_x_s = first_batch['x_s'].detach().cpu().numpy()  # shape: [batch, S]
        _, T, D = batch_x_d.shape
        _, S = batch_x_s.shape

        # Determine total number of samples
        num_samples = len(self.dataset)
        
        # Pre-allocate the output arrays
        x_d_full = np.empty((num_samples, T, D), dtype=batch_x_d.dtype)
        x_s_full = np.empty((num_samples, S), dtype=batch_x_s.dtype)

        start_idx = 0
        with torch.no_grad():
            for batch_data in tqdm(self.dataloader, desc="Extracting inputs"):
                batch_x_d = batch_data['x_d'].detach().cpu().numpy()  # shape: [batch, T, D]
                batch_x_s = batch_data['x_s'].detach().cpu().numpy()  # shape: [batch, S]
                batch_size = batch_x_d.shape[0]
                
                x_d_full[start_idx:start_idx + batch_size] = batch_x_d
                x_s_full[start_idx:start_idx + batch_size] = batch_x_s
                start_idx += batch_size

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

        inputs = {"x_d": x_d_full, "x_s": x_s_full}
        logging.info(f"Extracted inputs: x_d shape {x_d_full.shape}, x_s shape {x_s_full.shape}")
        return inputs