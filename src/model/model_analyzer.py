from pathlib import Path
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
            for batch_data in tqdm(self.dataloader, desc="Processing batches"):
                batch = {
                    k: v.to(self.cfg.device)
                    for k, v in batch_data.items()
                    if not (isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.datetime64))
                }
                model_output = self.model(batch)
                all_hidden_states.append(model_output["h_n"].squeeze(1).cpu())

        return torch.cat(all_hidden_states)
