import os
from pathlib import Path
import torch
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.datautils.utils import load_scaler

class BaseModelLoader:
    """
    Base class for loading configuration, scaler, and model.
    """
    def __init__(self, run_dir: Path, epoch: int):
        self.run_dir = run_dir
        self.epoch = epoch
        self.cfg = Config(self.run_dir / "config.yml")
        self.scaler = load_scaler(self.run_dir)
        self.model = self._load_model()

    def _load_model(self) -> CudaLSTM:
        model = CudaLSTM(cfg=self.cfg)
        model_path = self.run_dir / f"model_epoch{self.epoch:03d}.pt"
        model_weights = torch.load(str(model_path), map_location=self.cfg.device, weights_only=True)
        model.load_state_dict(model_weights)
        model.eval()
        return model.to(self.cfg.device)
