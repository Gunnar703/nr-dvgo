from torch.optim.optimizer import Optimizer
from typing import Any
from .dvgo import DVGO

import torch

def load_checkpoint(
    model: DVGO, 
    optimizer: Optimizer, 
    ckpt_path: str, 
    no_reload_optimizer: bool
) -> "tuple[DVGO, Optimizer, Any]":
    ckpt = torch.load(ckpt_path)
    start = ckpt["global_step"]
    model.load_state_dict(ckpt["model_state_dict"])
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, start

def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path)
    model = DVGO(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model