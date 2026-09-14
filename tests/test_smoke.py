import torch

import trajpred  # noqa: F401
from trajpred.env import pick_device


def test_accelerator_usable():
    device = pick_device()
    x = torch.ones(2, device=device)
    assert x.sum().item() == 2.0
    print(f"device: {device}")
