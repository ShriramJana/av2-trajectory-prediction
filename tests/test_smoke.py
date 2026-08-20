import torch

import trajpred  # noqa: F401


def test_cuda_available():
    assert torch.cuda.is_available()
