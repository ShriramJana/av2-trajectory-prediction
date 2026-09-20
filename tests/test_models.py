import torch

from trajpred.models.constant_velocity import constant_velocity


def test_cv_extrapolates_last_velocity():
    batch = {
        "agent_hist": torch.zeros(1, 1, 50, 2),
        "agent_mask": torch.ones(1, 1, 50, dtype=torch.bool),
    }
    batch["agent_hist"][0, 0, :, 0] = torch.arange(50) * 2.0  # 2 m/step in +x
    traj, probs = constant_velocity(batch)
    assert traj.shape == (1, 1, 60, 2) and probs.shape == (1, 1)
    assert torch.allclose(traj[0, 0, 0], torch.tensor([100.0, 0.0]))  # 98 + 2
    assert torch.allclose(traj[0, 0, -1], torch.tensor([218.0, 0.0]))  # 98 + 60*2
