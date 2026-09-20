import torch

from trajpred.models.constant_velocity import constant_velocity
from trajpred.models.gru import GRUPredictor
from trajpred.models.multimodal import MultiModalGRU


def random_batch(b: int = 4, n: int = 6, l: int = 9, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    agent_mask = torch.rand(b, n, 50, generator=g) > 0.3
    agent_mask[:, 0] = True  # focal is always fully observed
    agent_mask[:, :, 49] = True  # kept agents are visible at the last step
    return {
        "agent_hist": torch.randn(b, n, 50, 2, generator=g) * agent_mask[..., None],
        "agent_mask": agent_mask,
        "lanes": torch.randn(b, l, 20, 2, generator=g) * 10,
        "lane_mask": torch.ones(b, l, dtype=torch.bool),
        "lane_is_intersection": torch.rand(b, l, generator=g) > 0.5,
        "future": torch.randn(b, 60, 2, generator=g),
    }


def assert_trains(model, batch, k: int) -> None:
    """Output contract + every parameter receives a finite gradient."""
    traj, logits = model(batch)
    assert traj.shape == (len(batch["future"]), k, 60, 2) and logits.shape == traj.shape[:2]
    assert torch.isfinite(traj).all() and torch.isfinite(logits).all()
    loss = (traj - batch["future"][:, None]).abs().mean() + logits.logsumexp(-1).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


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


def test_gru_predictor_shapes_and_grads():
    assert_trains(GRUPredictor(hidden=32), random_batch(), k=1)


def test_multimodal_gru_shapes_and_grads():
    assert_trains(MultiModalGRU(hidden=32, k=6), random_batch(), k=6)
