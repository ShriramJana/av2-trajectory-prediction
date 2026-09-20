import torch

from trajpred.models.constant_velocity import constant_velocity
from trajpred.models.gru import GRUPredictor
from trajpred.models.multimodal import MultiModalGRU
from trajpred.models.polyline import PolylineNet


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


def small_polyline_net(**flags) -> PolylineNet:
    torch.manual_seed(0)
    return PolylineNet(d_model=32, n_layers=2, n_heads=4, **flags).eval()


def test_polyline_net_shapes_and_grads():
    for flags in ({}, {"use_map": False}, {"use_social": False}):
        assert_trains(small_polyline_net(**flags).train(), random_batch(), k=6)


def test_polyline_net_ignores_padding():
    """Appending padded (mask=False) agents and lanes must not change the output,
    even when the padded rows are filled with garbage."""
    model, batch = small_polyline_net(), random_batch()
    b = len(batch["future"])
    padded = dict(batch)
    padded["agent_hist"] = torch.cat([batch["agent_hist"], torch.full((b, 3, 50, 2), 1e3)], dim=1)
    padded["agent_mask"] = torch.cat([batch["agent_mask"], torch.zeros(b, 3, 50, dtype=torch.bool)], dim=1)
    padded["lanes"] = torch.cat([batch["lanes"], torch.full((b, 3, 20, 2), -1e3)], dim=1)
    padded["lane_mask"] = torch.cat([batch["lane_mask"], torch.zeros(b, 3, dtype=torch.bool)], dim=1)
    padded["lane_is_intersection"] = torch.cat(
        [batch["lane_is_intersection"], torch.ones(b, 3, dtype=torch.bool)], dim=1
    )
    with torch.no_grad():
        traj, logits = model(batch)
        traj_p, logits_p = model(padded)
    assert torch.allclose(traj, traj_p, atol=1e-5) and torch.allclose(logits, logits_p, atol=1e-5)


def test_polyline_net_ignores_unobserved_steps():
    """Values stored at masked-out timesteps of a real agent must not matter."""
    model, batch = small_polyline_net(), random_batch()
    garbage = dict(batch)
    garbage["agent_hist"] = torch.where(
        batch["agent_mask"][..., None], batch["agent_hist"], torch.tensor(777.0)
    )
    with torch.no_grad():
        assert torch.allclose(model(batch)[0], model(garbage)[0], atol=1e-5)


def test_polyline_net_ablation_flags():
    batch = random_batch()
    other = random_batch(seed=1)
    changed_lanes = {**batch, "lanes": other["lanes"]}
    changed_agents = {**batch, "agent_hist": batch["agent_hist"].clone()}
    changed_agents["agent_hist"][:, 1:] = other["agent_hist"][:, 1:]  # focal untouched

    with torch.no_grad():
        full = small_polyline_net()
        assert not torch.allclose(full(batch)[0], full(changed_lanes)[0], atol=1e-5)
        assert not torch.allclose(full(batch)[0], full(changed_agents)[0], atol=1e-5)

        no_map = small_polyline_net(use_map=False)
        assert torch.allclose(no_map(batch)[0], no_map(changed_lanes)[0], atol=1e-5)

        no_social = small_polyline_net(use_social=False)
        assert torch.allclose(no_social(batch)[0], no_social(changed_agents)[0], atol=1e-5)
