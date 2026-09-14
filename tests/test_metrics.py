import torch

from trajpred.metrics import brier_min_fde, evaluate_batch, min_ade, min_fde, miss_rate


def make_toy():
    """GT: straight +x at 1 m/step. Modes: exact / 3 m lateral offset / 0.5 m offset."""
    gt = torch.zeros(1, 60, 2)
    gt[0, :, 0] = torch.arange(60) + 1.0
    pred = torch.zeros(1, 3, 60, 2)
    pred[0, 0] = gt[0]
    pred[0, 1] = gt[0] + torch.tensor([0.0, 3.0])
    pred[0, 2, :, 0] = torch.arange(60) + 1.0
    pred[0, 2, :, 1] = 0.5
    return pred, gt


def test_min_ade_picks_exact_mode():
    pred, gt = make_toy()
    assert torch.allclose(min_ade(pred, gt), torch.tensor([0.0]))


def test_min_fde_zero_for_exact():
    pred, gt = make_toy()
    assert torch.allclose(min_fde(pred, gt), torch.tensor([0.0]))


def test_ade_without_exact_mode():
    pred, gt = make_toy()
    # only the 3 m and 0.5 m offset modes remain; best is a constant 0.5 m off
    assert torch.allclose(min_ade(pred[:, 1:], gt), torch.tensor([0.5]))


def test_miss_rate_thresholds():
    pred, gt = make_toy()
    assert miss_rate(pred[:, 1:2], gt).item() == 1.0  # 3 m endpoint error > 2 m
    assert miss_rate(pred[:, 2:3], gt).item() == 0.0  # 0.5 m < 2 m


def test_brier_adds_confidence_penalty():
    pred, gt = make_toy()
    probs = torch.tensor([[0.5, 0.25, 0.25]])
    # best-FDE mode is the exact one (p=0.5): 0 + (1 - 0.5)^2 = 0.25
    assert torch.allclose(brier_min_fde(pred, probs, gt), torch.tensor([0.25]))


def test_evaluate_batch_keys_and_shapes():
    pred, gt = make_toy()
    probs = torch.full((1, 3), 1 / 3)
    out = evaluate_batch(pred, probs, gt)
    assert set(out) == {"minADE", "minFDE", "MR", "brier_minFDE"}
    assert all(v.shape == (1,) for v in out.values())
