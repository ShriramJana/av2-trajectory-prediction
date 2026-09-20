import torch

from trajpred.losses import wta_loss


def test_wta_zero_regression_when_a_mode_is_exact():
    gt = torch.randn(2, 60, 2)
    traj = torch.randn(2, 4, 60, 2)
    traj[:, 2] = gt
    logits = torch.zeros(2, 4)
    # winner is mode 2 -> regression term 0; CE of uniform logits over 4 = log(4)
    assert torch.allclose(wta_loss(traj, logits, gt), torch.log(torch.tensor(4.0)), atol=1e-5)


def test_wta_gradient_only_flows_to_winner():
    gt = torch.zeros(1, 60, 2)
    traj = torch.zeros(1, 2, 60, 2)
    traj[0, 0] += 0.5  # mode 0 is close and wins
    traj[0, 1] += 100.0  # mode 1 is far
    traj.requires_grad_(True)
    wta_loss(traj, torch.zeros(1, 2), gt, alpha=0.0).backward()
    assert traj.grad[0, 0].abs().sum() > 0
    assert traj.grad[0, 1].abs().sum() == 0


def test_wta_classification_targets_the_winner():
    gt = torch.zeros(1, 60, 2)
    traj = torch.zeros(1, 3, 60, 2)
    traj[0, 0] += 5.0
    traj[0, 2] += 9.0  # mode 1 (exact) wins
    confident_right = wta_loss(traj, torch.tensor([[0.0, 10.0, 0.0]]), gt)
    confident_wrong = wta_loss(traj, torch.tensor([[10.0, 0.0, 0.0]]), gt)
    assert confident_right < 0.01 < confident_wrong
