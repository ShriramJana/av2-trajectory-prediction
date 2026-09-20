"""S1: unimodal GRU encoder-decoder over the focal agent's history only."""

import torch
from torch import Tensor, nn


class FocalGRUEncoder(nn.Module):
    """Focal history -> one feature vector. Shared by S1 and S2.

    Input is per-step displacements, not positions: displacements are small,
    roughly stationary numbers (a velocity signal), whereas positions drift
    from -100 m to 0 over the window and the network would have to
    differentiate them itself.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        hist = batch["agent_hist"][:, 0]  # (B, 50, 2), focal agent is row 0
        disp = hist[:, 1:] - hist[:, :-1]  # (B, 49, 2)
        _, h = self.gru(disp)
        return h[-1]  # (B, hidden)


class GRUPredictor(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.encoder = FocalGRUEncoder(hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 60 * 2)
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Returns traj (B, 1, 60, 2) and logits (B, 1) (zeros: a single mode)."""
        feat = self.encoder(batch)
        # one-shot decoding: all 60 displacements at once, then integrate.
        # The focal frame's origin is the last observed point, so the cumsum
        # of displacements is already the position.
        disp = self.decoder(feat).view(-1, 1, 60, 2)
        return disp.cumsum(dim=2), torch.zeros(len(feat), 1, device=feat.device)
