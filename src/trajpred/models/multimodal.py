"""S2: K-mode decoder on top of the S1 encoder. The head is reused verbatim by S3."""

from torch import Tensor, nn

from trajpred.models.gru import FocalGRUEncoder


class KModeHead(nn.Module):
    """Feature vector -> K candidate futures plus one confidence logit each."""

    def __init__(self, hidden: int, k: int = 6):
        super().__init__()
        self.k = k
        # per mode: 60 (dx, dy) displacements + 1 logit = 121 numbers
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, k * (60 * 2 + 1))
        )

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        """(B, hidden) -> traj (B, K, 60, 2), logits (B, K)."""
        out = self.mlp(feat).view(-1, self.k, 60 * 2 + 1)
        disp = out[..., :-1].reshape(-1, self.k, 60, 2)
        return disp.cumsum(dim=2), out[..., -1]


class MultiModalGRU(nn.Module):
    def __init__(self, hidden: int = 128, k: int = 6):
        super().__init__()
        self.encoder = FocalGRUEncoder(hidden)
        self.head = KModeHead(hidden, k)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        return self.head(self.encoder(batch))
