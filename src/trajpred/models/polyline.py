"""S3: VectorNet-style polyline transformer.

Agent histories and lane centerlines are both polylines. Each polyline is
squeezed into one token by a small PointNet-style encoder, then self-attention
over the token set mixes social and map context in a single mechanism. The
focal agent's token feeds the same K-mode head as S2.

Padding is kept inert in two places: invalid points are excluded from the
max-pool inside each polyline, and invalid polylines are excluded from
attention via key_padding_mask.
"""

import torch
from torch import Tensor, nn

from trajpred.models.multimodal import KModeHead

AGENT_FEATS = 6  # x, y, dx, dy, dx/dy valid, time
LANE_FEATS = 5  # x, y, dx, dy, is_intersection


def masked_max(h: Tensor, mask: Tensor) -> Tensor:
    """Max over dim 1 of h (M, P, D) using only points where mask (M, P) is True.
    Polylines with no valid point pool to zeros (they are masked out downstream)."""
    pooled = h.masked_fill(~mask[..., None], float("-inf")).amax(dim=1)
    return torch.where(mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))


class SubgraphBlock(nn.Module):
    """Per-point MLP, then append the polyline-wide max so every point sees its
    whole polyline. Output width is 2 * d_out."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU())

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.mlp(x)
        pooled = masked_max(h, mask)
        return torch.cat([h, pooled[:, None].expand_as(h)], dim=-1)


class PolylineEncoder(nn.Module):
    """(B, M, P, d_in) points + (B, M, P) mask -> (B, M, d_model) tokens."""

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.block1 = SubgraphBlock(d_in, d_model // 2)
        self.block2 = SubgraphBlock(d_model, d_model // 2)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b, m = x.shape[:2]
        x, mask = x.flatten(0, 1), mask.flatten(0, 1)
        h = self.block2(self.block1(x, mask), mask)
        return masked_max(h, mask).view(b, m, -1)


def agent_features(hist: Tensor, mask: Tensor) -> Tensor:
    """(B, N, 50, 2) positions -> (B, N, 50, 6) per-step features."""
    disp = torch.zeros_like(hist)
    disp[:, :, 1:] = hist[:, :, 1:] - hist[:, :, :-1]
    # a displacement is only meaningful if both of its endpoints were observed
    disp_valid = torch.zeros_like(mask)
    disp_valid[:, :, 1:] = mask[:, :, 1:] & mask[:, :, :-1]
    disp = disp * disp_valid[..., None]
    # max-pooling is order-blind, so time has to be a feature
    t = torch.linspace(-1.0, 0.0, hist.shape[2], device=hist.device).expand(mask.shape)
    return torch.cat([hist, disp, disp_valid[..., None].float(), t[..., None]], dim=-1)


def lane_features(lanes: Tensor, is_intersection: Tensor) -> Tensor:
    """(B, L, 20, 2) centerlines -> (B, L, 20, 5); dx/dy carries travel direction."""
    direction = torch.zeros_like(lanes)
    direction[:, :, :-1] = lanes[:, :, 1:] - lanes[:, :, :-1]
    direction[:, :, -1] = direction[:, :, -2]
    flag = is_intersection[:, :, None, None].expand(*lanes.shape[:3], 1).float()
    return torch.cat([lanes, direction, flag], dim=-1)


class PolylineNet(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        k: int = 6,
        dropout: float = 0.1,
        use_map: bool = True,
        use_social: bool = True,
    ):
        super().__init__()
        self.use_map, self.use_social = use_map, use_social
        self.agent_encoder = PolylineEncoder(AGENT_FEATS, d_model)
        if use_map:
            self.lane_encoder = PolylineEncoder(LANE_FEATS, d_model)
        self.type_embed = nn.Embedding(3, d_model)  # focal / other agent / lane
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = KModeHead(d_model, k)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        hist, agent_mask = batch["agent_hist"], batch["agent_mask"]
        if not self.use_social:  # ablation: the focal agent is alone in the world
            hist, agent_mask = hist[:, :1], agent_mask[:, :1]

        tokens = self.agent_encoder(agent_features(hist, agent_mask), agent_mask)
        token_type = torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device)
        token_type[:, 0] = 0
        valid = agent_mask.any(dim=-1)  # (B, N): padded agents have no observed step

        if self.use_map:
            lanes, lane_mask = batch["lanes"], batch["lane_mask"]
            point_mask = lane_mask[..., None].expand(*lanes.shape[:3])
            lane_tokens = self.lane_encoder(
                lane_features(lanes, batch["lane_is_intersection"]), point_mask
            )
            tokens = torch.cat([tokens, lane_tokens], dim=1)
            token_type = torch.cat([token_type, torch.full_like(lane_mask, 2, dtype=torch.long)], dim=1)
            valid = torch.cat([valid, lane_mask], dim=1)

        tokens = tokens + self.type_embed(token_type)
        tokens = self.transformer(tokens, src_key_padding_mask=~valid)
        return self.head(self.norm(tokens[:, 0]))  # focal agent is token 0
