"""1D U-Net FFR surrogate with FiLM conditioning and a monotone (physics-consistent) head.

Input: standardised per-point features (B, C, N) and global features (B, G) from
``surrogate.features``. Output: FFR(s) (B, N), dimensionless, with FFR(0) = 1 and
FFR non-increasing along s by construction.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vstent.config import StrictModel
from vstent.surrogate.features import GLOBAL_FEATURES, POINT_CHANNELS

HEAD_INIT_BIAS = -1.5
"""softplus(-1.5) ≈ 0.20, so an untrained model starts with ΔP(L)/Pa ≈ 0.2 (distal FFR ≈ 0.8)."""


class ModelConfig(StrictModel):
    """Architecture hyper-parameters."""

    in_channels: int = len(POINT_CHANNELS)
    global_dim: int = len(GLOBAL_FEATURES)
    base_channels: int = 32
    levels: int = 4
    groups: int = 8
    cond_dim: int = 128


def monotone_ffr_head(z: torch.Tensor) -> torch.Tensor:
    """Map raw outputs z (B, N) to FFR (B, N).

    The N−1 increments softplus(z[:, 1:]) / (N−1) ≥ 0 are the normalised pressure drops ΔP/Pa
    over each segment; FFR = 1 − [0, cumsum(increments)]. Computed in float32 even under
    autocast.
    """
    z = z.float()
    increments = F.softplus(z[:, 1:]) / (z.shape[1] - 1)
    drop = torch.cumsum(increments, dim=1)
    return 1.0 - torch.cat([torch.zeros_like(drop[:, :1]), drop], dim=1)


class FiLMBlock(nn.Module):
    """Two conv-GroupNorm-GELU layers with a residual path; FiLM after the first norm."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.film = nn.Linear(cond_dim, 2 * c_out)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h = self.act(self.norm1(self.conv1(x)) * (1.0 + gamma) + beta)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class FFRUNet(nn.Module):
    """U-Net over the centreline with FiLM conditioning on global features.

    Encoder levels have base·2^i channels (i = 0..levels−1). At the bottleneck, mean- and
    max-pooled features (a learned whole-vessel summary: flow depends on the total resistance)
    update the conditioning vector used by the decoder.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        chans = [cfg.base_channels * 2**i for i in range(cfg.levels)]
        self.cond = nn.Sequential(
            nn.Linear(cfg.global_dim, cfg.cond_dim),
            nn.GELU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        self.encoder = nn.ModuleList(
            FiLMBlock(
                cfg.in_channels if i == 0 else chans[i - 1], chans[i], cfg.cond_dim, cfg.groups
            )
            for i in range(cfg.levels)
        )
        self.down = nn.MaxPool1d(2)
        self.context = nn.Sequential(
            nn.Linear(2 * chans[-1] + cfg.cond_dim, cfg.cond_dim),
            nn.GELU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        self.up = nn.ModuleList(
            nn.ConvTranspose1d(chans[i], chans[i - 1], kernel_size=2, stride=2)
            for i in range(cfg.levels - 1, 0, -1)
        )
        self.decoder = nn.ModuleList(
            FiLMBlock(2 * chans[i - 1], chans[i - 1], cfg.cond_dim, cfg.groups)
            for i in range(cfg.levels - 1, 0, -1)
        )
        self.out = nn.Conv1d(chans[0], 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, HEAD_INIT_BIAS)

    def forward(self, x_point: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        """x_point (B, C, N) and x_global (B, G), standardised -> FFR (B, N).

        N must be divisible by 2^(levels−1).
        """
        cond = self.cond(x_global)
        skips = []
        h = x_point
        for i, block in enumerate(self.encoder):
            h = block(h, cond)
            if i < len(self.encoder) - 1:
                skips.append(h)
                h = self.down(h)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1), cond], dim=1)
        cond = cond + self.context(pooled)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips), strict=True):
            h = block(torch.cat([up(h), skip], dim=1), cond)
        return monotone_ffr_head(self.out(h).squeeze(1))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
