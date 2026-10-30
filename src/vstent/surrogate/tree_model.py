"""Tree FFR surrogate (P3): a 1D U-Net per branch with message passing between branches.

Inputs come from ``surrogate.tree_features`` (padded to M branches of N samples). Every branch
goes through the same FiLM-conditioned 1D U-Net as the single-vessel surrogate (M3). At the
bottleneck, each branch's summary (its conditioning plus pooled features) is updated by a few
rounds of message passing along the tree in plain PyTorch: from the children (sum), from the
parent and from the whole tree (masked mean), because the flow in a branch depends on every
other branch. The decoder then gives each branch non-negative pressure-drop increments
(softplus), as in M3's monotone head. FFR is assembled down the tree: a branch starts at its
parent's FFR at the take-off sample. So FFR(0) = 1 at the inlet, FFR is continuous at every
junction and non-increasing along every root-to-outlet path, by construction.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vstent.config import StrictModel
from vstent.surrogate.model import HEAD_INIT_BIAS, FiLMBlock
from vstent.surrogate.tree_features import BRANCH_FEATURES, POINT_CHANNELS


class TreeModelConfig(StrictModel):
    """Architecture hyper-parameters."""

    in_channels: int = len(POINT_CHANNELS)
    branch_dim: int = len(BRANCH_FEATURES)
    base_channels: int = 32
    levels: int = 4
    groups: int = 8
    cond_dim: int = 128
    message_layers: int = 3


class MessageLayer(nn.Module):
    """One round of message passing on branch summaries h (B, M, D)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.update = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, h: torch.Tensor, parent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``parent`` (B, M, M): parent[b, k, j] = 1 if j is k's parent; ``mask`` (B, M)."""
        from_children = parent.transpose(1, 2) @ h  # sum over each branch's children
        from_parent = parent @ h
        w = mask.unsqueeze(-1)
        tree = (h * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        msg = torch.cat([h, from_children, from_parent, tree.expand_as(h)], dim=-1)
        return h + self.update(self.norm(msg)) * w


class TreeFFRNet(nn.Module):
    """Branch-wise 1D U-Net + message passing + monotone tree head (see module docstring)."""

    def __init__(self, config: TreeModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or TreeModelConfig()
        self.config = cfg
        chans = [cfg.base_channels * 2**i for i in range(cfg.levels)]
        self.cond = nn.Sequential(
            nn.Linear(cfg.branch_dim, cfg.cond_dim),
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
        self.messages = nn.ModuleList(MessageLayer(cfg.cond_dim) for _ in range(cfg.message_layers))
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

    def forward(
        self,
        x_point: torch.Tensor,
        x_branch: torch.Tensor,
        parent: torch.Tensor,
        attach: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """x_point (B, M, C, N), x_branch (B, M, G), parent (B, M, M), attach (B, M, N),
        mask (B, M) -> FFR (B, M, N). Padded branches give meaningless values (mask them)."""
        b, m, c, n = x_point.shape
        cond = self.cond(x_branch.reshape(b * m, -1))
        skips = []
        h = x_point.reshape(b * m, c, n)
        for i, block in enumerate(self.encoder):
            h = block(h, cond)
            if i < len(self.encoder) - 1:
                skips.append(h)
                h = self.down(h)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1), cond], dim=1)
        node = (cond + self.context(pooled)).reshape(b, m, -1).float()
        for layer in self.messages:
            node = layer(node, parent.float(), mask.float())
        cond = node.reshape(b * m, -1).to(h.dtype)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips), strict=True):
            h = block(torch.cat([up(h), skip], dim=1), cond)
        z = self.out(h).reshape(b, m, n).float()
        return tree_ffr_head(z, parent.float(), attach.float())


def tree_ffr_head(z: torch.Tensor, parent: torch.Tensor, attach: torch.Tensor) -> torch.Tensor:
    """Raw outputs z (B, M, N) -> FFR (B, M, N), assembled down the tree in float32.

    Within a branch the N−1 increments softplus(z) / (N−1) ≥ 0 are normalised pressure drops,
    as in M3. Branch k starts at its parent's FFR at the take-off sample (the root at 1);
    branches are in topological order (parent index < child index).
    """
    b, m, n = z.shape
    drop = torch.cumsum(F.softplus(z[..., 1:]) / (n - 1), dim=-1)
    drop = torch.cat([torch.zeros_like(drop[..., :1]), drop], dim=-1)
    rows = [1.0 - drop[:, 0]]
    for k in range(1, m):
        prev = torch.stack(rows, dim=1)  # (B, k, N)
        at_take_off = (prev * attach[:, k : k + 1, :]).sum(dim=-1)  # (B, k)
        inlet = (at_take_off * parent[:, k, :k]).sum(dim=-1)  # (B,)
        rows.append(inlet.unsqueeze(-1) - drop[:, k])
    return torch.stack(rows, dim=1)
