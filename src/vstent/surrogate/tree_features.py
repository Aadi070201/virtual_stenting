"""Tree-surrogate input features (P3): the single pre-processing path for training, ONNX and
the app. Torch-free.

A batch of trees is padded to ``MAX_BRANCHES`` branches of ``N_POINTS`` samples each; branch
0 is the root and every parent comes before its children (as in ``VesselTree``). Raw inputs
are SI (m, Pa, Pa·s/m³); features are dimensionless or in mm / mmHg and are standardised with
training-set statistics saved to JSON. Per branch:

- point channels (B, M, 4, N), as for the single-vessel surrogate but relative to the
  branch's own inlet reference: r / r_ref(0), log(r / r_ref(0)), s / L, d(A/A_ref)/ds (1/mm);
- branch features (B, M, 7): L (mm), r_ref(0) (mm), ln R_out (R_out in Pa·s/m³), take-off
  position on the parent (fraction of its length; 0 for the root), root flag, Pa (mmHg),
  Pv (mmHg).

The structure is given as a parent one-hot matrix (B, M, M), a one-hot of the take-off
sample on the parent (B, M, N) and a mask of real branches (B, M). Padded branches have zero
features (after standardisation) and zero structure rows.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vstent.data.synth_tree import ABSENT, MAX_BRANCHES
from vstent.physics.geometry import VesselProfile
from vstent.physics.tree import VesselTree
from vstent.units import m_to_mm, pa_to_mmhg

N_POINTS = 256
TREE_FEATURE_VERSION = 1
TREE_NORM_STATS = "tree_norm_stats.json"
POINT_CHANNELS = ("r_over_r0", "log_r_over_r0", "s_over_L", "d_area_ratio_ds_per_mm")
BRANCH_FEATURES = (
    "length_mm",
    "r_ref0_mm",
    "log_r_outlet",
    "take_off_fraction",
    "is_root",
    "pa_mmhg",
    "pv_mmhg",
)
RAW_TREE_KEYS = ("r", "r_ref", "length", "parent", "attach", "pa", "pv", "r_outlet")


@dataclass(frozen=True, eq=False)
class TreeRawInputs:
    """Batch of trees in SI units, padded to M branches of N samples.

    Attributes:
        r, r_ref: lumen and reference radius (m), (B, M, N); 0 for absent branches.
        length: branch length (m), (B, M); s = linspace(0, length, N).
        parent: parent branch (B, M), int; -1 for the root, ``ABSENT`` for padding.
        attach: take-off sample on the parent's N-point grid (B, M), int; 0 for root / padding.
        pa, pv: aortic and venous pressure (Pa), (B,).
        r_outlet: outlet resistance of every branch (Pa·s/m³), (B, M); 0 for padding.
    """

    r: np.ndarray
    r_ref: np.ndarray
    length: np.ndarray
    parent: np.ndarray
    attach: np.ndarray
    pa: np.ndarray
    pv: np.ndarray
    r_outlet: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        """Real (not padded) branches, bool (B, M)."""
        return self.parent != ABSENT

    def __len__(self) -> int:
        return self.r.shape[0]

    def subset(self, idx) -> "TreeRawInputs":
        """Trees selected by an index, slice or boolean mask (array shapes kept)."""
        idx = np.atleast_1d(np.arange(len(self))[idx])
        return type(self)(*(getattr(self, k)[idx] for k in self.__dataclass_fields__))

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "TreeRawInputs":
        """From a dict with the keys of a synthetic-tree split (see ``data.synth_tree``)."""
        return cls(
            r=np.asarray(arrays["r"], np.float64),
            r_ref=np.asarray(arrays["r_ref"], np.float64),
            length=np.asarray(arrays["length"], np.float64),
            parent=np.asarray(arrays["parent"], np.int64),
            attach=np.asarray(arrays["attach"], np.int64),
            pa=np.asarray(arrays["pa"], np.float64),
            pv=np.asarray(arrays["pv"], np.float64),
            r_outlet=np.asarray(arrays["r_outlet"], np.float64),
        )

    @classmethod
    def from_trees(
        cls,
        trees: list[VesselTree],
        pa,
        pv,
        r_outlet: list[np.ndarray],
        n: int = N_POINTS,
        max_branches: int = MAX_BRANCHES,
    ) -> "TreeRawInputs":
        """Trees of any resolution, each branch resampled to ``n`` points; a take-off keeps its
        position along the parent. Pressures in Pa (scalar or one per tree), R_out per branch."""
        b = len(trees)
        pa = np.broadcast_to(np.asarray(pa, np.float64), (b,)).copy()
        pv = np.broadcast_to(np.asarray(pv, np.float64), (b,)).copy()
        out = {
            "r": np.zeros((b, max_branches, n)),
            "r_ref": np.zeros((b, max_branches, n)),
            "length": np.zeros((b, max_branches)),
            "parent": np.full((b, max_branches), ABSENT, np.int64),
            "attach": np.zeros((b, max_branches), np.int64),
            "r_outlet": np.zeros((b, max_branches)),
        }
        for i, (tree, rout) in enumerate(zip(trees, r_outlet, strict=True)):
            if tree.n_branches > max_branches:
                raise ValueError(
                    f"tree has {tree.n_branches} branches; the surrogate takes at most "
                    f"{max_branches}"
                )
            for k, br in enumerate(tree.branches):
                p = VesselProfile(br.s, br.r, br.reference_radius())
                p = p.resample(n) if p.n != n else p
                out["r"][i, k], out["r_ref"][i, k] = p.r, p.r_ref
                out["length"][i, k] = p.length
                out["parent"][i, k] = tree.parent[k]
                out["r_outlet"][i, k] = rout[k]
                if k > 0:
                    par = tree.branches[tree.parent[k]]
                    frac = par.s[tree.attach[k]] / par.length
                    out["attach"][i, k] = int(np.clip(round(frac * (n - 1)), 1, n - 2))
        return cls(pa=pa, pv=pv, **out)


def point_features(raw: TreeRawInputs) -> np.ndarray:
    """Unstandardised per-point channels, float64 (B, M, 4, N); 0 for absent branches."""
    m = raw.mask
    r = np.where(m[..., None], raw.r, 1.0)
    r_ref = np.where(m[..., None], raw.r_ref, 1.0)
    n = r.shape[-1]
    ratio = r / r_ref[..., :1]
    s_over_l = np.broadcast_to(np.linspace(0.0, 1.0, n), r.shape)
    ds_mm = m_to_mm(np.where(m, raw.length, 1.0)) / (n - 1)
    d_area = np.gradient((r / r_ref) ** 2, axis=-1) / ds_mm[..., None]
    feats = np.stack([ratio, np.log(ratio), s_over_l, d_area], axis=2)
    return np.where(m[..., None, None], feats, 0.0)


def branch_features(raw: TreeRawInputs) -> np.ndarray:
    """Unstandardised branch features, float64 (B, M, 7); 0 for absent branches."""
    m = raw.mask
    b, mm = m.shape
    n = raw.r.shape[-1]
    root = raw.parent == -1
    feats = np.stack(
        [
            m_to_mm(raw.length),
            m_to_mm(raw.r_ref[..., 0]),
            np.log(np.where(m, raw.r_outlet, 1.0)),
            np.where(root, 0.0, raw.attach / (n - 1)),
            root.astype(np.float64),
            np.broadcast_to(pa_to_mmhg(raw.pa)[:, None], (b, mm)),
            np.broadcast_to(pa_to_mmhg(raw.pv)[:, None], (b, mm)),
        ],
        axis=-1,
    )
    return np.where(m[..., None], feats, 0.0)


def structure(raw: TreeRawInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parent one-hot (B, M, M), take-off one-hot on the parent (B, M, N), mask (B, M);
    float32."""
    b, mm = raw.parent.shape
    n = raw.r.shape[-1]
    child = raw.parent >= 0
    parent = np.zeros((b, mm, mm), np.float32)
    bi, ki = np.nonzero(child)
    parent[bi, ki, raw.parent[bi, ki]] = 1.0
    attach = np.zeros((b, mm, n), np.float32)
    attach[bi, ki, raw.attach[bi, ki]] = 1.0
    return parent, attach, raw.mask.astype(np.float32)


@dataclass(frozen=True)
class TreeNormStats:
    """Training-set mean / std of every feature over real branches (JSON-serialisable)."""

    point_mean: list[float]
    point_std: list[float]
    branch_mean: list[float]
    branch_std: list[float]
    n_points: int = N_POINTS
    max_branches: int = MAX_BRANCHES
    feature_version: int = TREE_FEATURE_VERSION
    point_channels: tuple[str, ...] = POINT_CHANNELS
    branch_features: tuple[str, ...] = BRANCH_FEATURES

    @classmethod
    def fit(cls, raw: TreeRawInputs) -> "TreeNormStats":
        """Statistics over real branches (and, for point channels, all positions)."""
        m = raw.mask
        xp = point_features(raw)[m]  # (n_branches, 4, N)
        xb = branch_features(raw)[m]  # (n_branches, 7)
        point_std, branch_std = xp.std(axis=(0, 2)), xb.std(axis=0)
        return cls(
            point_mean=xp.mean(axis=(0, 2)).tolist(),
            point_std=np.where(point_std > 0, point_std, 1.0).tolist(),
            branch_mean=xb.mean(axis=0).tolist(),
            branch_std=np.where(branch_std > 0, branch_std, 1.0).tolist(),
            n_points=raw.r.shape[-1],
            max_branches=raw.r.shape[1],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "TreeNormStats":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["point_channels"] = tuple(data["point_channels"])
        data["branch_features"] = tuple(data["branch_features"])
        stats = cls(**data)
        if stats.feature_version != TREE_FEATURE_VERSION:
            raise ValueError(
                f"{path}: tree feature version {stats.feature_version} != {TREE_FEATURE_VERSION}"
            )
        return stats


def prepare_tree_inputs(
    raw: TreeRawInputs, stats: TreeNormStats
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardised model inputs, float32: x_point (B, M, 4, N), x_branch (B, M, 7),
    parent one-hot (B, M, M), take-off one-hot (B, M, N), mask (B, M)."""
    if raw.r.shape[1:] != (stats.max_branches, stats.n_points):
        raise ValueError(
            f"trees are padded to {raw.r.shape[1:]}; the model expects "
            f"({stats.max_branches}, {stats.n_points})"
        )
    m = raw.mask[..., None]
    xp = (point_features(raw) - np.array(stats.point_mean)[:, None]) / np.array(stats.point_std)[
        :, None
    ]
    xb = (branch_features(raw) - np.array(stats.branch_mean)) / np.array(stats.branch_std)
    xp = np.where(m[..., None], xp, 0.0)
    xb = np.where(m, xb, 0.0)
    parent, attach, mask = structure(raw)
    return xp.astype(np.float32), xb.astype(np.float32), parent, attach, mask
