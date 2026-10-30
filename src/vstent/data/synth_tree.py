"""Synthetic vessel trees (docs/PLAN.md, P2): a main vessel with 1–4 side branches.

Geometry, in addition to the single-vessel factory (M2, :mod:`vstent.data.synth`):

- **Main vessel.** Reference diameter D0 at the inlet. Within each segment between take-offs it
  tapers linearly by a factor in ``distal_taper``. After every take-off it steps down so that
  d_after^e = d_before^e − d_side^e, with e sampled from ``split_exponent`` (Murray-like).
- **Side branches.** Take-offs are spaced along the main vessel. A side branch starts at
  ``side_diameter_ratio`` × the main's reference diameter at the take-off and tapers linearly to
  ``side_taper`` × that at its end.
- **Lesions.** Shapes, throat DS distribution and lengths are as in M2: 0–3 on the main vessel
  and 0–2 per side branch. A *bifurcation* lesion is centred on a take-off on the main vessel;
  an *ostial* lesion sits at a side branch's start. Optional diffuse disease on the main vessel.
- **Noise.** Smoothed measurement noise on every branch, as in M2.

Labels come from the tree solver (:mod:`vstent.physics.tree`) on exactly the stored float32
radii. The single-path FFR of the main vessel alone (M1 solver, same Pa, Pv and v_hyp) is stored
for comparison. Units: SI internally; config fields ending in ``_mm``, ``_mmhg`` or ``_pct`` are
converted on use. Stored arrays and their units are listed in ``TREE_ARRAY_UNITS``.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import Field, field_validator, model_validator
from scipy.ndimage import gaussian_filter1d

from vstent.config import StrictModel
from vstent.data.synth import (
    SHAPES,
    LesionSpec,
    Range,
    SamplingConfig,
    SplitConfig,
    _modulation,
    _place_lesions,
    _plateau,
    lesion_bump,
)
from vstent.physics.geometry import VesselProfile
from vstent.physics.solver import PhysicsParams, solve_ffr_batch
from vstent.physics.tree import VesselTree, solve_tree_batch
from vstent.units import mm_to_m, mmhg_to_pa

MAX_SIDE = 4
MAX_BRANCHES = 1 + MAX_SIDE
MAX_TREE_LESIONS = 12
LESION_KINDS = ("regular", "bifurcation", "ostial")
ABSENT = -2  # ``parent`` code of a padded (absent) branch

TREE_ARRAY_UNITS = {
    "r": f"m, float32 (n, {MAX_BRANCHES}, N): noisy lumen radius per branch, 0 if absent",
    "r_ref": f"m, float32 (n, {MAX_BRANCHES}, N): true reference radius, 0 if absent",
    "ffr": f"-, float32 (n, {MAX_BRANCHES}, N): tree-solver FFR at every sample, 0 if absent",
    "ffr_main_single": "-, float32 (n, N): M1 single-path FFR of the main vessel alone",
    "length": f"m, float64 (n, {MAX_BRANCHES}): branch length, 0 if absent",
    "parent": f"int8 (n, {MAX_BRANCHES}): parent branch, -1 root, {ABSENT} absent",
    "attach": f"int16 (n, {MAX_BRANCHES}): take-off sample index on the parent, 0 root/absent",
    "branch_count": "int8 (n,)",
    "pa": "Pa, float64 (n,)",
    "pv": "Pa, float64 (n,)",
    "v_hyp": "m/s, float64 (n,)",
    "q_ref": "m³/s, float64 (n,): inflow of the stenosis-free tree",
    "q_outlet": f"m³/s, float64 (n, {MAX_BRANCHES}): outlet flow per branch",
    "q_inlet": f"m³/s, float64 (n, {MAX_BRANCHES}): inflow per branch",
    "q_outlet_healthy": f"m³/s, float64 (n, {MAX_BRANCHES}): Murray outlet flows (healthy)",
    "r_outlet": f"Pa·s/m³, float64 (n, {MAX_BRANCHES}): outlet resistance per branch",
    "r_outlet_clamped": f"bool (n, {MAX_BRANCHES})",
    "noise_sigma_frac": "-, float64 (n,): noise σ / r_ref",
    "lesion_branch": f"int8 (n, {MAX_TREE_LESIONS}): branch of each lesion, -1 padded",
    "lesion_start": f"m, float32 (n, {MAX_TREE_LESIONS}): from the branch start, NaN-padded",
    "lesion_length": f"m, float32 (n, {MAX_TREE_LESIONS}), NaN-padded",
    "lesion_ds_pct": f"%, float32 (n, {MAX_TREE_LESIONS}): nominal throat DS, NaN-padded",
    "lesion_shape": f"int8 (n, {MAX_TREE_LESIONS}): index into SHAPES, -1 padded",
    "lesion_kind": f"int8 (n, {MAX_TREE_LESIONS}): index into LESION_KINDS, -1 padded",
}


def _probs(name: str, probs: list[float], max_len: int) -> list[float]:
    if len(probs) > max_len or min(probs) < 0 or not math.isclose(sum(probs), 1.0, abs_tol=1e-9):
        raise ValueError(f"{name}: at most {max_len} non-negative entries summing to 1")
    return probs


class TreeSamplingConfig(SamplingConfig):
    """Tree sampling distributions; uniform over [low, high] unless noted. May be tuned.

    Inherited fields keep their M2 meaning, applied to the main vessel: ``length_mm`` and
    ``d0_mm`` are its length and inlet diameter, ``lesion_count_probs`` its lesion count, and
    ``distal_taper`` the linear taper *within each segment* between take-offs.
    """

    length_mm: Range = (50.0, 100.0)
    d0_mm: Range = (2.5, 4.5)
    distal_taper: Range = (0.90, 1.0)
    lesion_count_probs: list[float] = [0.15, 0.45, 0.30, 0.10]
    side_count_probs: list[float] = [0.0, 0.30, 0.35, 0.20, 0.15]  # P(0), …, P(4) side branches
    side_margin_mm: float = Field(8.0, ge=0)  # take-offs ≥ this from both ends of the main
    side_spacing_mm: float = Field(8.0, ge=0)  # and ≥ this apart
    side_diameter_ratio: Range = (0.45, 0.85)  # side start diameter / main diameter at take-off
    side_length_mm: Range = (15.0, 60.0)
    side_taper: Range = (0.70, 0.95)  # side end diameter / side start diameter
    split_exponent: Range = (2.33, 3.0)  # main step-down after a take-off (Murray-like)
    side_lesion_count_probs: list[float] = [0.60, 0.30, 0.10]  # per side branch: 0, 1, 2
    bifurcation_lesion_prob: float = Field(0.30, ge=0, le=1)  # main lesion centred on a take-off
    bifurcation_offset_mm: float = Field(3.0, ge=0)  # its centre within ± this of the take-off
    ostial_lesion_prob: float = Field(0.15, ge=0, le=1)  # per side branch
    ostial_center_mm: Range = (0.5, 3.0)  # throat distance from the side branch's start

    @field_validator("side_diameter_ratio", "side_length_mm", "side_taper", "split_exponent",
                     "ostial_center_mm")  # fmt: skip
    @classmethod
    def _ordered_tree(cls, v: Range) -> Range:
        if v[0] > v[1]:
            raise ValueError(f"range low > high: {v}")
        return v

    @model_validator(mode="after")
    def _check_tree(self) -> "TreeSamplingConfig":
        _probs("side_count_probs", self.side_count_probs, MAX_SIDE + 1)
        _probs("side_lesion_count_probs", self.side_lesion_count_probs, 3)
        if len(self.lesion_count_probs) > 4:
            raise ValueError("at most 3 lesions on the main vessel")
        return self


class TreeSynthConfig(StrictModel):
    """Full synthetic-tree dataset config (``configs/synth_tree_v1.yaml``)."""

    name: str = "tree_v1"
    out_dir: str = "data/synth/tree_v1"
    n_points: int = Field(256, ge=16)  # per branch
    shard_size: int = Field(5_000, gt=0)
    workers: int | None = Field(None, gt=0)
    sampling: TreeSamplingConfig = TreeSamplingConfig()
    physics: PhysicsParams = PhysicsParams()
    splits: dict[str, SplitConfig]

    @model_validator(mode="after")
    def _check_overrides(self) -> "TreeSynthConfig":
        for split in self.splits:
            self.sampling_for(split)
        return self

    def sampling_for(self, split: str) -> TreeSamplingConfig:
        """Base sampling config with the split's overrides applied (validated)."""
        overrides = self.splits[split].overrides
        return TreeSamplingConfig.model_validate({**self.sampling.model_dump(), **overrides})

    def shard_arrays(self, split: str, n: int, seed: tuple[int, int]) -> dict[str, np.ndarray]:
        """Sample and label one shard of ``split`` (see :func:`generate_tree_shard`)."""
        return generate_tree_shard(self.sampling_for(split), self.physics, n, self.n_points, seed)

    def manifest_extra(self) -> dict[str, Any]:
        """Dataset-specific manifest fields."""
        return {
            "kind": "tree",
            "max_branches": MAX_BRANCHES,
            "max_lesions": MAX_TREE_LESIONS,
            "shapes": list(SHAPES),
            "lesion_kinds": list(LESION_KINDS),
            "units": TREE_ARRAY_UNITS,
        }


@dataclass(frozen=True)
class TreeLesion:
    """A sampled lesion on one branch (positions in m from the branch start)."""

    branch: int
    spec: LesionSpec
    kind: str


@dataclass(frozen=True, eq=False)
class SyntheticTree:
    """A sampled tree before labelling. Pressures in Pa, v_hyp in m/s."""

    tree: VesselTree
    pa: float
    pv: float
    v_hyp: float
    lesions: tuple[TreeLesion, ...]
    noise_sigma_frac: float


def _take_offs(rng: np.random.Generator, length: float, k: int, cfg: TreeSamplingConfig):
    """``k`` sorted take-off positions (m), ≥ margin from the ends and ≥ spacing apart; fewer
    if they cannot fit."""
    margin, gap = mm_to_m(cfg.side_margin_mm), mm_to_m(cfg.side_spacing_mm)
    while k > 0:
        room = length - 2.0 * margin - gap * (k - 1)
        if room > 0:
            offsets = np.sort(rng.uniform(0.0, room, size=k))
            return (margin + offsets + gap * np.arange(k)).tolist()
        k -= 1
    return []


def _lesion(rng, s, start, length, cfg, kind, branch):
    """One lesion's radius factor and spec (start and length in m)."""
    shape = str(rng.choice(list(cfg.shape_probs), p=list(cfg.shape_probs.values())))
    lo, hi = cfg.lesion_ds_pct
    ds_pct = lo + (hi - lo) * rng.beta(*cfg.lesion_ds_beta)
    factor = 1.0 - ds_pct / 100.0 * lesion_bump(s, start, length, shape, rng, cfg)
    return factor, TreeLesion(branch, LesionSpec(float(start), float(length), float(ds_pct),
                                                 shape), kind)  # fmt: skip


def _branch_lesions(rng, s, k, fixed, cfg, branch):
    """Radius factor and specs: ``fixed`` (start, length, kind) lesions plus ``k`` regular
    ones placed by the M2 rule, kept only where they do not overlap a fixed one."""
    length = float(s[-1])
    factor, specs = np.ones(s.size), []
    for start, ln, kind in fixed:
        f, spec = _lesion(rng, s, start, ln, cfg, kind, branch)
        factor *= f
        specs.append(spec)
    gap = mm_to_m(cfg.lesion_gap_mm)
    while k > 0:
        try:
            placed = _place_lesions(rng, k, length, cfg)
        except ValueError:  # the branch is too short for k lesions
            k -= 1
            continue
        clear = [
            (st, ln) for st, ln in placed
            if all(st + ln + gap <= fs or st >= fs + fl + gap for fs, fl, _ in fixed)
        ]  # fmt: skip
        for st, ln in clear:
            f, spec = _lesion(rng, s, st, ln, cfg, "regular", branch)
            factor *= f
            specs.append(spec)
        break
    return factor, specs


def _noisy(rng, r_ref, factor, sigma, ds, cfg):
    r_clean = r_ref * factor
    noise = gaussian_filter1d(rng.standard_normal(r_ref.size), mm_to_m(cfg.noise_corr_mm) / ds)
    noise /= noise.std()
    return np.maximum(r_clean + sigma * r_ref * noise, cfg.min_radius_frac * r_clean)


def sample_tree(
    rng: np.random.Generator, cfg: TreeSamplingConfig, n_points: int = 256
) -> SyntheticTree:
    """Draw one synthetic tree (geometry, pressures, v_hyp). Not yet labelled."""
    length = mm_to_m(rng.uniform(*cfg.length_mm))
    s = np.linspace(0.0, length, n_points)
    ds = s[1]
    n_side = int(rng.choice(len(cfg.side_count_probs), p=cfg.side_count_probs))
    positions = _take_offs(rng, length, n_side, cfg)
    attach = sorted({int(np.clip(round(p / ds), 1, n_points - 2)) for p in positions})

    # main reference: linear taper within segments, Murray-like step-down after take-offs
    r_ref = np.empty(n_points)
    d = mm_to_m(rng.uniform(*cfg.d0_mm))
    bounds = [0, *attach, n_points - 1]
    side_d = []
    for i0, i1 in zip(bounds[:-1], bounds[1:], strict=True):
        f = rng.uniform(*cfg.distal_taper)
        lo = i0 if i0 == 0 else i0 + 1  # the step-down starts one sample after the take-off
        seg = slice(lo, i1 + 1)
        r_ref[seg] = 0.5 * d * (1.0 + (f - 1.0) * (s[seg] - s[lo]) / (s[i1] - s[lo]))
        if i1 == n_points - 1:
            break
        d_before = 2.0 * r_ref[i1]
        d_s = d_before * rng.uniform(*cfg.side_diameter_ratio)
        e = rng.uniform(*cfg.split_exponent)
        d = (d_before**e - d_s**e) ** (1.0 / e)
        side_d.append(d_s)

    sigma = rng.uniform(*cfg.noise_sigma_frac)
    lesions: list[TreeLesion] = []

    # main vessel lesions (one may be centred on a take-off), diffuse disease as in M2
    k_main = int(rng.choice(len(cfg.lesion_count_probs), p=cfg.lesion_count_probs))
    fixed = []
    if attach and k_main > 0 and rng.random() < cfg.bifurcation_lesion_prob:
        t = s[attach[int(rng.integers(len(attach)))]]
        centre = t + mm_to_m(rng.uniform(-cfg.bifurcation_offset_mm, cfg.bifurcation_offset_mm))
        ln = mm_to_m(rng.uniform(*cfg.lesion_length_mm))
        fixed.append((centre - 0.5 * ln, ln, "bifurcation"))
        k_main -= 1
    factor, specs = _branch_lesions(rng, s, k_main, fixed, cfg, 0)
    lesions += specs
    if rng.random() < cfg.diffuse_prob:
        d_len = length * rng.uniform(*cfg.diffuse_length_frac)
        d_start = rng.uniform(0.0, length - d_len)
        x = np.clip((s - d_start) / d_len, 0.0, 1.0)
        bump = _plateau(x, 0.5) * _modulation(rng, x, cfg.irregular_amplitude)
        bump = np.where((s >= d_start) & (s <= d_start + d_len), bump, 0.0)
        factor *= 1.0 - rng.uniform(*cfg.diffuse_ds_pct) / 100.0 * bump / max(bump.max(), 1e-12)
    branches = [VesselProfile(s, _noisy(rng, r_ref, factor, sigma, ds, cfg), r_ref)]

    # side branches
    for d_s in side_d:
        b = len(branches)
        ln_b = mm_to_m(rng.uniform(*cfg.side_length_mm))
        sb = np.linspace(0.0, ln_b, n_points)
        rb_ref = 0.5 * d_s * (1.0 + (rng.uniform(*cfg.side_taper) - 1.0) * sb / ln_b)
        k = int(rng.choice(len(cfg.side_lesion_count_probs), p=cfg.side_lesion_count_probs))
        fixed = []
        if rng.random() < cfg.ostial_lesion_prob:
            ln = mm_to_m(rng.uniform(*cfg.lesion_length_mm))
            centre = mm_to_m(rng.uniform(*cfg.ostial_center_mm))
            fixed.append((centre - 0.5 * ln, ln, "ostial"))
            k = max(k - 1, 0)
        fb, specs = _branch_lesions(rng, sb, k, fixed, cfg, b)
        lesions += specs
        branches.append(VesselProfile(sb, _noisy(rng, rb_ref, fb, sigma, sb[1], cfg), rb_ref))

    n_b = len(branches)
    tree = VesselTree(
        tuple(branches),
        (-1,) + (0,) * (n_b - 1),
        (0, *attach),
        ("main", *(f"side {k}" for k in range(1, n_b))),
    )
    return SyntheticTree(
        tree=tree,
        pa=float(mmhg_to_pa(rng.uniform(*cfg.pa_mmhg))),
        pv=float(mmhg_to_pa(rng.uniform(*cfg.pv_mmhg))),
        v_hyp=float(rng.uniform(*cfg.v_hyp)),
        lesions=tuple(lesions),
        noise_sigma_frac=float(sigma),
    )


def tree_from_arrays(arrays: dict[str, np.ndarray], i: int) -> VesselTree:
    """Rebuild tree ``i`` of a loaded split (float64 radii, as labelled)."""
    nb = int(arrays["branch_count"][i])
    n_points = arrays["r"].shape[-1]
    branches = tuple(
        VesselProfile(
            np.linspace(0.0, float(arrays["length"][i, k]), n_points),
            arrays["r"][i, k].astype(np.float64),
            arrays["r_ref"][i, k].astype(np.float64),
        )
        for k in range(nb)
    )
    names = ("main", *(f"side {k}" for k in range(1, nb)))
    return VesselTree(
        branches,
        tuple(int(p) for p in arrays["parent"][i, :nb]),
        tuple(int(a) for a in arrays["attach"][i, :nb]),
        names,
    )


def generate_tree_shard(
    sampling: TreeSamplingConfig,
    physics: PhysicsParams,
    n: int,
    n_points: int,
    seed,
) -> dict[str, np.ndarray]:
    """Sample and label ``n`` trees; returns the arrays described in ``TREE_ARRAY_UNITS``."""
    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    samples = [sample_tree(rng, sampling, n_points) for _ in range(n)]
    shape = (n, MAX_BRANCHES, n_points)
    out: dict[str, np.ndarray] = {
        "r": np.zeros(shape, np.float32),
        "r_ref": np.zeros(shape, np.float32),
        "ffr": np.zeros(shape, np.float32),
        "length": np.zeros((n, MAX_BRANCHES)),
        "parent": np.full((n, MAX_BRANCHES), ABSENT, np.int8),
        "attach": np.zeros((n, MAX_BRANCHES), np.int16),
        "branch_count": np.zeros(n, np.int8),
    }
    for i, t in enumerate(samples):
        nb = t.tree.n_branches
        out["branch_count"][i] = nb
        out["parent"][i, :nb] = t.tree.parent
        out["attach"][i, :nb] = t.tree.attach
        for k, b in enumerate(t.tree.branches):
            out["r"][i, k] = b.r
            out["r_ref"][i, k] = b.r_ref
            out["length"][i, k] = b.length
    # label exactly what is stored (float32 radii), as in M2
    trees = [tree_from_arrays(out, i) for i in range(n)]
    pa = np.array([t.pa for t in samples])
    pv = np.array([t.pv for t in samples])
    v_hyp = np.array([t.v_hyp for t in samples])
    results = solve_tree_batch(trees, physics, pa=pa, pv=pv, v_hyp=v_hyp)
    main = solve_ffr_batch(
        out["r"][:, 0].astype(np.float64), out["length"][:, 0] / (n_points - 1),
        out["r_ref"][:, 0].astype(np.float64), physics, pa=pa, pv=pv, v_hyp=v_hyp,
    )  # fmt: skip

    per_branch = ("q_outlet", "q_inlet", "q_outlet_healthy", "r_outlet")
    for key in per_branch:
        out[key] = np.zeros((n, MAX_BRANCHES))
    out["r_outlet_clamped"] = np.zeros((n, MAX_BRANCHES), bool)
    lesion = {
        "lesion_branch": np.full((n, MAX_TREE_LESIONS), -1, np.int8),
        "lesion_start": np.full((n, MAX_TREE_LESIONS), np.nan, np.float32),
        "lesion_length": np.full((n, MAX_TREE_LESIONS), np.nan, np.float32),
        "lesion_ds_pct": np.full((n, MAX_TREE_LESIONS), np.nan, np.float32),
        "lesion_shape": np.full((n, MAX_TREE_LESIONS), -1, np.int8),
        "lesion_kind": np.full((n, MAX_TREE_LESIONS), -1, np.int8),
    }
    for i, (t, res) in enumerate(zip(samples, results, strict=True)):
        nb = t.tree.n_branches
        for k in range(nb):
            out["ffr"][i, k] = res.ffr[k]
        for key in per_branch:
            out[key][i, :nb] = getattr(res, key)
        out["r_outlet_clamped"][i, :nb] = res.r_outlet_clamped
        for j, les in enumerate(t.lesions[:MAX_TREE_LESIONS]):
            lesion["lesion_branch"][i, j] = les.branch
            lesion["lesion_start"][i, j] = les.spec.start
            lesion["lesion_length"][i, j] = les.spec.length
            lesion["lesion_ds_pct"][i, j] = les.spec.ds_pct
            lesion["lesion_shape"][i, j] = SHAPES.index(les.spec.shape)
            lesion["lesion_kind"][i, j] = LESION_KINDS.index(les.kind)
    out.update(lesion)
    out.update(
        {
            "ffr_main_single": main.ffr.astype(np.float32),
            "pa": pa,
            "pv": pv,
            "v_hyp": v_hyp,
            "q_ref": np.array([res.q_ref for res in results]),
            "noise_sigma_frac": np.array([t.noise_sigma_frac for t in samples]),
        }
    )
    return out
