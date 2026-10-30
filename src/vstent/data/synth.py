"""Synthetic vessel factory (docs/PLAN.md, M2).

Samples single-vessel radius profiles (tapered reference, 0-4 focal lesions of four shapes,
optional diffuse disease, smoothed measurement noise) and labels them with the reduced-order
solver. Labels are computed from exactly the stored float32 noisy profile.

Units: SI internally (m, Pa, m³/s, m/s). Config fields ending in ``_mm``, ``_mmhg`` or
``_pct`` are converted on use. Stored arrays and their units are listed in ``ARRAY_UNITS``.
"""

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, field_validator, model_validator
from scipy.ndimage import gaussian_filter1d

from vstent.config import StrictModel, save_run_metadata
from vstent.physics.geometry import VesselProfile
from vstent.physics.solver import PhysicsParams, solve_ffr_batch
from vstent.sysinfo import cpu_name
from vstent.units import mm_to_m, mmhg_to_pa

SHAPES = ("cosine", "skewed_cosine", "plateau", "irregular")
"""Lesion shape names; a shape's index is its code in the ``lesion_shape`` array."""
MAX_LESIONS = 4
MANIFEST = "manifest.json"

ARRAY_UNITS = {
    "r": "m, float32 (n, N): noisy lumen radius the model sees",
    "r_ref": "m, float32 (n, N): true reference radius",
    "ffr": "-, float32 (n, N): FFR(s) from the solver on the stored r",
    "dp": "Pa, float32 (n, N): pressure loss ΔP(s)",
    "length": "m, float64 (n,): vessel length; s = linspace(0, length, N)",
    "pa": "Pa, float64 (n,)",
    "pv": "Pa, float64 (n,)",
    "v_hyp": "m/s, float64 (n,)",
    "r_micro": "Pa·s/m³, float64 (n,)",
    "r_micro_clamped": "bool (n,)",
    "q": "m³/s, float64 (n,): hyperemic flow",
    "q_ref": "m³/s, float64 (n,): reference flow",
    "noise_sigma_frac": "-, float64 (n,): noise σ / r_ref",
    "lesion_count": "int8 (n,)",
    "lesion_start": f"m, float32 (n, {MAX_LESIONS}), NaN-padded",
    "lesion_length": f"m, float32 (n, {MAX_LESIONS}), NaN-padded",
    "lesion_ds_pct": f"%, float32 (n, {MAX_LESIONS}): nominal throat DS, NaN-padded",
    "lesion_shape": f"int8 (n, {MAX_LESIONS}): index into SHAPES, -1 padded",
    "diffuse_start": "m, float32 (n,), NaN if none",
    "diffuse_length": "m, float32 (n,), NaN if none",
    "diffuse_ds_pct": "%, float32 (n,), NaN if none",
}

Range = tuple[float, float]


class SamplingConfig(StrictModel):
    """Sampling distributions (uniform over [low, high] unless noted). May be tuned."""

    length_mm: Range = (30.0, 80.0)
    d0_mm: Range = (2.25, 4.5)
    distal_taper: Range = (0.6, 0.95)
    lesion_count_probs: list[float] = [0.10, 0.55, 0.25, 0.10]
    lesion_ds_pct: Range = (20.0, 90.0)
    lesion_ds_beta: tuple[float, float] = (3.0, 3.5)
    lesion_length_mm: Range = (3.0, 25.0)
    lesion_margin_mm: float = Field(5.0, ge=0)
    lesion_gap_mm: float = Field(2.0, ge=0)
    shape_probs: dict[str, float] = {s: 0.25 for s in SHAPES}
    skew_peak: Range = (0.2, 0.8)
    plateau_fraction: Range = (0.3, 0.7)
    irregular_amplitude: float = Field(0.3, ge=0, lt=1)
    diffuse_prob: float = Field(0.2, ge=0, le=1)
    diffuse_ds_pct: Range = (0.0, 20.0)
    diffuse_length_frac: Range = (0.3, 0.8)
    noise_sigma_frac: Range = (0.0, 0.03)
    noise_corr_mm: float = Field(1.0, gt=0)
    min_radius_frac: float = Field(0.5, gt=0, le=1)
    pa_mmhg: Range = (70.0, 110.0)
    pv_mmhg: Range = (2.0, 8.0)
    v_hyp: Range = (0.25, 0.45)

    @field_validator(
        "length_mm",
        "d0_mm",
        "distal_taper",
        "lesion_ds_pct",
        "lesion_length_mm",
        "skew_peak",
        "plateau_fraction",
        "diffuse_ds_pct",
        "diffuse_length_frac",
        "noise_sigma_frac",
        "pa_mmhg",
        "pv_mmhg",
        "v_hyp",
    )
    @classmethod
    def _ordered(cls, v: Range) -> Range:
        if v[0] > v[1]:
            raise ValueError(f"range low > high: {v}")
        return v

    @model_validator(mode="after")
    def _check(self) -> "SamplingConfig":
        if len(self.lesion_count_probs) > MAX_LESIONS + 1:
            raise ValueError(f"at most {MAX_LESIONS} lesions supported")
        for name, probs in (
            ("lesion_count_probs", self.lesion_count_probs),
            ("shape_probs", list(self.shape_probs.values())),
        ):
            if min(probs) < 0 or not math.isclose(sum(probs), 1.0, abs_tol=1e-9):
                raise ValueError(f"{name} must be non-negative and sum to 1")
        if unknown := set(self.shape_probs) - set(SHAPES):
            raise ValueError(f"unknown lesion shapes {sorted(unknown)}")
        if self.diffuse_length_frac[1] > 1.0:
            raise ValueError("diffuse_length_frac must be ≤ 1")
        return self


class SplitConfig(StrictModel):
    """One dataset split: size, seed and optional sampling overrides."""

    n: int = Field(gt=0)
    seed: int
    overrides: dict[str, Any] = {}


class SynthConfig(StrictModel):
    """Full synthetic-dataset config (``configs/synth_v1.yaml``)."""

    name: str = "v1"
    out_dir: str = "data/synth/v1"
    n_points: int = Field(256, ge=16)
    shard_size: int = Field(10_000, gt=0)
    workers: int | None = Field(None, gt=0)
    sampling: SamplingConfig = SamplingConfig()
    physics: PhysicsParams = PhysicsParams()
    splits: dict[str, SplitConfig]

    @model_validator(mode="after")
    def _check_overrides(self) -> "SynthConfig":
        for split in self.splits:
            self.sampling_for(split)
        return self

    def sampling_for(self, split: str) -> SamplingConfig:
        """Base sampling config with the split's overrides applied (validated)."""
        overrides = self.splits[split].overrides
        return SamplingConfig.model_validate({**self.sampling.model_dump(), **overrides})

    def shard_arrays(self, split: str, n: int, seed: tuple[int, int]) -> dict[str, np.ndarray]:
        """Sample and label one shard of ``split`` (see :func:`generate_shard`)."""
        return generate_shard(self.sampling_for(split), self.physics, n, self.n_points, seed)

    def manifest_extra(self) -> dict[str, Any]:
        """Dataset-specific manifest fields."""
        return {"max_lesions": MAX_LESIONS, "shapes": list(SHAPES), "units": ARRAY_UNITS}


@dataclass(frozen=True)
class LesionSpec:
    """A sampled lesion: start and length in m, nominal throat DS in %."""

    start: float
    length: float
    ds_pct: float
    shape: str


@dataclass(frozen=True, eq=False)
class SyntheticVessel:
    """A sampled vessel before labelling. Pressures in Pa, v_hyp in m/s."""

    profile: VesselProfile
    pa: float
    pv: float
    v_hyp: float
    lesions: tuple[LesionSpec, ...]
    diffuse: LesionSpec | None
    noise_sigma_frac: float


def _modulation(rng: np.random.Generator, x: np.ndarray, amplitude: float) -> np.ndarray:
    """Smooth random factor 1 + amplitude·m(x), m = 3 random sinusoids scaled to max|m| = 1."""
    freq = rng.uniform(1.0, 4.0, 3)
    phase = rng.uniform(0.0, 2.0 * np.pi, 3)
    coef = rng.uniform(-1.0, 1.0, 3)
    m = (coef[:, None] * np.sin(2.0 * np.pi * freq[:, None] * x[None, :] + phase[:, None])).sum(0)
    peak = np.abs(m).max()
    return 1.0 + amplitude * (m / peak if peak > 0 else m)


def _plateau(x: np.ndarray, fraction: float) -> np.ndarray:
    """Flat top over ``fraction`` of [0, 1] with half-cosine shoulders."""
    w = 0.5 * (1.0 - fraction)
    rise = 0.5 * (1.0 - np.cos(np.pi * x / w))
    fall = 0.5 * (1.0 + np.cos(np.pi * (x - 1.0 + w) / w))
    return np.where(x < w, rise, np.where(x > 1.0 - w, fall, 1.0))


def lesion_bump(
    s: np.ndarray,
    start: float,
    length: float,
    shape: str,
    rng: np.random.Generator,
    sampling: SamplingConfig,
) -> np.ndarray:
    """Normalised narrowing in [0, 1]: peak 1 inside [start, start + length] (m), 0 outside.

    The radius factor of a lesion with throat DS d (%) is ``1 − d/100 · bump``. The bump is
    normalised on the grid, so the sampled throat DS is exactly attained at a grid point.
    """
    x = (s - start) / length
    xc = np.clip(x, 0.0, 1.0)
    if shape == "cosine":
        bump = 0.5 * (1.0 - np.cos(2.0 * np.pi * xc))
    elif shape == "skewed_cosine":
        p = rng.uniform(*sampling.skew_peak)
        rise = 0.5 * (1.0 - np.cos(np.pi * xc / p))
        fall = 0.5 * (1.0 + np.cos(np.pi * (xc - p) / (1.0 - p)))
        bump = np.where(xc < p, rise, fall)
    elif shape == "plateau":
        bump = _plateau(xc, rng.uniform(*sampling.plateau_fraction))
    elif shape == "irregular":
        cosine = 0.5 * (1.0 - np.cos(2.0 * np.pi * xc))
        bump = cosine * _modulation(rng, xc, sampling.irregular_amplitude)
    else:
        raise ValueError(f"unknown lesion shape {shape!r}")
    bump = np.where((x >= 0.0) & (x <= 1.0), bump, 0.0)
    peak = bump.max()
    return bump / peak if peak > 0 else bump


def _place_lesions(
    rng: np.random.Generator, k: int, length: float, sampling: SamplingConfig
) -> list[tuple[float, float]]:
    """Non-overlapping (start, length) pairs in m, ≥ margin from both ends, ≥ gap apart.

    Lengths are sampled uniformly and scaled down together if they cannot fit (only for short
    vessels with many lesions); free space is distributed uniformly at random.
    """
    if k == 0:
        return []
    lengths = mm_to_m(rng.uniform(*sampling.lesion_length_mm, size=k))
    margin, gap = mm_to_m(sampling.lesion_margin_mm), mm_to_m(sampling.lesion_gap_mm)
    room = length - 2.0 * margin - gap * (k - 1)
    if room <= 0:
        raise ValueError(f"vessel of {length * 1e3:.1f} mm cannot hold {k} lesions")
    if lengths.sum() > room:
        lengths *= room / lengths.sum()
    free = max(room - lengths.sum(), 0.0)  # may round to -1e-18 after scaling
    offsets = np.sort(rng.uniform(0.0, free, size=k))
    starts = margin + offsets + np.concatenate(([0.0], np.cumsum(lengths[:-1] + gap)))
    return list(zip(starts.tolist(), lengths.tolist(), strict=True))


def sample_vessel(
    rng: np.random.Generator, sampling: SamplingConfig, n_points: int = 256
) -> SyntheticVessel:
    """Draw one synthetic vessel (geometry, pressures, v_hyp). Not yet labelled."""
    length = mm_to_m(rng.uniform(*sampling.length_mm))
    s = np.linspace(0.0, length, n_points)
    r0 = mm_to_m(rng.uniform(*sampling.d0_mm)) / 2.0
    r_ref = r0 * (1.0 + (rng.uniform(*sampling.distal_taper) - 1.0) * s / length)

    probs = sampling.lesion_count_probs
    k = int(rng.choice(len(probs), p=probs))
    shape_names = list(sampling.shape_probs)
    shape_p = list(sampling.shape_probs.values())
    factor = np.ones(n_points)
    lesions = []
    for start, lesion_len in _place_lesions(rng, k, length, sampling):
        shape = str(rng.choice(shape_names, p=shape_p))
        lo, hi = sampling.lesion_ds_pct
        ds_pct = lo + (hi - lo) * rng.beta(*sampling.lesion_ds_beta)
        factor *= 1.0 - ds_pct / 100.0 * lesion_bump(s, start, lesion_len, shape, rng, sampling)
        lesions.append(LesionSpec(start, lesion_len, float(ds_pct), shape))

    diffuse = None
    if rng.random() < sampling.diffuse_prob:
        d_len = length * rng.uniform(*sampling.diffuse_length_frac)
        d_start = rng.uniform(0.0, length - d_len)
        d_pct = rng.uniform(*sampling.diffuse_ds_pct)
        x = np.clip((s - d_start) / d_len, 0.0, 1.0)
        bump = _plateau(x, 0.5) * _modulation(rng, x, sampling.irregular_amplitude)
        bump = np.where((s >= d_start) & (s <= d_start + d_len), bump, 0.0)
        bump /= max(bump.max(), 1e-12)
        factor *= 1.0 - d_pct / 100.0 * bump
        diffuse = LesionSpec(float(d_start), float(d_len), float(d_pct), "diffuse")

    r_clean = r_ref * factor
    sigma = rng.uniform(*sampling.noise_sigma_frac)
    noise = gaussian_filter1d(rng.standard_normal(n_points), mm_to_m(sampling.noise_corr_mm) / s[1])
    noise /= noise.std()
    r = np.maximum(r_clean + sigma * r_ref * noise, sampling.min_radius_frac * r_clean)

    return SyntheticVessel(
        profile=VesselProfile(s, r, r_ref),
        pa=float(mmhg_to_pa(rng.uniform(*sampling.pa_mmhg))),
        pv=float(mmhg_to_pa(rng.uniform(*sampling.pv_mmhg))),
        v_hyp=float(rng.uniform(*sampling.v_hyp)),
        lesions=tuple(lesions),
        diffuse=diffuse,
        noise_sigma_frac=float(sigma),
    )


def generate_shard(
    sampling: SamplingConfig,
    physics: PhysicsParams,
    n: int,
    n_points: int,
    seed: Sequence[int],
) -> dict[str, np.ndarray]:
    """Sample and label ``n`` vessels; returns the arrays described in ``ARRAY_UNITS``."""
    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    vessels = [sample_vessel(rng, sampling, n_points) for _ in range(n)]

    r = np.stack([v.profile.r for v in vessels]).astype(np.float32)
    r_ref = np.stack([v.profile.r_ref for v in vessels]).astype(np.float32)
    length = np.array([v.profile.length for v in vessels])
    pa = np.array([v.pa for v in vessels])
    pv = np.array([v.pv for v in vessels])
    v_hyp = np.array([v.v_hyp for v in vessels])
    res = solve_ffr_batch(
        r.astype(np.float64),
        length / (n_points - 1),
        r_ref.astype(np.float64),
        physics,
        pa=pa,
        pv=pv,
        v_hyp=v_hyp,
    )

    lesion_start = np.full((n, MAX_LESIONS), np.nan, dtype=np.float32)
    lesion_length = np.full((n, MAX_LESIONS), np.nan, dtype=np.float32)
    lesion_ds = np.full((n, MAX_LESIONS), np.nan, dtype=np.float32)
    lesion_shape = np.full((n, MAX_LESIONS), -1, dtype=np.int8)
    diffuse = np.full((n, 3), np.nan, dtype=np.float32)
    for i, v in enumerate(vessels):
        for j, les in enumerate(v.lesions):
            lesion_start[i, j] = les.start
            lesion_length[i, j] = les.length
            lesion_ds[i, j] = les.ds_pct
            lesion_shape[i, j] = SHAPES.index(les.shape)
        if v.diffuse is not None:
            diffuse[i] = (v.diffuse.start, v.diffuse.length, v.diffuse.ds_pct)

    return {
        "r": r,
        "r_ref": r_ref,
        "ffr": res.ffr.astype(np.float32),
        "dp": res.dp.astype(np.float32),
        "length": length,
        "pa": pa,
        "pv": pv,
        "v_hyp": v_hyp,
        "r_micro": res.r_micro,
        "r_micro_clamped": res.r_micro_clamped,
        "q": res.q,
        "q_ref": res.q_ref,
        "noise_sigma_frac": np.array([v.noise_sigma_frac for v in vessels]),
        "lesion_count": np.array([len(v.lesions) for v in vessels], dtype=np.int8),
        "lesion_start": lesion_start,
        "lesion_length": lesion_length,
        "lesion_ds_pct": lesion_ds,
        "lesion_shape": lesion_shape,
        "diffuse_start": diffuse[:, 0],
        "diffuse_length": diffuse[:, 1],
        "diffuse_ds_pct": diffuse[:, 2],
    }


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 over array names, dtypes, shapes and bytes (independent of file timestamps)."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        h.update(f"{key}|{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _shard_task(config, split: str, n: int, seed: tuple[int, int], path: Path) -> dict[str, Any]:
    arrays = config.shard_arrays(split, n, seed)
    np.savez(path, **arrays)
    return {"file": path.name, "n": n, "seed": list(seed), "sha256": content_hash(arrays)}


def generate_dataset(
    config,
    *,
    out_dir: str | Path | None = None,
    workers: int | None = None,
    overwrite: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate every split into ``.npz`` shards plus ``manifest.json``, ``config.yaml`` and
    ``run_meta.yaml`` (git commit). Shard seeds are (split seed, shard index), so the output
    does not depend on the number of workers.

    ``config`` is a :class:`SynthConfig` or any config with the same fields and the methods
    ``shard_arrays(split, n, seed)`` and ``manifest_extra()`` (e.g. synthetic trees).
    """
    log = log or (lambda _msg: None)
    out = Path(out_dir if out_dir is not None else config.out_dir)
    existing = sorted(out.glob("*.npz")) + [p for p in [out / MANIFEST] if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{out} already contains a dataset (use overwrite=True)")
    for path in existing:
        path.unlink()
    save_run_metadata(out, config)

    tasks = []
    for split_name, split in config.splits.items():
        for i in range(math.ceil(split.n / config.shard_size)):
            n = min(config.shard_size, split.n - i * config.shard_size)
            path = out / f"{split_name}_{i:04d}.npz"
            tasks.append(((split_name, i), (config, split_name, n, (split.seed, i), path)))

    n_workers = min(workers or config.workers or os.cpu_count() or 1, len(tasks))
    t0 = time.perf_counter()
    results: dict[tuple[str, int], dict[str, Any]] = {}
    if n_workers == 1:
        for key, args in tasks:
            results[key] = _shard_task(*args)
            log(f"{results[key]['file']}: {results[key]['n']} samples")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_shard_task, *args): key for key, args in tasks}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                log(f"{results[futures[fut]]['file']}: {results[futures[fut]]['n']} samples")
    elapsed = time.perf_counter() - t0

    manifest = {
        "name": config.name,
        "n_points": config.n_points,
        **config.manifest_extra(),
        "splits": {
            name: {
                "n": split.n,
                "seed": split.seed,
                "shards": [
                    results[(name, i)] for i in range(math.ceil(split.n / config.shard_size))
                ],
            }
            for name, split in config.splits.items()
        },
        "generation_seconds": elapsed,
        "workers": n_workers,
        "cpu": cpu_name(),
    }
    (out / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    return manifest


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    """Read ``manifest.json`` of a generated dataset."""
    return json.loads((Path(data_dir) / MANIFEST).read_text(encoding="utf-8"))


def load_split(
    data_dir: str | Path, split: str, keys: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """Concatenate the shards of one split (all arrays, or only ``keys``)."""
    data_dir = Path(data_dir)
    shards = load_manifest(data_dir)["splits"][split]["shards"]
    parts: dict[str, list[np.ndarray]] = {}
    for shard in shards:
        with np.load(data_dir / shard["file"]) as npz:
            for key in keys if keys is not None else npz.files:
                parts.setdefault(key, []).append(npz[key])
    return {key: np.concatenate(chunks) for key, chunks in parts.items()}
