"""Surrogate input features: the single pre-processing path for training, ONNX and the app.

Raw inputs are SI (m, Pa, Pa·s/m³). Features are dimensionless or in mm / mmHg (an ML
boundary) and are standardised with training-set statistics saved to JSON.

Per-point channels (shape (B, 4, N)):
    r / r_ref(0), log(r / r_ref(0)), s / L, d(A/A_ref)/ds with s in mm (1/mm).
Global features (shape (B, 5)):
    L (mm), r_ref(0) (mm), Pa (mmHg), Pv (mmHg), ln R_micro (R_micro in Pa·s/m³).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vstent.physics.geometry import VesselProfile
from vstent.units import m_to_mm, pa_to_mmhg

N_POINTS = 256
FEATURE_VERSION = 1
NORM_STATS = "norm_stats.json"
POINT_CHANNELS = ("r_over_r0", "log_r_over_r0", "s_over_L", "d_area_ratio_ds_per_mm")
GLOBAL_FEATURES = ("length_mm", "r_ref0_mm", "pa_mmhg", "pv_mmhg", "log_r_micro")


@dataclass(frozen=True, eq=False)
class RawInputs:
    """Batch of vessels in SI units.

    Attributes:
        r: lumen radius (m), shape (B, N).
        r_ref: reference radius (m), shape (B, N).
        length: vessel length (m), shape (B,); s = linspace(0, length, N).
        pa: aortic pressure (Pa), shape (B,).
        pv: venous pressure (Pa), shape (B,).
        r_micro: microvascular resistance (Pa·s/m³), shape (B,).
    """

    r: np.ndarray
    r_ref: np.ndarray
    length: np.ndarray
    pa: np.ndarray
    pv: np.ndarray
    r_micro: np.ndarray

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "RawInputs":
        """From a dict with the keys of a synthetic-data split (see ``data.synth``)."""
        return cls(*(np.asarray(arrays[k], dtype=np.float64) for k in cls.__dataclass_fields__))

    @classmethod
    def from_profile(
        cls, profile: VesselProfile, pa: float, pv: float, r_micro: float, n: int = N_POINTS
    ) -> "RawInputs":
        """One vessel, resampled to ``n`` points. Pressures in Pa, R_micro in Pa·s/m³."""
        p = profile.resample(n) if profile.n != n else profile
        return cls(
            r=p.r[None, :].astype(np.float64),
            r_ref=p.reference_radius()[None, :].astype(np.float64),
            length=np.array([p.length]),
            pa=np.array([pa], dtype=np.float64),
            pv=np.array([pv], dtype=np.float64),
            r_micro=np.array([r_micro], dtype=np.float64),
        )

    def __len__(self) -> int:
        return np.atleast_2d(self.r).shape[0]

    def subset(self, idx) -> "RawInputs":
        """Vessels selected by an index, slice or boolean mask (always 2-D / 1-D arrays)."""
        idx = np.atleast_1d(np.arange(len(self))[idx])
        return type(self)(
            *(
                (np.atleast_2d if k in ("r", "r_ref") else np.atleast_1d)(getattr(self, k))[idx]
                for k in self.__dataclass_fields__
            )
        )


def point_features(raw: RawInputs) -> np.ndarray:
    """Unstandardised per-point channels, float64 (B, 4, N)."""
    r, r_ref = np.atleast_2d(raw.r), np.atleast_2d(raw.r_ref)
    n = r.shape[1]
    ratio = r / r_ref[:, :1]
    s_over_l = np.broadcast_to(np.linspace(0.0, 1.0, n), r.shape)
    ds_mm = m_to_mm(np.asarray(raw.length, dtype=np.float64)) / (n - 1)
    d_area = np.gradient((r / r_ref) ** 2, axis=1) / ds_mm[:, None]
    return np.stack([ratio, np.log(ratio), s_over_l, d_area], axis=1)


def global_features(raw: RawInputs) -> np.ndarray:
    """Unstandardised global features, float64 (B, 5)."""
    return np.stack(
        [
            m_to_mm(np.asarray(raw.length, dtype=np.float64)),
            m_to_mm(np.atleast_2d(raw.r_ref)[:, 0]),
            pa_to_mmhg(np.asarray(raw.pa, dtype=np.float64)),
            pa_to_mmhg(np.asarray(raw.pv, dtype=np.float64)),
            np.log(np.asarray(raw.r_micro, dtype=np.float64)),
        ],
        axis=1,
    )


@dataclass(frozen=True)
class NormStats:
    """Training-set mean / std of every feature (lists so they serialise to JSON)."""

    point_mean: list[float]
    point_std: list[float]
    global_mean: list[float]
    global_std: list[float]
    n_points: int = N_POINTS
    feature_version: int = FEATURE_VERSION
    point_channels: tuple[str, ...] = POINT_CHANNELS
    global_features: tuple[str, ...] = GLOBAL_FEATURES

    @classmethod
    def fit(cls, raw: RawInputs) -> "NormStats":
        """Statistics over all samples (and, for point channels, all positions)."""
        xp, xg = point_features(raw), global_features(raw)
        point_std = xp.std(axis=(0, 2))
        global_std = xg.std(axis=0)
        return cls(
            point_mean=xp.mean(axis=(0, 2)).tolist(),
            point_std=np.where(point_std > 0, point_std, 1.0).tolist(),
            global_mean=xg.mean(axis=0).tolist(),
            global_std=np.where(global_std > 0, global_std, 1.0).tolist(),
            n_points=xp.shape[2],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["point_channels"] = tuple(data["point_channels"])
        data["global_features"] = tuple(data["global_features"])
        stats = cls(**data)
        if stats.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"{path}: feature version {stats.feature_version} != {FEATURE_VERSION}"
            )
        return stats


def prepare_inputs(raw: RawInputs, stats: NormStats) -> tuple[np.ndarray, np.ndarray]:
    """Standardised model inputs: float32 (B, 4, N) per-point and (B, 5) global features."""
    n = np.atleast_2d(raw.r).shape[1]
    if n != stats.n_points:
        raise ValueError(f"profiles have {n} points; the model expects {stats.n_points}")
    xp = point_features(raw)
    xg = global_features(raw)
    xp = (xp - np.array(stats.point_mean)[None, :, None]) / np.array(stats.point_std)[None, :, None]
    xg = (xg - np.array(stats.global_mean)[None, :]) / np.array(stats.global_std)[None, :]
    return xp.astype(np.float32), xg.astype(np.float32)
