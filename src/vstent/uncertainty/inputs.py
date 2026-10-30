"""Input-uncertainty propagation through the reduced-order solver (P1). Torch-free.

Monte Carlo over the assumptions that matter most, for one case. Each draw perturbs:

- ``pixel_size``: all lengths and diameters scale by k ~ U(1 − rel, 1 + rel); only when the
  pixel size was assumed (image cases without calibration);
- ``qca_diameter``: diameter error e(s) (px), zero-mean Gaussian with the SD and correlation
  length measured on rendered tubes (``reports/qca_synthetic.json``); image cases only. The
  reference, a robust fit over the whole path, moves by the path mean of e;
- ``v_hyp``: hyperemic reference velocity ± half its synthetic sampling range (m/s), which
  recalibrates R_micro;
- ``pa_pv``: Pa and Pv ± half their synthetic sampling ranges (mmHg; Pv ≥ 0).

Distributions are uniform except the QCA error. Post-PCI draws reuse the same draws (common
random numbers) and the draw's own pre-PCI R_micro; a stent keeps its physical length and
nominal diameter, and its start moves with the pixel scale (it is placed where it was drawn).
The interval covers only these assumptions, not the reduced-order model's own error.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import Field
from scipy.ndimage import convolve1d

from vstent.config import StrictModel
from vstent.physics.geometry import VesselProfile
from vstent.physics.solver import PhysicsParams, solve_ffr_batch
from vstent.planning.stent import DEFAULT_EDGE, Stent, stent_weight
from vstent.units import mm_to_m, mmhg_to_pa

SOURCES = ("pixel_size", "qca_diameter", "v_hyp", "pa_pv")
SOURCE_LABELS = {
    "pixel_size": "pixel size",
    "qca_diameter": "QCA diameter error",
    "v_hyp": "hyperemic velocity",
    "pa_pv": "Pa / Pv",
}
MIN_V_HYP = 0.05  # m/s, floor for sampled velocities
MIN_PA_PV_GAP_MMHG = 1.0


class InputUncertaintyConfig(StrictModel):
    """Monte Carlo settings (``configs/uncertainty.yaml``, ``inputs``). Units as named."""

    n_samples: int = Field(500, ge=20)
    seed: int = 0
    pixel_size_rel: float = Field(0.10, ge=0, lt=1)
    v_hyp_halfwidth: float = Field(0.10, ge=0)  # m/s
    pa_halfwidth_mmhg: float = Field(20.0, ge=0)
    pv_halfwidth_mmhg: float = Field(3.0, ge=0)
    qca_error_report: str = "reports/qca_synthetic.json"


@dataclass(frozen=True)
class QCAErrorModel:
    """Per-point QCA diameter error: SD and half-correlation lag along the vessel (px)."""

    sd_px: float
    corr_half_px: float

    @classmethod
    def from_report(cls, path: str | Path) -> "QCAErrorModel | None":
        """Read ``error_model`` from ``reports/qca_synthetic.json`` (``None`` if absent)."""
        import json

        path = Path(path)
        if not path.exists():
            return None
        em = json.loads(path.read_text(encoding="utf-8")).get("error_model")
        return None if em is None else cls(float(em["sd_px"]), float(em["corr_half_px"]))


@dataclass(frozen=True)
class Measurement:
    """How a case's geometry was obtained (decides which assumptions apply).

    ``mm_per_px`` is the pixel size used for an image-derived profile (mm/px), ``None`` for a
    synthetic geometry that is known exactly.
    """

    mm_per_px: float | None = None
    pixel_size_assumed: bool = False


def correlated_noise(
    rng: np.random.Generator, n: int, m: int, corr_half_points: float
) -> np.ndarray:
    """(n, m) zero-mean, unit-variance Gaussian noise whose autocorrelation falls to 0.5 after
    ``corr_half_points`` points (Gaussian kernel; white noise below 0.3 points)."""
    white = rng.standard_normal((n, m))
    sigma = corr_half_points / (2.0 * np.sqrt(np.log(2.0)))  # ACF exp(−τ²/4σ²) = 0.5
    if sigma < 0.3:
        return white
    half = int(np.ceil(4 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= np.sqrt(np.sum(kernel**2))  # keeps unit variance
    return convolve1d(white, kernel, axis=1, mode="reflect")


def applicable_sources(measurement: Measurement, qca_error: QCAErrorModel | None) -> list[str]:
    """The assumptions that apply to a case measured as ``measurement``."""
    out = []
    if measurement.mm_per_px is not None and measurement.pixel_size_assumed:
        out.append("pixel_size")
    if measurement.mm_per_px is not None and qca_error is not None:
        out.append("qca_diameter")
    return [*out, "v_hyp", "pa_pv"]


class InputUncertainty:
    """Monte Carlo of the solver's distal FFR over the input assumptions of one case.

    Args:
        profile: the measured (pre-PCI) profile, SI units.
        params: physics constants; ``params.v_hyp`` is the velocity used for the case.
        pa, pv: pressures used for the case (Pa).
        measurement: image-derived or synthetic (see :class:`Measurement`).
        config: Monte Carlo settings.
        qca_error: QCA error model (image cases); ``None`` skips that source.
        sources: subset of :data:`SOURCES` to vary (default: all that apply).
    """

    def __init__(
        self,
        profile: VesselProfile,
        params: PhysicsParams,
        pa: float,
        pv: float,
        *,
        measurement: Measurement,
        config: InputUncertaintyConfig | None = None,
        qca_error: QCAErrorModel | None = None,
        sources: Collection[str] | None = None,
    ) -> None:
        cfg = config or InputUncertaintyConfig()
        allowed = applicable_sources(measurement, qca_error)
        self.sources = [s for s in allowed if sources is None or s in sources]
        self.profile, self.params = profile, params
        rng = np.random.default_rng(cfg.seed)
        n, m = cfg.n_samples, profile.n

        # All draws are made, in a fixed order, whichever sources are varied: switching one
        # source off does not change the random numbers of the others.
        u_k, u_v, u_pa, u_pv = rng.uniform(-1.0, 1.0, size=(4, n))
        e_unit = np.zeros((n, m))
        if measurement.mm_per_px is not None and qca_error is not None:
            spacing_px = profile.ds / mm_to_m(measurement.mm_per_px)
            e_unit = correlated_noise(rng, n, m, qca_error.corr_half_px / spacing_px)

        k = 1.0 + (cfg.pixel_size_rel * u_k if "pixel_size" in self.sources else 0.0)
        r, r_ref = profile.r[None, :], profile.reference_radius()[None, :]
        if "qca_diameter" in self.sources:
            e = e_unit * qca_error.sd_px * mm_to_m(measurement.mm_per_px)  # diameter error (m)
            floor = 0.1 * r
            r = np.maximum(r + e / 2.0, floor)
            r_ref = np.maximum(r_ref + e.mean(axis=1, keepdims=True) / 2.0, floor)
        k = np.broadcast_to(np.asarray(k, dtype=float), (n,))
        self.k = k
        self.r = np.broadcast_to(r * k[:, None], (n, m)).copy()
        self.r_ref = np.broadcast_to(r_ref * k[:, None], (n, m)).copy()
        self.s = k[:, None] * profile.s[None, :]
        self.ds = k * profile.ds
        self.v_hyp = params.v_hyp + (cfg.v_hyp_halfwidth * u_v if "v_hyp" in self.sources else 0.0)
        self.v_hyp = np.maximum(np.broadcast_to(self.v_hyp, (n,)), MIN_V_HYP)
        pa_mmhg = pa / mmhg_to_pa(1.0)
        pv_mmhg = pv / mmhg_to_pa(1.0)
        if "pa_pv" in self.sources:
            pa_mmhg = pa_mmhg + cfg.pa_halfwidth_mmhg * u_pa
            pv_mmhg = np.clip(
                pv_mmhg + cfg.pv_halfwidth_mmhg * u_pv, 0.0, pa_mmhg - MIN_PA_PV_GAP_MMHG
            )
        self.pa = mmhg_to_pa(np.broadcast_to(pa_mmhg, (n,)).astype(float))
        self.pv = mmhg_to_pa(np.broadcast_to(pv_mmhg, (n,)).astype(float))

        pre = solve_ffr_batch(
            self.r, self.ds, self.r_ref, params, pa=self.pa, pv=self.pv, v_hyp=self.v_hyp
        )
        self.r_micro = pre.r_micro
        self.pre_samples = pre.ffr_distal

    @property
    def n_samples(self) -> int:
        return self.pre_samples.size

    def post_samples(self, stents: Sequence[Stent], edge: float = DEFAULT_EDGE) -> np.ndarray:
        """Post-PCI distal FFR of every draw with ``stents`` implanted (pre-PCI R_micro).

        Stent positions are distances from the inlet (m), as in ``apply_stents``; ``edge`` is
        the half-cosine blending length (m) inside each stent end, as in the planner.
        """
        r = self.r
        out = r.copy()
        for st in stents:
            s0 = self.k * st.s_start
            w = stent_weight(self.s, s0, s0 + st.length, edge)
            out = np.maximum(out, r + w * np.maximum(0.0, st.radius - r))
        res = solve_ffr_batch(
            out, self.ds, self.r_ref, self.params, pa=self.pa, pv=self.pv, r_micro=self.r_micro
        )
        return res.ffr_distal


def percentile_interval(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Central ``level`` interval of Monte Carlo samples."""
    lo, hi = np.percentile(samples, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)
