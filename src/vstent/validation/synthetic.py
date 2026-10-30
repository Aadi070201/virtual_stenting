"""Synthetic stand-in cases for the validation tool (P4). Not clinical data.

So that ``vstent validate`` can be run and tested end to end before any real data exists, this
module makes a small synthetic "study": patients with one to three vessels. Every vessel is a
synthetic tree (the P2 sampler with the overrides in ``configs/validation_synthetic.yaml``),
laid out in 2D and rendered as an X-ray projection
(:func:`vstent.imaging.synthetic.render_projection`). As on an angiogram, the main vessel
continues a little before the start point and beyond the pressure sensor.

The stand-in reference ("wire FFR") is the tree solver's FFR at the end of the main vessel, on
the true geometry, with the patient's sampled Pa and Pv and the vessel's sampled hyperemic
velocity. The manifest gives the pipeline only what a study records: the image, the start
point and the sensor position (each with a click error), the measured Pa, and for a share of
the patients a calibrated pixel size (with a calibration error). A validation run on these
cases therefore measures the pipeline's image → geometry → FFR error *under its own physics*,
plus the effect of what it assumes (fixed velocity and Pv, an assumed pixel size). It says
nothing about agreement with real wire FFR.

Each case also gets its true lumen mask (``masks/``, the ``mask`` column), standing in for an
analyst-corrected contour: the automatic segmenter, trained on clinical angiograms, misses many
of these rendered vessels (docs/DECISIONS.md, D39), so a run with the masks exercises the rest
of the pipeline and the statistics without that domain gap.

Writes ``<out_dir>/{images/, masks/, cases.csv, reference.csv, truth.csv, config.yaml,
meta.yaml}``.
Units: SI internally, px in the image; CSV columns name their units (mm, mmHg).
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import Field, model_validator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from vstent.config import StrictModel, save_run_metadata
from vstent.data.synth import Range
from vstent.data.synth_tree import TreeSamplingConfig, sample_tree
from vstent.imaging.synthetic import render_projection, render_tubes
from vstent.physics.solver import PhysicsParams, solve_ffr
from vstent.physics.tree import VesselTree, solve_tree
from vstent.units import m_to_mm, pa_to_mmhg
from vstent.validation.manifest import CASE_COLUMNS, REFERENCE_COLUMNS

STEP_PX = 0.25  # spacing of the dense centrelines that are rendered
MAX_RESAMPLES = 50  # trees drawn per vessel before giving up on a layout


class SyntheticStudyConfig(StrictModel):
    """``configs/validation_synthetic.yaml``. Ranges are [low, high], sampled uniformly.

    Sampling choices for a stand-in, not measured values; they may be tuned."""

    name: str = "validation_synthetic"
    out_dir: str = "data/validation_synthetic"
    seed: int = 4001
    n_patients: int = Field(60, gt=0)
    vessels_per_patient_probs: list[float] = [0.5, 0.3, 0.2]  # P(1), P(2), P(3) vessels
    territories: list[str] = ["LAD", "LCX", "RCA"]
    n_points: int = Field(256, ge=16)  # per branch, as in the tree dataset
    sampling: TreeSamplingConfig = TreeSamplingConfig()
    # imaging
    image_size: int = Field(512, ge=128)
    mm_per_px: Range = (0.22, 0.30)  # true pixel size at the vessel, per patient
    calibrated_fraction: float = Field(0.8, ge=0, le=1)  # patients with a given pixel size
    calibration_error_rel: float = Field(0.03, ge=0)  # SD of the given size's relative error
    click_sd_px: float = Field(1.5, ge=0)  # start / sensor point placement error
    # contrast and processing fitted to ARCADE (docs/DECISIONS.md, D39): attenuation x pixel
    # size 0.010-0.020 per px of lumen, then edge enhancement, noise as measured there
    mu_per_mm: Range = (0.04, 0.08)  # contrast attenuation per mm of lumen
    background: Range = (0.49, 0.64)
    texture_sd: float = Field(0.04, ge=0)
    texture_scale_px: float = Field(40.0, gt=0)
    blur_px: Range = (0.8, 1.2)
    edge_amount: Range = (2.0, 3.0)
    edge_sigma_px: Range = (1.5, 3.0)
    noise_sd: Range = (0.012, 0.022)
    # layout
    stub_mm: float = Field(6.0, ge=0)  # main vessel drawn before the start point
    continuation_mm: Range = (5.0, 15.0)  # and beyond the sensor
    side_angle_deg: Range = (35.0, 65.0)
    curvature_rad: Range = (0.0, 0.5)  # heading oscillation amplitude of the main vessel
    side_curvature_rad: Range = (0.0, 0.3)
    wavelength_px: Range = (150.0, 400.0)
    clearance_px: float = Field(3.0, ge=0)  # between vessel walls that do not join
    margin_px: float = Field(8.0, ge=0)  # from the image border
    layout_tries: int = Field(40, gt=0)

    @model_validator(mode="after")
    def _check(self) -> "SyntheticStudyConfig":
        p = self.vessels_per_patient_probs
        if not p or abs(sum(p) - 1.0) > 1e-6 or min(p) < 0:
            raise ValueError("vessels_per_patient_probs must be non-negative and sum to 1")
        if len(p) > len(self.territories) or len(set(self.territories)) != len(self.territories):
            raise ValueError("at most one vessel per territory: need distinct territories")
        return self


@dataclass(frozen=True, eq=False)
class Layout:
    """A tree drawn in the image plane (px): ``tubes`` are (points (n, 2), radius (n,)); the
    main vessel (with its stub and continuation) comes first. ``start_px`` / ``end_px`` are the
    main centreline at its first and last sample (the start point and the sensor)."""

    tubes: list[tuple[np.ndarray, np.ndarray]]
    start_px: np.ndarray
    end_px: np.ndarray


def _heading(u: np.ndarray, theta0: float, amp: float, wavelength: float, phase: float):
    return theta0 + amp * np.sin(2.0 * np.pi * u / wavelength + phase)


def _trace(u: np.ndarray, *heading_args) -> np.ndarray:
    """Positions (n, 2) as (row, col) along arc length ``u`` (px), relative to ``u[0]``."""
    th = _heading(0.5 * (u[1:] + u[:-1]), *heading_args)
    steps = np.diff(u)[:, None] * np.stack([np.sin(th), np.cos(th)], axis=1)
    return np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])


def _at(u: np.ndarray, pos: np.ndarray, x: float) -> np.ndarray:
    return np.array([np.interp(x, u, pos[:, 0]), np.interp(x, u, pos[:, 1])])


def _inside(pos: np.ndarray, radius: np.ndarray, size: int, margin: float) -> bool:
    pad = radius[:, None] + margin
    return bool(np.all(pos >= pad) and np.all(pos <= size - 1 - pad))


def _clear(pos, radius, others, gap: float) -> bool:
    """Whether every point keeps ``gap`` px between its wall and every tube in ``others``."""
    for pts, rad in others:
        d, idx = cKDTree(pts).query(pos)
        if np.any(d < radius + rad[idx] + gap):
            return False
    return True


def _main_layout(rng, tree: VesselTree, px_m: float, cfg: SyntheticStudyConfig):
    main = tree.branches[0]
    length = main.length / px_m
    stub = cfg.stub_mm * 1e-3 / px_m
    cont = rng.uniform(*cfg.continuation_mm) * 1e-3 / px_m
    u = np.arange(-stub, length + cont + STEP_PX / 2, STEP_PX)
    args = (rng.uniform(0, 2 * np.pi), rng.uniform(*cfg.curvature_rad),
            rng.uniform(*cfg.wavelength_px), rng.uniform(0, 2 * np.pi))  # fmt: skip
    pos = _trace(u, *args)
    r_ref = main.reference_radius()
    radius = np.interp(u, main.s / px_m, main.r / px_m)
    radius[u < 0] = r_ref[0] / px_m
    beyond = u > length
    radius[beyond] = r_ref[-1] / px_m * (1.0 - 0.15 * (u[beyond] - length) / max(cont, 1e-9))
    pad = radius.max() + cfg.margin_px
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    slack = (cfg.image_size - 1 - 2 * pad) - (hi - lo)
    if np.any(slack < 0):
        return None
    pos = pos - lo + pad + rng.uniform(0, 1, 2) * slack
    return u, pos, radius, args


def layout_tree(
    rng: np.random.Generator, tree: VesselTree, mm_per_px: float, cfg: SyntheticStudyConfig
) -> Layout | None:
    """Draw ``tree`` in a ``cfg.image_size`` image at ``mm_per_px``: a gently curving main
    vessel, side branches leaving at ``side_angle_deg`` on alternating sides, inside the image
    and without touching each other. ``None`` if no layout is found in ``cfg.layout_tries``."""
    px_m = mm_per_px * 1e-3
    for _ in range(cfg.layout_tries):
        laid = _main_layout(rng, tree, px_m, cfg)
        if laid is None:
            continue
        u, pos, radius, args = laid
        tubes = [(pos, radius)]
        side = rng.choice([-1.0, 1.0])
        for k in range(1, tree.n_branches):
            b = tree.branches[k]
            u_t = tree.branches[0].s[tree.attach[k]] / px_m
            p_t = _at(u, pos, u_t)
            r_parent = float(np.interp(u_t, u, radius))
            ub = np.arange(0.0, b.length / px_m + STEP_PX / 2, STEP_PX)
            rb = np.interp(ub, b.s / px_m, b.r / px_m)
            placed = False
            for _ in range(5):
                angle = side * np.deg2rad(rng.uniform(*cfg.side_angle_deg))
                curve = (_heading(u_t, *args) + angle, rng.uniform(*cfg.side_curvature_rad),
                         rng.uniform(*cfg.wavelength_px), rng.uniform(0, 2 * np.pi))  # fmt: skip
                pb = p_t + _trace(ub, *curve)
                # beyond this distance from the take-off the branch must be clear of the others
                near = 2.0 * (r_parent + rb[0] + cfg.clearance_px)
                far = np.linalg.norm(pb - p_t, axis=1) > near
                if _inside(pb, rb, cfg.image_size, cfg.margin_px) and _clear(
                    pb[far], rb[far], tubes, cfg.clearance_px
                ):
                    placed = True
                    break
            if not placed:
                break
            tubes.append((pb, rb))
            side = -side
        else:
            length = tree.branches[0].length / px_m
            return Layout(tubes, _at(u, pos, 0.0), _at(u, pos, length))
    return None


def render_case(
    rng: np.random.Generator, layout: Layout, mm_per_px: float, cfg: SyntheticStudyConfig
) -> np.ndarray:
    """The layout as an 8-bit angiogram-like image (textured background, dark vessels)."""
    size = cfg.image_size
    texture = gaussian_filter(rng.standard_normal((size, size)), cfg.texture_scale_px)
    texture *= cfg.texture_sd / max(float(texture.std()), 1e-12)
    img = render_projection(
        (size, size), layout.tubes, mm_per_px=mm_per_px,
        mu_per_mm=rng.uniform(*cfg.mu_per_mm),
        background=rng.uniform(*cfg.background) + texture,
        blur_sigma=rng.uniform(*cfg.blur_px), edge_amount=rng.uniform(*cfg.edge_amount),
        edge_sigma=rng.uniform(*cfg.edge_sigma_px), noise_std=rng.uniform(*cfg.noise_sd),
        seed=int(rng.integers(2**31)),
    )  # fmt: skip
    return np.round(img * 255.0).astype(np.uint8)


def lumen_mask(layout: Layout, size: int) -> np.ndarray:
    """The true lumen as an 8-bit mask: each pixel's covered fraction × 255 (an ideal,
    analyst-corrected contour; the QCA reads it like a probability map)."""
    cover = render_tubes((size, size), [(p, 2.0 * r) for p, r in layout.tubes], blur_sigma=0.0,
                         noise_std=0.0)  # fmt: skip
    return np.round(cover * 255.0).astype(np.uint8)


def _click(rng, point: np.ndarray, sd: float, size: int) -> tuple[int, int]:
    p = np.clip(np.round(point + rng.normal(0.0, sd, 2)), 0, size - 1)
    return int(p[0]), int(p[1])


def _write_csv(path: Path, columns, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def generate_synthetic_study(
    cfg: SyntheticStudyConfig, physics: PhysicsParams, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Generate the stand-in study (see the module docstring); returns a summary."""
    out = save_run_metadata(out_dir or cfg.out_dir, cfg)
    (out / "images").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    cases, refs, truth = [], [], []
    resampled = 0
    n_choices = len(cfg.vessels_per_patient_probs)
    for p in range(cfg.n_patients):
        patient = f"SYN-{p + 1:03d}"
        n_vessels = 1 + int(rng.choice(n_choices, p=cfg.vessels_per_patient_probs))
        territories = rng.choice(cfg.territories, size=n_vessels, replace=False)
        mm_per_px = float(rng.uniform(*cfg.mm_per_px))
        calibrated = bool(rng.random() < cfg.calibrated_fraction)
        given = mm_per_px * (1.0 + rng.normal(0.0, cfg.calibration_error_rel))
        pa = pv = None
        for territory in sorted(territories, key=cfg.territories.index):
            for _ in range(MAX_RESAMPLES):
                st = sample_tree(rng, cfg.sampling, cfg.n_points)
                layout = layout_tree(rng, st.tree, mm_per_px, cfg)
                if layout is not None:
                    break
                resampled += 1
            else:
                raise RuntimeError(f"{patient}: no layout found in {MAX_RESAMPLES} trees")
            if pa is None:  # one Pa and Pv per patient
                pa, pv = st.pa, st.pv
            case = f"{patient}-{territory}"
            image = render_case(rng, layout, mm_per_px, cfg)
            cv2.imwrite(str(out / "images" / f"{case}.png"), image)
            cv2.imwrite(str(out / "masks" / f"{case}.png"), lumen_mask(layout, cfg.image_size))
            ref = solve_tree(st.tree, physics, pa=pa, pv=pv, v_hyp=st.v_hyp)
            fixed = solve_tree(st.tree, physics, pa=pa)  # the pipeline's Pv and velocity
            truth_params = physics.model_copy(
                update={"pa_mmhg": pa_to_mmhg(pa), "pv_mmhg": pa_to_mmhg(pv), "v_hyp": st.v_hyp}
            )
            main = st.tree.branches[0]
            start = _click(rng, layout.start_px, cfg.click_sd_px, cfg.image_size)
            end = _click(rng, layout.end_px, cfg.click_sd_px, cfg.image_size)
            cases.append({
                "case_id": case, "patient_id": patient, "vessel": territory,
                "image": f"images/{case}.png", "frame": "",
                "start_row": start[0], "start_col": start[1],
                "end_row": end[0], "end_col": end[1],
                "mm_per_px": f"{given:.5f}" if calibrated else "",
                "calibration": "synthetic calibration" if calibrated else "",
                "pa_mmhg": f"{pa_to_mmhg(pa):.1f}", "pv_mmhg": "", "mask": f"masks/{case}.png",
            })  # fmt: skip
            refs.append({"case_id": case, "ref_type": "FFR",
                         "ref_value": f"{ref.ffr[0][-1]:.4f}", "excluded": ""})  # fmt: skip
            truth.append({
                "case_id": case,
                "true_mm_per_px": round(mm_per_px, 5),
                "length_mm": round(m_to_mm(main.length), 2),
                "mld_mm": round(float(m_to_mm(main.diameter.min())), 3),
                "ref_inlet_mm": round(float(m_to_mm(2.0 * main.reference_radius()[0])), 3),
                "max_ds_pct": round(float(main.percent_stenosis().max()), 1),
                "n_side_branches": st.tree.n_branches - 1,
                "side_lengths_mm": " ".join(
                    f"{m_to_mm(b.length):.1f}" for b in st.tree.branches[1:]
                ),
                "pv_mmhg": round(pa_to_mmhg(pv), 2),
                "v_hyp": round(st.v_hyp, 4),
                "ffr_tree": round(float(ref.ffr[0][-1]), 4),
                "ffr_tree_fixed_inputs": round(float(fixed.ffr[0][-1]), 4),
                "ffr_main_alone": round(float(solve_ffr(main, truth_params).ffr[-1]), 4),
            })  # fmt: skip
    _write_csv(out / "cases.csv", CASE_COLUMNS, cases)
    _write_csv(out / "reference.csv", REFERENCE_COLUMNS, refs)
    _write_csv(out / "truth.csv", list(truth[0]), truth)
    ffr = np.array([float(r["ref_value"]) for r in refs])
    return {
        "out_dir": str(out),
        "n_patients": cfg.n_patients,
        "n_cases": len(cases),
        "layout_resamples": resampled,
        "prevalence_ffr_le_080": float(np.mean(ffr <= 0.80)),
        "calibrated_cases": sum(bool(c["mm_per_px"]) for c in cases),
    }
