"""Two-view 3D reconstruction on synthetic cases (P5) → ``reports/two_view_synthetic.md``.

Every case is a synthetic 3D tree (:mod:`vstent.imaging.synthetic3d`) rendered in two views and
analysed with the product's own steps: centreline between the clicked start and end points,
QCA with intensity edges, then :func:`~vstent.imaging.twoview.reconstruct` (with the motion
correction, the default) and :func:`~vstent.imaging.twoview.two_view_profile`. Compared with
the truth: the 3D centreline, its length, the lumen diameters and the distal FFR of the
reduced-order solver. Single-view QCA of each view is the comparison, with the pixel size at the
isocentre (DICOM geometry, P5) and with the uncorrected detector spacing.

Scenarios:
- **static:** views chosen without overlap of the analysed vessel, as QCA practice asks
  (acceptance targets: median 3D error ≤ 1 mm, median |length error| ≤ 5%);
- **motion:** the same cases with the heart shifted between the runs (``motion_sd_mm``),
  reconstructed without and with the shift correction;
- **stress:** random views, overlap allowed (a separate set of cases).

A case whose analysis fails (no path between the points, too few points) counts as an infinite
error in the medians.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from vstent import DISCLAIMER, __version__
from vstent.config import git_info, load_config
from vstent.imaging.centerline import trace_centerline
from vstent.imaging.qca import qca_from_centerline
from vstent.imaging.synthetic3d import (
    TwoViewCase,
    TwoViewSynthConfig,
    main_overlaps,
    render_view,
    sample_case,
)
from vstent.imaging.twoview import Reconstruction, reconstruct, two_view_profile
from vstent.physics.geometry import VesselProfile
from vstent.physics.solver import PhysicsParams, solve_ffr

CONFIG = Path("configs/two_view_synthetic.yaml")
PHYSICS = Path("configs/physics.yaml")
REPORT = "two_view_synthetic"
STRESS_SEED_OFFSET = 100_000
TARGET_ERROR_MM = 1.0  # median over static cases of the per-case mean 3D distance
TARGET_LENGTH = 0.05  # median over static cases of |length error|


def _trace(view, *, trimmed: bool):
    """Centreline and QCA of the main vessel in one rendered view (isocentre pixel size)."""
    cl = trace_centerline(view.prob >= 0.5, view.start, view.end)
    q = qca_from_centerline(
        view.prob,
        cl,
        mm_per_px=view.geometry.iso_pixel_m * 1e3,
        image=view.image,
        **({} if trimmed else {"trim_diameters": 0}),
    )
    return cl, q


def _shape_error(points: np.ndarray, kd: cKDTree, truth: np.ndarray) -> float:
    """Mean distance (m) after removing the best translation (iterated nearest points)."""
    x = points.copy()
    for _ in range(20):
        _, idx = kd.query(x)
        delta = (truth[idx] - x).mean(axis=0)
        x += delta
        if np.linalg.norm(delta) < 1e-7:
            break
    return float(kd.query(x)[0].mean())


def _true_diameter(case: TwoViewCase, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, r_ref = case.radii(0)
    return 2.0 * r[idx], 2.0 * r_ref[idx]


def _diameter_errors(d: np.ndarray, d_ref: np.ndarray, true_d, true_ref) -> dict[str, float]:
    return {
        "d_mae_mm": float(np.mean(np.abs(d - true_d)) * 1e3),
        "ref_err": float(np.median(d_ref) / np.median(true_ref) - 1.0),
    }


def _scaled(profile: VesselProfile, k: float) -> VesselProfile:
    return VesselProfile(profile.s * k, profile.r * k, profile.reference_radius() * k)


def _two_view(case, rec: Reconstruction, traced, kd, params) -> dict[str, Any]:
    """Metrics of one reconstruction against the truth."""
    truth = case.curves[0]
    (cl1, q1), (cl2, q2) = traced
    err = kd.query(rec.points)[0]
    out = {
        "length_err": rec.length / case.main_length - 1.0,
        "err3d_mm": float(err.mean() * 1e3),
        "shape_mm": _shape_error(rec.points, kd, truth) * 1e3,
        "gap_mm": rec.gap_m * 1e3,
        "residual_px": rec.residual_px.tolist(),
        "weak_fraction": rec.weak_fraction,
        "shift_px": rec.shift_px.tolist(),
        "warnings": rec.warnings(),
    }
    prof = two_view_profile(
        rec, (q1.diameters.diameter_px, q2.diameters.diameter_px), (cl1.s, cl2.s)
    )
    s3 = prof.profile.s + prof.offset_m
    pts = np.stack([np.interp(s3, rec.s, rec.points[:, k]) for k in range(3)], axis=1)
    _, idx = kd.query(pts)
    out.update(
        _diameter_errors(
            2.0 * prof.profile.r, 2.0 * prof.profile.reference_radius(), *_true_diameter(case, idx)
        )
    )
    out["ffr"] = float(solve_ffr(prof.profile, params).ffr[-1])
    # geometry alone: the true lumen with the reconstruction's length and magnification
    _, idx = kd.query(rec.points)
    g1 = rec.geometries[0]
    k_r = float(np.median(g1.pixel_size_at(rec.points) / g1.pixel_size_at(truth[idx])))
    s_true = case.curve_s[0] * rec.length / case.main_length
    out["ffr_geom"] = float(solve_ffr(_true_profile(case, s_true, k_r), params).ffr[-1])
    return out


def _true_profile(case, s: np.ndarray, k_r: np.ndarray) -> VesselProfile:
    """The true lumen at positions ``s`` (m, non-decreasing, one per fine truth point) with
    radii scaled by ``k_r``, resampled uniformly."""
    r, r_ref = case.radii(0)
    su = np.linspace(0.0, float(s[-1]), case.tree.tree.branches[0].n)
    return VesselProfile(su, np.interp(su, s, r * k_r), np.interp(su, s, r_ref * k_r))


def _geometry_ffr(case, g, m_per_px: float, params) -> float:
    """Distal FFR of the true lumen seen in one view with a perfect measurement in pixels:
    length = projected length × ``m_per_px``, radius × ``m_per_px`` / the true pixel size at
    each point. Only the view's geometry (foreshortening, magnification) is wrong."""
    x = case.curves[0]
    p = g.project(x)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]) * m_per_px
    return float(solve_ffr(_true_profile(case, s, m_per_px / g.pixel_size_at(x)), params).ffr[-1])


def _single(case, k: int, view, traced, params) -> dict[str, Any]:
    """Single-view QCA of view ``k``: isocentre pixel size and uncorrected detector spacing."""
    cl, q = traced
    g = view.geometry
    magnification = g.magnification
    length = cl.length * g.iso_pixel_m
    proj = g.project(case.curves[0])
    _, idx = cKDTree(proj).query(q.centerline.points)
    true_d, true_ref = _true_diameter(case, idx)
    d, d_ref = 2.0 * q.profile.r, 2.0 * q.profile.reference_radius()
    iso = _diameter_errors(d, d_ref, true_d, true_ref)
    det = _diameter_errors(d * magnification, d_ref * magnification, true_d, true_ref)
    return {
        "view": k,
        "length_err": length / case.main_length - 1.0,
        "length_err_detector": length * magnification / case.main_length - 1.0,
        **iso,
        "d_mae_mm_detector": det["d_mae_mm"],
        "ref_err_detector": det["ref_err"],
        "ffr": float(solve_ffr(q.profile, params).ffr[-1]),
        "ffr_detector": float(solve_ffr(_scaled(q.profile, magnification), params).ffr[-1]),
        "ffr_geom": _geometry_ffr(case, g, g.iso_pixel_m, params),
        "pixel_err": float(np.median(g.iso_pixel_m / g.pixel_size_at(case.curves[0])) - 1.0),
        "ffr_geom_detector": _geometry_ffr(case, g, float(np.mean(g.pixel_m)), params),
    }


def run_case(seed: int, cfg: TwoViewSynthConfig, scenario: str) -> dict[str, Any]:
    """All metrics of one case. ``scenario``: ``"static"`` (also runs the motion scenario) or
    ``"stress"``."""
    params = load_config(PHYSICS, PhysicsParams)
    case = sample_case(seed, cfg)
    g1, g2 = case.views
    row: dict[str, Any] = {
        "seed": seed,
        "scenario": scenario,
        "views": [g1.label(), g2.label()],
        "separation_deg": g1.separation_deg(g2),
        "length_mm": case.main_length * 1e3,
        "n_branches": case.tree.tree.n_branches,
        "motion_mm": float(np.linalg.norm(case.motion_m) * 1e3),
        "true_ffr": float(solve_ffr(case.tree.tree.branches[0], params).ffr[-1]),
        "true_ffr_geom": float(
            solve_ffr(_true_profile(case, case.curve_s[0], 1.0), params).ffr[-1]
        ),
    }
    views = [render_view(case, 0, cfg), render_view(case, 1, cfg)]
    if scenario == "stress":
        r_ref = [
            np.interp(a, b.s, b.reference_radius())
            for a, b in zip(case.curve_s, case.tree.tree.branches, strict=True)
        ]
        row["overlap"] = any(
            main_overlaps(case.curves, r_ref, g, cfg.overlap_gap_px) for g in case.views
        )
    kd = cKDTree(case.curves[0])
    try:
        traced = [_trace(v, trimmed=False) for v in views]
        single = [_trace(v, trimmed=True) for v in views]
        row["single"] = [_single(case, k, views[k], single[k], params) for k in range(2)]
        rec = reconstruct(g1, traced[0][0], g2, traced[1][0])
        row["two_view"] = _two_view(case, rec, traced, kd, params)
        if scenario == "static":
            rec_nc = reconstruct(g1, traced[0][0], g2, traced[1][0], correct_shift=False)
            row["two_view_no_correction"] = _two_view(case, rec_nc, traced, kd, params)
            moved = render_view(case, 1, cfg, moved=True)
            traced_m = [traced[0], _trace(moved, trimmed=False)]
            for key, correct in (("motion_no_correction", False), ("motion", True)):
                rec_m = reconstruct(g1, traced[0][0], g2, traced_m[1][0], correct_shift=correct)
                row[key] = _two_view(case, rec_m, traced_m, kd, params)
    except ValueError as exc:
        row["failed"] = str(exc)
    return row


def _task(args) -> dict[str, Any]:
    seed, cfg_dict, scenario = args
    return run_case(seed, TwoViewSynthConfig.model_validate(cfg_dict), scenario)


def run_cases(cfg: TwoViewSynthConfig, n_stress: int, workers: int = 1) -> list[dict[str, Any]]:
    """Static cases (seeds ``cfg.seed + i``) and stress cases (random views; seeds from
    ``cfg.seed + STRESS_SEED_OFFSET``). Row order does not depend on ``workers``."""
    stress_cfg = cfg.model_copy(update={"avoid_overlap": False}).model_dump(mode="json")
    tasks = [(cfg.seed + i, cfg.model_dump(mode="json"), "static") for i in range(cfg.n_cases)]
    tasks += [(cfg.seed + STRESS_SEED_OFFSET + i, stress_cfg, "stress") for i in range(n_stress)]
    if workers <= 1:
        return [_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_task, tasks, chunksize=1))


def _stats(values, *, absolute: bool = False, scale: float = 1.0) -> dict[str, float | None]:
    """Median, 90th percentile and max (of |values| if ``absolute``); NaN = failed (→ inf)."""
    v = np.asarray([np.inf if x is None else x for x in values], dtype=float)
    if v.size == 0:
        return {"n": 0, "median": None, "p90": None, "max": None}
    v = np.abs(v) if absolute else v
    v = np.where(np.isnan(v), np.inf, v) * scale

    def num(x):
        return None if not np.isfinite(x) else float(x)

    return {
        "n": int(v.size),
        "median": num(np.median(v)),
        "p90": num(np.percentile(v, 90)),
        "max": num(np.max(v)),
    }


def _get(rows, key: str, field: str) -> list:
    return [r[key][field] if key in r else None for r in rows]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary statistics and the acceptance targets."""
    static = [r for r in rows if r["scenario"] == "static"]
    stress = [r for r in rows if r["scenario"] == "stress"]
    ok = [r for r in static if "failed" not in r]

    def geom_err(rs, key):
        return [abs(r[key]["ffr_geom"] - r["true_ffr_geom"]) if key in r else None for r in rs]

    def ffr_err(rs, key, field="ffr"):
        return [abs(r[key][field] - r["true_ffr"]) if key in r else None for r in rs]

    singles = [s for r in ok for s in r["single"]]
    summary: dict[str, Any] = {
        "n_static": len(static),
        "n_failed": len(static) - len(ok),
        "failures": [{"seed": r["seed"], "reason": r["failed"]} for r in static if "failed" in r],
        "err3d_mm": _stats(_get(static, "two_view", "err3d_mm")),
        "length_abs": _stats(_get(static, "two_view", "length_err"), absolute=True, scale=100),
        "length_bias": _stats(_get(ok, "two_view", "length_err"), scale=100),
        "within_5pct": float(
            np.mean([abs(x) <= TARGET_LENGTH for x in _get(ok, "two_view", "length_err")])
        )
        if ok
        else None,
        "ffr_abs": _stats(ffr_err(static, "two_view")),
        "ffr_geom_abs": _stats(geom_err(static, "two_view")),
        "ref_abs": _stats(_get(static, "two_view", "ref_err"), absolute=True, scale=100),
        "ref_bias": _stats(_get(ok, "two_view", "ref_err"), scale=100),
        "d_mae_mm": _stats(_get(static, "two_view", "d_mae_mm")),
        "no_correction": {
            "err3d_mm": _stats(_get(static, "two_view_no_correction", "err3d_mm")),
            "length_abs": _stats(
                _get(static, "two_view_no_correction", "length_err"), absolute=True, scale=100
            ),
        },
        "warned": int(sum(bool(r["two_view"]["warnings"]) for r in ok)),
        "large": [
            {
                "seed": r["seed"],
                "length_err": r["two_view"]["length_err"],
                "err3d_mm": r["two_view"]["err3d_mm"],
                "separation_deg": r["separation_deg"],
                "single_length_err": [s["length_err"] for s in r["single"]],
                "flagged": bool(r["two_view"]["warnings"]),
            }
            for r in ok
            if abs(r["two_view"]["length_err"]) > TARGET_LENGTH
        ],
        "single": {
            "n": len(singles),
            "length_bias": _stats([s["length_err"] for s in singles], scale=100),
            "length_abs": _stats([s["length_err"] for s in singles], absolute=True, scale=100),
            "length_abs_detector": _stats(
                [s["length_err_detector"] for s in singles], absolute=True, scale=100
            ),
            "within_5pct": float(np.mean([abs(s["length_err"]) <= TARGET_LENGTH for s in singles]))
            if singles
            else None,
            "ref_abs": _stats([s["ref_err"] for s in singles], absolute=True, scale=100),
            "ref_abs_detector": _stats(
                [s["ref_err_detector"] for s in singles], absolute=True, scale=100
            ),
            "d_mae_mm": _stats([s["d_mae_mm"] for s in singles]),
            "pixel_abs": _stats([s["pixel_err"] for s in singles], absolute=True, scale=100),
            "ffr_abs": _stats([abs(s["ffr"] - r["true_ffr"]) for r in ok for s in r["single"]]),
            "ffr_abs_detector": _stats(
                [abs(s["ffr_detector"] - r["true_ffr"]) for r in ok for s in r["single"]]
            ),
            "ffr_geom_abs": _stats(
                [abs(s["ffr_geom"] - r["true_ffr_geom"]) for r in ok for s in r["single"]]
            ),
            "ffr_geom_abs_detector": _stats(
                [abs(s["ffr_geom_detector"] - r["true_ffr_geom"]) for r in ok for s in r["single"]]
            ),
        },
        "motion": {
            key: {
                "length_abs": _stats(_get(static, key, "length_err"), absolute=True, scale=100),
                "err3d_mm": _stats(_get(static, key, "err3d_mm")),
                "shape_mm": _stats(_get(static, key, "shape_mm")),
                "ffr_abs": _stats(ffr_err(static, key)),
                "warned": int(sum(bool(r[key]["warnings"]) for r in ok if key in r)),
            }
            for key in ("motion_no_correction", "motion")
        },
        "motion_mm": _stats([r["motion_mm"] for r in static]),
        "separation_deg": _stats([r["separation_deg"] for r in static]),
        "length_mm": _stats([r["length_mm"] for r in static]),
    }
    stress_ok = [r for r in stress if "failed" not in r]
    summary["stress"] = {
        "n": len(stress),
        "n_failed": len(stress) - len(stress_ok),
        "n_overlap": int(sum(r["overlap"] for r in stress)),
        **{
            group: {
                "n": len(rs),
                "length_abs": _stats(_get(rs, "two_view", "length_err"), absolute=True, scale=100),
                "err3d_mm": _stats(_get(rs, "two_view", "err3d_mm")),
                "single_length_abs": _stats(
                    [s["length_err"] for r in rs if "single" in r for s in r["single"]],
                    absolute=True,
                    scale=100,
                ),
            }
            for group, rs in (
                ("overlap", [r for r in stress if r["overlap"]]),
                ("clear", [r for r in stress if not r["overlap"]]),
            )
        },
    }
    err, length = summary["err3d_mm"]["median"], summary["length_abs"]["median"]
    summary["targets"] = [
        {
            "metric": "3D centreline error, median over cases of the per-case mean (mm)",
            "value": "–" if err is None else f"{err:.2f}",
            "target": f"≤ {TARGET_ERROR_MM:g}",
            "passed": err is not None and err <= TARGET_ERROR_MM,
        },
        {
            "metric": "Length error, median over cases of |error| (%)",
            "value": "–" if length is None else f"{length:.1f}",
            "target": f"≤ {TARGET_LENGTH:.0%}",
            "passed": length is not None and length <= TARGET_LENGTH * 100,
        },
    ]
    summary["targets_met"] = all(t["passed"] for t in summary["targets"])
    return summary


METHOD_COLOURS = {"two": "SERIES_1", "iso": "SERIES_2", "detector": "SERIES_3"}


def _example_figure(cfg: TwoViewSynthConfig, path: Path) -> None:
    """The first static case: both views with the traced and the reprojected centreline, and
    the 3D centreline against the truth."""
    from vstent.plotting import (
        INK_MUTED,
        INK_SECONDARY,
        SERIES_1,
        SERIES_2,
        new_figure,
        save_figure,
    )

    case = sample_case(cfg.seed, cfg)
    views = [render_view(case, 0, cfg), render_view(case, 1, cfg)]
    traced = [_trace(v, trimmed=False)[0] for v in views]
    rec = reconstruct(case.views[0], traced[0], case.views[1], traced[1])
    single = " and ".join(
        f"{c.length * v.geometry.iso_pixel_m * 1e3:.1f}" for c, v in zip(traced, views, strict=True)
    )
    fig = new_figure(
        10.5,
        4.2,
        "Two-view reconstruction, first synthetic case",
        f"{case.views[0].label()} and {case.views[1].label()}, "
        f"{rec.separation_deg:.0f}° apart. True length {case.main_length * 1e3:.1f} "
        f"mm, two views {rec.length * 1e3:.1f} mm, single views {single} mm",
    )
    for k in range(2):
        ax = fig.add_axes((0.04 + 0.31 * k, 0.1, 0.29, 0.74))
        ax.imshow(views[k].image, cmap="gray", vmin=0, vmax=255)
        ax.plot(
            traced[k].points[:, 1],
            traced[k].points[:, 0],
            color=SERIES_2,
            lw=1.0,
            label="traced in the image",
        )
        back = rec.geometries[k].project(rec.points)
        ax.plot(
            back[:, 1],
            back[:, 0],
            color=SERIES_1,
            lw=1.0,
            ls=(0, (3, 2)),
            label="3D centreline, projected back",
        )
        ax.set_title(f"View {k + 1}: {case.views[k].label()}", fontsize=8, color=INK_SECONDARY)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.legend(fontsize=6.5, loc="lower left", frameon=False, labelcolor="white")
    ax = fig.add_axes((0.66, 0.1, 0.32, 0.74), projection="3d")
    t, p = case.curves[0] * 1e3, rec.points * 1e3
    ax.plot(t[:, 0], t[:, 1], t[:, 2], color=INK_MUTED, lw=3.0, alpha=0.5, label="truth")
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=SERIES_1, lw=1.0, label="two views")
    for setter, name in (
        (ax.set_xlabel, "x (mm, left)"),
        (ax.set_ylabel, "y (mm, posterior)"),
        (ax.set_zlabel, "z (mm, head)"),
    ):
        setter(name, fontsize=7, color=INK_SECONDARY)
    ax.tick_params(labelsize=6, colors=INK_MUTED)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    save_figure(fig, path)


def _errors_figure(rows: list[dict[str, Any]], path: Path) -> None:
    """Length error and |FFR error| of single-view and two-view analysis, one dot per view
    (single) or case (two views)."""
    from vstent import plotting
    from vstent.plotting import INK_PRIMARY, INK_SECONDARY, new_figure, save_figure, style_axes

    ok = [r for r in rows if r["scenario"] == "static" and "failed" not in r]
    singles = [(s, r) for r in ok for s in r["single"]]
    iso, two = "Single view,\nisocentre pixel", "Two views"
    groups = {
        "length": [
            (iso, "iso", [100 * s["length_err"] for s, _ in singles]),
            (two, "two", [100 * r["two_view"]["length_err"] for r in ok]),
        ],
        "ffr": [
            (
                "Single view,\ndetector spacing",
                "detector",
                [abs(s["ffr_geom_detector"] - r["true_ffr_geom"]) for s, r in singles],
            ),
            (iso, "iso", [abs(s["ffr_geom"] - r["true_ffr_geom"]) for s, r in singles]),
            (two, "two", [abs(r["two_view"]["ffr_geom"] - r["true_ffr_geom"]) for r in ok]),
        ],
    }
    fig = new_figure(
        9.0,
        3.6,
        "Single view vs two views on synthetic cases",
        f"{len(ok)} static cases (single view: both views); short line: median",
    )
    rng = np.random.default_rng(0)
    for k, (key, xlabel) in enumerate(
        (
            ("length", "Length error vs truth (%)"),
            ("ffr", "|FFR error| from geometry alone (distal)"),
        )
    ):
        ax = fig.add_axes((0.17 + 0.47 * k, 0.17, 0.33, 0.66))
        style_axes(ax)
        for y, (_, method, values) in enumerate(groups[key]):
            colour = getattr(plotting, METHOD_COLOURS[method])
            jitter = rng.uniform(-0.18, 0.18, len(values))
            ax.scatter(values, y + jitter, s=9, color=colour, alpha=0.55, lw=0, zorder=3)
            ax.plot(
                [np.median(values)] * 2, [y - 0.32, y + 0.32], color=INK_PRIMARY, lw=1.5, zorder=4
            )
        ax.set_yticks(range(len(groups[key])), [g[0] for g in groups[key]], fontsize=7.5)
        ax.set_ylim(-0.6, len(groups[key]) - 0.4)
        ax.set_xlabel(xlabel, fontsize=8, color=INK_SECONDARY)
        if key == "length":
            ax.axvline(0.0, color=INK_SECONDARY, lw=0.8, zorder=2)
    save_figure(fig, path)


def _fmt(stats: dict[str, Any], digits: int = 1, field: str = "median") -> str:
    v = stats.get(field)
    if v is None:
        return "∞ (failures)" if stats.get("n") else "–"
    return f"{v:.{digits}f}"


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def write_two_view_report(
    report_dir: str | Path = "reports",
    config: str | Path = CONFIG,
    workers: int = 1,
    n_stress: int | None = None,
) -> dict[str, Any]:
    """Run the synthetic cases and write ``two_view_synthetic.md`` / ``.json`` and figures."""
    cfg = load_config(config, TwoViewSynthConfig)
    n_stress = cfg.n_stress if n_stress is None else n_stress
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    rows = run_cases(cfg, n_stress, workers)
    summary = summarise(rows)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        **summary,
        "config": cfg.model_dump(mode="json"),
        "rows": rows,
        "provenance": provenance,
    }
    (report_dir / f"{REPORT}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    _example_figure(cfg, report_dir / "figures" / "two_view_example.png")
    _errors_figure(rows, report_dir / "figures" / "two_view_errors.png")
    s, single, motion, stress = summary, summary["single"], summary["motion"], summary["stress"]
    ok = [r for r in rows if r["scenario"] == "static" and "failed" not in r]
    nc = s["no_correction"]

    def single_ffr(r):
        return ", ".join(f"{x['ffr']:.3f}" for x in r["single"])

    lines = [
        "# Two-view 3D reconstruction on synthetic angiograms (P5)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent two-view-synthetic`. Do not edit by hand.",
        "",
        f"{s['n_static']} synthetic coronary trees (P2 lumen profiles with lesions and side "
        "branches) laid on a heart-sized sphere, each seen in two C-arm views and rendered as "
        "X-ray-like images with the ARCADE-fitted contrast, blur, edge enhancement and noise "
        "(D39). The views are chosen as QCA practice asks: the analysed vessel overlaps neither "
        "itself nor a side branch; foreshortening is not avoided. Each case is analysed with "
        "the product's own steps: the centreline between the clicked start and end points "
        f"({cfg.click_sd_px:g} px click error), QCA with intensity edges, then the two-view "
        "reconstruction (D49) with its motion correction. Single-view QCA of each view is the "
        "comparison, with the pixel size at the isocentre (from the DICOM geometry, P5) and "
        "with the uncorrected detector spacing (what the loader offered before P5). FFR: the "
        "reduced-order solver (default physics) on each profile and on the true lumen.",
        "",
        "**Caveats.** The images are rendered with the same projection model the "
        "reconstruction uses, so the errors of a real C-arm are not tested: the angle "
        "convention (D49, to be confirmed on real two-view DICOM), isocentre and calibration "
        "errors, image distortion, and non-rigid motion (two runs at different cardiac phases). "
        "The segmentation stand-in is the rendered vessel. This is the method's best case on "
        "known geometry; real two-view data are needed next.",
        "",
        "| Target | Value | Required | Status |",
        "|---|---|---|---|",
        *(
            f"| {t['metric']} | {t['value']} | {t['target']} | "
            f"{'PASS' if t['passed'] else 'FAIL'} |"
            for t in s["targets"]
        ),
        "",
        f"Cases: {s['n_static']}; the analysis failed in {s['n_failed']} (counted as infinite "
        f"error). The views were {_fmt(s['separation_deg'], 0)}° apart (median), and the true "
        f"vessel length was {_fmt(s['length_mm'], 0)} mm (median).",
        "",
        "## Length: one view vs two views",
        "",
        "Foreshortening makes a single view read a vessel shorter than it is, and the pixel "
        "size at the isocentre is exact only for vessels at the isocentre's depth: here it was "
        f"{_fmt(single['pixel_abs'])}% off at the vessel (median abs.; 90th pct "
        f"{_fmt(single['pixel_abs'], 1, 'p90')}%), with the heart within "
        f"{cfg.heart_offset_sd_mm:g} mm (SD per axis) of the isocentre.",
        "",
        "| Method | Median abs. error (%) | 90th pct (%) | Median signed (%) | Within 5% |",
        "|---|---:|---:|---:|---:|",
        f"| Single view, detector spacing (no magnification correction) | "
        f"{_fmt(single['length_abs_detector'])} | "
        f"{_fmt(single['length_abs_detector'], 1, 'p90')} | – | – |",
        f"| Single view, isocentre pixel size | {_fmt(single['length_abs'])} | "
        f"{_fmt(single['length_abs'], 1, 'p90')} | {_fmt(single['length_bias'])} | "
        f"{_pct(single['within_5pct'])} |",
        f"| Two views | {_fmt(s['length_abs'])} | {_fmt(s['length_abs'], 1, 'p90')} | "
        f"{_fmt(s['length_bias'])} | {_pct(s['within_5pct'])} |",
        "",
        f"{len(s['large'])} of {len(ok)} cases were off by more than 5%; the check flagged "
        f"{sum(c['flagged'] for c in s['large'])} of them" + (":" if s["large"] else "."),
        "",
        *(
            f"- seed {c['seed']}: {100 * c['length_err']:+.1f}% ({c['err3d_mm']:.1f} mm 3D error), "
            f"views {c['separation_deg']:.0f}° apart, single views "
            + ", ".join(f"{100 * x:+.0f}%" for x in c["single_length_err"])
            + ("; flagged" if c["flagged"] else "; not flagged")
            for c in sorted(s["large"], key=lambda c: -abs(c["length_err"]))
        ),
        *([""] if s["large"] else []),
        "## FFR error from the geometry alone",
        "",
        "The true lumen seen through each method's geometry with a perfect measurement in "
        "pixels: the length is the method's, and each radius is scaled by the pixel size the "
        "method uses over the true pixel size at that point. This isolates what the geometry "
        "costs.",
        "",
        "| Method | abs. FFR error, median | 90th pct | max |",
        "|---|---:|---:|---:|",
        f"| Single view, detector spacing | {_fmt(single['ffr_geom_abs_detector'], 3)} | "
        f"{_fmt(single['ffr_geom_abs_detector'], 3, 'p90')} | "
        f"{_fmt(single['ffr_geom_abs_detector'], 3, 'max')} |",
        f"| Single view, isocentre pixel size | {_fmt(single['ffr_geom_abs'], 3)} | "
        f"{_fmt(single['ffr_geom_abs'], 3, 'p90')} | {_fmt(single['ffr_geom_abs'], 3, 'max')} |",
        f"| Two views | {_fmt(s['ffr_geom_abs'], 3)} | {_fmt(s['ffr_geom_abs'], 3, 'p90')} | "
        f"{_fmt(s['ffr_geom_abs'], 3, 'max')} |",
        "",
        "![Single view vs two views](figures/two_view_errors.png)",
        "",
        "## Diameter and FFR end to end",
        "",
        "With the product's lumen measurement (intensity edges) in each view.",
        "",
        "| Method | Reference diameter, median abs. error (%) | Lumen diameter MAE, median "
        "(mm) | abs. FFR error, median | abs. FFR error, 90th pct |",
        "|---|---:|---:|---:|---:|",
        f"| Single view, detector spacing | {_fmt(single['ref_abs_detector'])} | – | "
        f"{_fmt(single['ffr_abs_detector'], 3)} | "
        f"{_fmt(single['ffr_abs_detector'], 3, 'p90')} |",
        f"| Single view, isocentre pixel size | {_fmt(single['ref_abs'])} | "
        f"{_fmt(single['d_mae_mm'], 2)} | {_fmt(single['ffr_abs'], 3)} | "
        f"{_fmt(single['ffr_abs'], 3, 'p90')} |",
        f"| Two views | {_fmt(s['ref_abs'])} | {_fmt(s['d_mae_mm'], 2)} | "
        f"{_fmt(s['ffr_abs'], 3)} | {_fmt(s['ffr_abs'], 3, 'p90')} |",
        "",
        f"Here the lumen measurement, which both methods share, dominates: intensity edges "
        f"read the reference diameter {_fmt(s['ref_bias'])}% off (two views, median signed) on "
        "these renders, because the brightness changes fastest inside the blurred, "
        "edge-enhanced vessel wall, and tight lesions read milder than they are (P4c; the "
        "imaging-chain model lumen resolves them better). Same solver and physics throughout; "
        "the true lumen is not trimmed at its ends, the QCA profiles are (one diameter at each "
        "end, where the synthetic vessels have no lesion).",
        "",
        "![First case](figures/two_view_example.png)",
        "",
        "## Motion between the runs",
        "",
        f"The same cases with the heart shifted between the two runs (median "
        f"{_fmt(s['motion_mm'])} mm; SD {cfg.motion_sd_mm:g} mm per axis). A shift across the "
        "epipolar lines is estimated from the vessel itself. A shift along them cannot be "
        "seen in two views: it moves the reconstruction with little change of shape (shape "
        "error: after removing the best translation).",
        "",
        "| Reconstruction | Median abs. length error (%) | 90th pct (%) | 3D error, median "
        "(mm) | Shape error, median (mm) | abs. FFR error, median | Flagged by the check |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| No motion, no correction | {_fmt(nc['length_abs'])} | "
        f"{_fmt(nc['length_abs'], 1, 'p90')} | {_fmt(nc['err3d_mm'], 2)} | – | – | – |",
        f"| No motion, with correction (default) | {_fmt(s['length_abs'])} | "
        f"{_fmt(s['length_abs'], 1, 'p90')} | {_fmt(s['err3d_mm'], 2)} | – | "
        f"{_fmt(s['ffr_abs'], 3)} | {s['warned']} of {len(ok)} |",
        *(
            f"| Motion, {label} | {_fmt(m['length_abs'])} | {_fmt(m['length_abs'], 1, 'p90')} | "
            f"{_fmt(m['err3d_mm'], 2)} | {_fmt(m['shape_mm'], 2)} | {_fmt(m['ffr_abs'], 3)} | "
            f"{m['warned']} of {len(ok)} |"
            for label, m in (
                ("no correction", motion["motion_no_correction"]),
                ("with correction (default)", motion["motion"]),
            )
        ),
        "",
        "## Stress test: random views, overlap allowed",
        "",
        f"{stress['n']} further cases with random views (not chosen to avoid overlap); in "
        f"{stress['n_overlap']} of them the analysed vessel overlaps itself or a side branch in "
        f"a view. The analysis failed in {stress['n_failed']}. Where vessels cross, the "
        "shortest path along the segmentation can follow the wrong vessel; when the wrong path "
        "still fits the other view, the check does not notice it. The analyst must choose "
        "views in which the segment does not overlap, or correct the path.",
        "",
        "| Views | Cases | Two views: median abs. length error (%) | 90th pct (%) "
        "| 3D error, median (mm) | 3D error, 90th pct (mm) | Single view: median abs. length "
        "error (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {label} | {stress[g]['n']} | {_fmt(stress[g]['length_abs'])} | "
            f"{_fmt(stress[g]['length_abs'], 1, 'p90')} | {_fmt(stress[g]['err3d_mm'], 2)} | "
            f"{_fmt(stress[g]['err3d_mm'], 2, 'p90')} | {_fmt(stress[g]['single_length_abs'])} |"
            for g, label in (("clear", "No overlap"), ("overlap", "Overlap"))
        ),
        "",
    ]
    if s["failures"]:
        lines += [
            "Failed static cases:",
            "",
            *(f"- seed {f['seed']}: {f['reason']}" for f in s["failures"]),
            "",
        ]
    lines += [
        "<details><summary>Per case (static)</summary>",
        "",
        "| Seed | Views | Apart (°) | Length (mm) | Two views: length error (%) | 3D error (mm) "
        "| True FFR | Two views FFR | Single view FFR (view 1, 2) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        *(
            f"| {r['seed']} | {' / '.join(r['views'])} | {r['separation_deg']:.0f} | "
            f"{r['length_mm']:.1f} | {100 * r['two_view']['length_err']:+.1f} | "
            f"{r['two_view']['err3d_mm']:.2f} | {r['true_ffr']:.3f} | "
            f"{r['two_view']['ffr']:.3f} | {single_ffr(r)} |"
            for r in ok
        ),
        "",
        "</details>",
        "",
        f"Config `{Path(config).as_posix()}` (seed {cfg.seed}; stress seeds from "
        f"{cfg.seed + STRESS_SEED_OFFSET}). Commit `{provenance['git_commit']}` (dirty: "
        f"{provenance['git_dirty']}), created {provenance['created_utc']}.",
        "",
    ]
    (report_dir / f"{REPORT}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report
