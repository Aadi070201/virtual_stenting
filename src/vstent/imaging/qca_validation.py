"""QCA validation on rendered synthetic tubes (``vstent qca-synthetic``).

Five curved tubes with blur, noise and a 60% cosine stenosis (known centreline and diameter)
are measured with :func:`vstent.imaging.qca.run_qca` from deliberately imprecise start / end
clicks. Writes ``reports/qca_synthetic.{md,json}``; ``tests/test_qca.py`` checks the same cases
against the plan's targets. All quantities in pixels / percent.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from vstent import DISCLAIMER, __version__
from vstent.config import git_info
from vstent.imaging.qca import QCAResult, run_qca
from vstent.imaging.synthetic import (
    SyntheticTube,
    render_tube,
    sine_centerline,
    stenosis_diameter,
)

SHAPE = (400, 400)
TRUE_DS_PCT = 60.0
TARGET_MEDIAN_ERROR_PX = 0.7
TARGET_MLD_LOCATION_PX = 3.0
TARGET_DS_ERROR_PCT = 7.0
CLICK_OFFSETS = ((1.5, -2.0), (-2.0, 1.0))  # start / end click errors (row, col px)
# (angle °, sine amplitude px, reference diameter px, blur sigma px, noise std)
CASES = (
    (0.0, 0.0, 12.0, 1.0, 0.03),
    (20.0, 25.0, 12.0, 1.0, 0.03),
    (-35.0, 30.0, 10.0, 1.2, 0.05),
    (60.0, 20.0, 14.0, 1.0, 0.03),
    (100.0, 35.0, 8.0, 1.0, 0.03),
)


def render_case(
    angle: float,
    amplitude: float,
    ref: float,
    blur: float,
    noise: float,
    *,
    seed: int = 0,
    severity: float = TRUE_DS_PCT / 100.0,
    length: float = 280.0,
) -> SyntheticTube:
    """A curved tube centred in a 400 × 400 image with one cosine stenosis (40 px long)."""
    pts, s = sine_centerline((0.0, 0.0), length, angle_deg=angle, amplitude_px=amplitude)
    pts = pts - (pts.min(axis=0) + pts.max(axis=0)) / 2 + np.array(SHAPE) / 2
    d = stenosis_diameter(s, ref, severity, s[-1] / 2, 40.0)
    return render_tube(SHAPE, pts, s, d, blur_sigma=blur, noise_std=noise, seed=seed)


def measure_case(tube: SyntheticTube) -> tuple[QCAResult, dict[str, float]]:
    """QCA of a rendered tube from imprecise clicks, and its errors against the truth."""
    p, d = tube.points, tube.diameter
    start = p[0] + np.array(CLICK_OFFSETS[0])
    end = p[-1] + np.array(CLICK_OFFSETS[1])
    res = run_qca(tube.image, start, end)
    _, idx = cKDTree(p).query(res.centerline.points)
    err = res.diameters.diameter_px - d[idx]
    throat = p[np.argmin(np.abs(tube.s - tube.s[-1] / 2))]
    metrics = {
        "median_abs_diameter_error_px": float(np.median(np.abs(err))),
        "p95_abs_diameter_error_px": float(np.percentile(np.abs(err), 95)),
        "median_diameter_bias_px": float(np.median(err)),
        "mld_location_error_px": float(np.linalg.norm(res.mld_point_px - throat)),
        "mld_px": float(res.diameters.diameter_px.min()),
        "true_mld_px": float(d.min()),
        "max_ds_pct": res.max_percent_stenosis,
        "ds_error_pct": res.max_percent_stenosis - TRUE_DS_PCT,
        "fallback_fraction": res.fallback_fraction,
        "traced_length_fraction": res.centerline.length / float(tube.s[-1]),
    }
    metrics["passed"] = bool(
        metrics["median_abs_diameter_error_px"] <= TARGET_MEDIAN_ERROR_PX
        and metrics["mld_location_error_px"] <= TARGET_MLD_LOCATION_PX
        and abs(metrics["ds_error_pct"]) <= TARGET_DS_ERROR_PCT
    )
    return res, metrics


def write_qca_validation(report_dir: str | Path = "reports") -> dict[str, Any]:
    """Measure every case and write the report."""
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    rows = []
    for k, case in enumerate(CASES):
        _, metrics = measure_case(render_case(*case, seed=k))
        rows.append(
            {
                "case": dict(
                    zip(
                        (
                            "angle_deg",
                            "amplitude_px",
                            "ref_diameter_px",
                            "blur_sigma_px",
                            "noise_std",
                        ),
                        case,
                        strict=True,
                    )
                ),
                **metrics,
            }
        )
    report = {
        "targets": [
            {
                "metric": "Median abs. diameter error (worst case)",
                "value": f"{max(r['median_abs_diameter_error_px'] for r in rows):.2f} px",
                "target": f"≤ {TARGET_MEDIAN_ERROR_PX} px",
                "passed": all(
                    r["median_abs_diameter_error_px"] <= TARGET_MEDIAN_ERROR_PX for r in rows
                ),
            },
            {
                "metric": "MLD location error (worst case)",
                "value": f"{max(r['mld_location_error_px'] for r in rows):.1f} px",
                "target": f"≤ {TARGET_MLD_LOCATION_PX:g} px",
                "passed": all(r["mld_location_error_px"] <= TARGET_MLD_LOCATION_PX for r in rows),
            },
            {
                "metric": "DS error (worst case)",
                "value": f"{max(abs(r['ds_error_pct']) for r in rows):.1f} points",
                "target": f"within ±{TARGET_DS_ERROR_PCT:g} points",
                "passed": all(abs(r["ds_error_pct"]) <= TARGET_DS_ERROR_PCT for r in rows),
            },
        ],
        "rows": rows,
        "provenance": provenance,
    }
    report["targets_met"] = all(t["passed"] for t in report["targets"])
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "qca_synthetic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    lines = [
        "# QCA validation on synthetic tubes",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent qca-synthetic`. Do not edit by hand.",
        "",
        f"{len(CASES)} curved tubes rendered at {SHAPE[0]} × {SHAPE[1]} px with Gaussian blur and "
        f"noise, each with one cosine stenosis of {TRUE_DS_PCT:g}% DS; measured from start / end "
        f"clicks offset by {CLICK_OFFSETS[0]} and {CLICK_OFFSETS[1]} px. Diameter errors compare "
        "every measured centreline point with the true diameter at the nearest true centreline "
        "point.",
        "",
        "| Metric | Value | Target | Pass |",
        "|---|---|---|---|",
    ]
    for t in report["targets"]:
        lines.append(
            f"| {t['metric']} | {t['value']} | {t['target']} | {'yes' if t['passed'] else 'NO'} |"
        )
    lines += [
        "",
        "| Angle | Amplitude | Ref. diameter | Blur σ | Noise | Median abs. error | p95 | Bias "
        "| MLD (true) | MLD location error | Max DS | Fallback |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["case"]
        lines.append(
            f"| {c['angle_deg']:g}° | {c['amplitude_px']:g} px | {c['ref_diameter_px']:g} px "
            f"| {c['blur_sigma_px']:g} px | {c['noise_std']:g} "
            f"| {r['median_abs_diameter_error_px']:.2f} px "
            f"| {r['p95_abs_diameter_error_px']:.2f} px "
            f"| {r['median_diameter_bias_px']:+.2f} px "
            f"| {r['mld_px']:.2f} ({r['true_mld_px']:.2f}) px "
            f"| {r['mld_location_error_px']:.1f} px | {r['max_ds_pct']:.1f}% "
            f"| {r['fallback_fraction']:.0%} |"
        )
    lines += [
        "",
        "Synthetic tubes are an idealised check of the measurement chain (skeleton, path, "
        "normals, sub-pixel edges, reference fit); real angiograms add overlap, foreshortening, "
        "branches and segmentation errors (see `reports/qca_gallery.md`).",
        "",
        f"Commit `{provenance['git_commit']}` (dirty: {provenance['git_dirty']}), vstent "
        f"{provenance['vstent_version']}, {provenance['created_utc']}.",
        "",
    ]
    (report_dir / "qca_synthetic.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report
