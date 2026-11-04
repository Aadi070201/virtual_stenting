"""App logic without Streamlit: inputs, FFR before / after virtual stents, figures, report.

Everything here is plain Python so it can be unit-tested and timed; ``streamlit_app.py`` only
wires widgets to these functions. Units: SI internally, mm / mmHg for display.
"""

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vstent import DISCLAIMER, __version__
from vstent.config import git_info, load_config
from vstent.data.synth import SamplingConfig, sample_vessel
from vstent.imaging.image_io import ImageInput
from vstent.imaging.projection import ProjectionGeometry
from vstent.imaging.qca import LUMEN_METHOD_LABELS, QCAResult
from vstent.imaging.segment import THRESHOLD
from vstent.inference.models import SEG_MODELS, SURROGATE_MODELS, latest_model_dir
from vstent.inference.onnx_runtime import (
    ONNX_FILE,
    SEG_ONNX_FILE,
    SegmentationRunner,
    SurrogateRunner,
)
from vstent.inference.pipeline import TwoViewAnalysis
from vstent.physics.geometry import VesselProfile, cosine_narrowing
from vstent.physics.solver import PhysicsParams, solve_ffr_batch
from vstent.planning.optimizer import PlannerConfig, PlanResult, StentCatalog
from vstent.planning.stent import DEFAULT_EDGE, Stent, apply_stents
from vstent.plotting import BAND, INK_MUTED, INK_SECONDARY, SERIES_1, SERIES_2
from vstent.uncertainty.greyzone import (
    Assessment,
    Interval,
    UncertaintyConfig,
    assessment,
    model_interval,
)
from vstent.uncertainty.inputs import SOURCE_LABELS, InputUncertainty, Measurement, QCAErrorModel
from vstent.units import m_to_mm, pa_to_mmhg

CUTOFF = 0.80
BANNER = (
    f"{DISCLAIMER} This research prototype is not a medical device, is not validated, and "
    "must never be used to make decisions about real patients."
)
UNCERTAINTY_NOTE = (
    "Model intervals cover only the surrogate's error against the reduced-order solver "
    "(split-conformal, calibrated on synthetic vessels). Input intervals cover only the listed "
    "assumptions. Neither covers the error of the reduced-order model itself."
)
# Environment overrides (tests, demos): model directories and the ARCADE folder.
ENV_SURROGATE = "VSTENT_SURROGATE_MODEL"
ENV_SEG = "VSTENT_SEG_MODEL"
ENV_ARCADE = "VSTENT_ARCADE_DIR"
PRESETS = ("Focal lesion (70% DS)", "Tandem lesions (2 × 60% DS)", "Random synthetic vessel")


# --- models ----------------------------------------------------------------------------------


@dataclass
class Models:
    """Loaded ONNX runners and where they came from (for the report)."""

    surrogate: SurrogateRunner | None
    segmentation: SegmentationRunner | None
    surrogate_dir: Path | None
    seg_dir: Path | None

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, d in (("surrogate", self.surrogate_dir), ("segmentation", self.seg_dir)):
            meta_path = d / "export_meta.json" if d else None
            meta = (
                json.loads(meta_path.read_text("utf-8")) if meta_path and meta_path.exists() else {}
            )
            out[name] = {
                "dir": d.as_posix() if d else None,
                "run": meta.get("run"),
                "run_git_commit": meta.get("run_git_commit"),
                "onnx_sha256": meta.get("onnx_sha256"),
            }
        return out


def load_models() -> Models:
    """Newest exported models (or the directories named by the environment overrides)."""
    sur = os.environ.get(ENV_SURROGATE) or latest_model_dir(SURROGATE_MODELS, ONNX_FILE)
    seg = os.environ.get(ENV_SEG) or latest_model_dir(SEG_MODELS, SEG_ONNX_FILE)
    sur, seg = (Path(sur) if sur else None), (Path(seg) if seg else None)
    return Models(
        SurrogateRunner.from_dir(sur) if sur else None,
        SegmentationRunner.from_dir(seg) if seg else None,
        sur,
        seg,
    )


def cine_caption(n_frames: int, rf) -> str:
    """What the app says about a cine run (``rf``:
    :class:`~vstent.imaging.run_analysis.RunFrames`)."""
    op = rf.opacification
    first, last = op.window
    return (
        f"Cine run of {n_frames} frames: contrast arrival at frame {op.arrival}, opacified frames "
        f"{first}–{last}. {rf.ed.note} Frame {rf.analysis_frame} (0-based) is analysed: "
        f"{rf.frame_note} The first frames of a run are usually recorded before the contrast "
        "arrives."
    )


def flow_caption(fv, tr=None) -> str:
    """The measured flow velocity (``fv``: :class:`~vstent.imaging.flow.FlowVelocity`; ``tr``:
    its transit, for the speed in px per frame when it cannot be converted to m/s)."""
    text = fv.note()
    if tr is not None and tr.plausible and math.isnan(fv.cfv):
        lo, hi = tr.interval
        text = f"Contrast front {tr.px_per_frame:.1f} px/frame (95%: {lo:.1f}–{hi:.1f}). {text}"
    return text + (" The fixed velocity stays the default; a measured one is not yet validated "
                   "against real flow.")  # fmt: skip


def flow_dict(fv, used: bool, ed=None) -> dict[str, Any]:
    """The case report's ``flow`` section (P7)."""
    out = {
        "contrast_flow_velocity_m_s": None if math.isnan(fv.cfv) else round(fv.cfv, 4),
        "interval_m_s": None if math.isnan(fv.cfv) else [round(x, 4) for x in fv.interval],
        "runs": fv.n_runs,
        "single_view": fv.single_view,
        "hyperemic_run": fv.hyperemic_run,
        "conversion": "none (hyperemic run)" if fv.hyperemic_run else "FAVOR Pilot (resting run)",
        "accepted": fv.accepted,
        "reasons": list(fv.reasons),
        "used_for_ffr": used,
    }
    if fv.accepted:
        out["hyperemic_velocity_m_s"] = round(fv.v_hyp, 4)
        out["hyperemic_interval_m_s"] = [round(x, 4) for x in fv.v_hyp_interval]
    if ed is not None:
        out["end_diastole"] = {"source": ed.source, "frames": list(ed.frames)}
    return out


def load_settings(root: Path = Path(".")):
    """Physics, planner and catalogue configs from ``configs/`` (defaults if missing)."""

    def cfg(name, model):
        path = root / "configs" / name
        return load_config(path, model) if path.exists() else model()

    physics = cfg("physics.yaml", PhysicsParams)
    planner = cfg("planner.yaml", PlannerConfig)
    path = root / "configs" / "stent_catalog.yaml"
    catalog = (
        load_config(path, StentCatalog)
        if path.exists()
        else StentCatalog(
            diameters_mm=[2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0],
            lengths_mm=[8, 12, 15, 18, 22, 26, 30, 34, 38],
        )
    )
    return physics, planner, catalog


# --- inputs ----------------------------------------------------------------------------------


def arcade_examples(root: str | Path | None = None, n: int = 30) -> list[Path]:
    """The first ``n`` ARCADE ``stenosis/val`` images, if the dataset is present."""
    root = Path(root or os.environ.get(ENV_ARCADE) or "data/raw/arcade")
    folder = root / "arcade" / "stenosis" / "val" / "images"
    if not folder.exists():
        return []
    return sorted(folder.glob("*.png"), key=lambda p: (len(p.stem), p.stem))[:n]


@dataclass(frozen=True, eq=False)
class SyntheticCase:
    """A synthetic vessel (SI) with its pressures (Pa) and hyperemic velocity (m/s)."""

    profile: VesselProfile
    pa: float
    pv: float
    v_hyp: float
    description: str


def synthetic_case(preset: str, seed: int, params: PhysicsParams) -> SyntheticCase:
    """One of ``PRESETS``; the random preset draws from the synthetic sampling distributions."""
    if preset == PRESETS[0]:
        s = np.linspace(0.0, 0.06, 256)
        r_ref = np.full_like(s, 1.5e-3)
        r = r_ref * cosine_narrowing(s, 0.028, 0.012, 0.7)
        desc = "60 mm, 3.0 mm reference, one 12 mm cosine lesion of 70% DS"
    elif preset == PRESETS[1]:
        s = np.linspace(0.0, 0.07, 256)
        r_ref = np.full_like(s, 1.5e-3)
        r = r_ref * cosine_narrowing(s, 0.02, 0.01, 0.6) * cosine_narrowing(s, 0.05, 0.01, 0.6)
        desc = "70 mm, 3.0 mm reference, two 10 mm lesions of 60% DS at 20 and 50 mm"
    else:
        v = sample_vessel(np.random.default_rng(seed), SamplingConfig())
        lesions = ", ".join(f"{le.ds_pct:.0f}%" for le in v.lesions) or "none"
        desc = (
            f"seed {seed}: {m_to_mm(v.profile.length):.0f} mm, "
            f"D0 {m_to_mm(2 * v.profile.reference_radius()[0]):.2f} mm, lesions DS {lesions}"
        )
        return SyntheticCase(v.profile, v.pa, v.pv, v.v_hyp, desc)
    return SyntheticCase(VesselProfile(s, r, r_ref), params.pa, params.pv, params.v_hyp, desc)


# --- FFR before / after stenting ------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Curves:
    """FFR pullbacks of one geometry. Positions in m; surrogate on its own 256-point grid.

    With an ensemble surrogate, ``surrogate`` is the members' mean, ``surrogate_std`` their SD
    and ``model_interval`` the calibrated interval of distal FFR.
    """

    s: np.ndarray
    solver: np.ndarray
    s_surrogate: np.ndarray | None
    surrogate: np.ndarray | None
    r_micro: float
    q: float
    surrogate_std: np.ndarray | None = None
    model_interval: Interval | None = None

    @property
    def distal_solver(self) -> float:
        return float(self.solver[-1])

    @property
    def distal_surrogate(self) -> float | None:
        return None if self.surrogate is None else float(self.surrogate[-1])


def evaluate(
    profile: VesselProfile,
    params: PhysicsParams,
    pa: float,
    pv: float,
    *,
    r_micro: float | None = None,
    runner: SurrogateRunner | None = None,
) -> Curves:
    """Solver (and surrogate) FFR. ``r_micro=None`` calibrates it (pre-PCI); post-PCI passes
    the pre-PCI value so that only the epicardial geometry changes. Pressures in Pa."""
    res = solve_ffr_batch(
        profile.r, profile.ds, profile.reference_radius(), params, pa=pa, pv=pv, r_micro=r_micro
    )
    rm = float(res.r_micro[0])
    s_sur = ffr_sur = std = interval = None
    if runner is not None:
        s_sur, ffr_sur, std = runner.predict_profile_with_spread(profile, pa, pv, rm)
        interval = model_interval(runner, ffr_sur[-1], None if std is None else std[-1])
    return Curves(profile.s, res.ffr[0], s_sur, ffr_sur, rm, float(res.q[0]), std, interval)


def input_uncertainty(
    profile: VesselProfile,
    params: PhysicsParams,
    pa: float,
    pv: float,
    measurement: Measurement,
    settings: tuple[UncertaintyConfig, QCAErrorModel | None],
    v_hyp_range: tuple[float, float] | None = None,
) -> InputUncertainty:
    """Monte Carlo over the case's input assumptions (run once per case; pressures in Pa);
    ``v_hyp_range``: a measured velocity's interval (m/s, P7)."""
    cfg, qca_error = settings
    return InputUncertainty(
        profile,
        params,
        pa,
        pv,
        measurement=measurement,
        config=cfg.inputs,
        qca_error=qca_error,
        v_hyp_range=v_hyp_range,
    )


def assess(
    curves: Curves, samples: np.ndarray | None, threshold: float, level: float
) -> Assessment:
    """Intervals and grey-zone flag of a distal FFR (``samples``: input Monte Carlo draws)."""
    return assessment(
        threshold, curves.distal_solver, curves.distal_surrogate, curves.model_interval,
        samples, level,
    )  # fmt: skip


def stent_warnings(profile: VesselProfile, stents: list[Stent], max_ratio: float) -> list[str]:
    """Manual stents that the planner's sizing rule would reject (diameter vs distal reference)."""
    ref_d = 2.0 * profile.reference_radius()
    out = []
    for st in stents:
        distal_ref = float(np.interp(st.s_end, profile.s, ref_d))
        if st.diameter > max_ratio * distal_ref:
            out.append(
                f"Stent {m_to_mm(st.diameter):.2f} mm is {st.diameter / distal_ref:.1f}× the "
                f"reference diameter at its distal end ({m_to_mm(distal_ref):.2f} mm); the planner "
                f"allows at most {max_ratio:g}×. An oversized stent creates an abrupt widening "
                "that the model treats as an expansion loss."
            )
    return out


def stent_update(
    profile: VesselProfile,
    stents: list[Stent],
    pre: Curves,
    params: PhysicsParams,
    pa: float,
    pv: float,
    runner: SurrogateRunner | None,
    edge: float = DEFAULT_EDGE,
) -> tuple[VesselProfile, Curves]:
    """Stented profile and its post-PCI curves (the work done on every slider move)."""
    stented = apply_stents(profile, stents, edge=edge)
    return stented, evaluate(stented, params, pa, pv, r_micro=pre.r_micro, runner=runner)


# --- figures ---------------------------------------------------------------------------------

_LAYOUT = {
    "template": "plotly_white",
    "margin": {"l": 50, "r": 10, "t": 30, "b": 40},
    "legend": {"orientation": "h", "y": -0.25},
    "height": 340,
}


def profile_figure(
    profile: VesselProfile,
    stented: VesselProfile | None = None,
    stents: list[Stent] | None = None,
    measured_mm: np.ndarray | None = None,
) -> go.Figure:
    """Diameter (mm) along the vessel with reference, lesions (DS > 30%) and stents."""
    s = m_to_mm(profile.s)
    fig = go.Figure()
    for les in profile.lesions():
        fig.add_vrect(
            x0=m_to_mm(les.s_start),
            x1=m_to_mm(les.s_end),
            fillcolor=BAND,
            line_width=0,
            layer="below",
        )
    for st in stents or []:
        fig.add_vrect(
            x0=m_to_mm(st.s_start),
            x1=m_to_mm(st.s_end),
            fillcolor=SERIES_2,
            opacity=0.12,
            line_width=0,
            layer="below",
        )
    if measured_mm is not None:
        fig.add_scatter(
            x=s, y=measured_mm, name="Measured (raw)", line={"color": "#c3c2b7", "width": 1}
        )
    fig.add_scatter(
        x=s, y=m_to_mm(profile.diameter), name="Lumen", line={"color": SERIES_1, "width": 2}
    )
    fig.add_scatter(
        x=s,
        y=m_to_mm(2 * profile.reference_radius()),
        name="Reference",
        line={"color": INK_MUTED, "width": 1.2, "dash": "dash"},
    )
    if stented is not None:
        fig.add_scatter(
            x=s,
            y=m_to_mm(stented.diameter),
            name="Lumen after stent",
            line={"color": SERIES_2, "width": 2},
        )
    mld_mm, j = profile.mld()
    fig.add_scatter(
        x=[s[j]],
        y=[m_to_mm(mld_mm)],
        mode="markers",
        name="MLD",
        marker={"color": SERIES_2, "symbol": "x", "size": 10},
    )
    fig.update_layout(
        **_LAYOUT,
        xaxis_title="Distance from start (mm)",
        yaxis_title="Diameter (mm)",
        yaxis_rangemode="tozero",
    )
    return fig


def opacification_figure(rf, frame: int) -> go.Figure:
    """A cine run's vessel area per frame (``rf``:
    :class:`~vstent.imaging.run_analysis.RunFrames`), the opacified window, the heartbeat
    frames and the analysed ``frame`` (P7)."""
    op = rf.opacification
    t = np.arange(op.smoothed.size)
    peak = max(float(op.smoothed.max()), 1.0)
    fig = go.Figure()
    fig.add_vrect(x0=op.window[0] - 0.5, x1=op.window[1] + 0.5, fillcolor=BAND, line_width=0,
                  layer="below")  # fmt: skip
    fig.add_scatter(x=t, y=100 * op.smoothed / peak, name="Vessel area (smoothed)",
                    line={"color": SERIES_1, "width": 2})  # fmt: skip
    ed = [f for f in rf.ed.frames if 0 <= f < t.size]
    fig.add_scatter(x=ed, y=[100 * op.smoothed[f] / peak for f in ed], mode="markers",
                    name=f"{rf.ed.name(plural=True).capitalize()} ({rf.ed.source})",
                    marker={"color": SERIES_2, "symbol": "diamond", "size": 9})  # fmt: skip
    fig.add_vline(x=frame, line={"color": INK_MUTED, "width": 1.2, "dash": "dash"})
    fig.update_layout(**{**_LAYOUT, "height": 240}, xaxis_title="Frame (0-based)",
                      yaxis_title="% of peak", yaxis_rangemode="tozero")  # fmt: skip
    return fig


def _interval_bar(fig: go.Figure, x: float, iv: Interval, color: str, name: str) -> None:
    """Vertical bar from iv.low to iv.high at the distal end ``x`` (mm)."""
    fig.add_scatter(
        x=[x, x],
        y=[iv.low, iv.high],
        mode="lines+markers",
        name=name,
        line={"color": color, "width": 3},
        marker={"symbol": "line-ew", "size": 10, "line": {"color": color, "width": 2}},
        row=1,
        col=1,
    )


def ffr_figure(
    pre: Curves,
    post: Curves | None = None,
    assessments: tuple[Assessment | None, Assessment | None] = (None, None),
) -> go.Figure:
    """FFR pullbacks (solver solid, surrogate dashed; post-PCI orange) + surrogate − solver.

    ``assessments`` (pre, post) add their intervals as bars at the distal end: the input
    interval next to the solver's end point, the model interval next to the surrogate's.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06
    )
    for curves, color, label, a in (
        (pre, INK_SECONDARY, "pre-PCI", assessments[0]),
        (post, SERIES_2, "post-PCI", assessments[1]),
    ):
        if curves is None:
            continue
        s = m_to_mm(curves.s)
        if a is not None:
            gap = 0.012 * (s[-1] - s[0])
            if a.inputs is not None:
                name = f"{a.inputs.level:.0%} inputs ({label})"
                _interval_bar(fig, s[-1] + gap, a.inputs, color, name)
            if a.model is not None:
                sur_color = SERIES_1 if curves is pre else SERIES_2
                name = f"{a.model.level:.0%} model ({label})"
                _interval_bar(fig, s[-1] + 2 * gap, a.model, sur_color, name)
        fig.add_scatter(
            x=s,
            y=curves.solver,
            name=f"Solver {label}",
            line={"color": color, "width": 2},
            row=1,
            col=1,
        )
        if curves.surrogate is not None:
            s_sur = m_to_mm(curves.s_surrogate)
            sur_color = SERIES_1 if curves is pre else SERIES_2
            fig.add_scatter(
                x=s_sur,
                y=curves.surrogate,
                name=f"Surrogate {label}",
                line={"color": sur_color, "width": 2, "dash": "dash"},
                row=1,
                col=1,
            )
            diff = curves.surrogate - np.interp(s_sur, s, curves.solver)
            fig.add_scatter(
                x=s_sur,
                y=diff,
                name=f"Surrogate − solver ({label})",
                showlegend=False,
                line={"color": sur_color, "width": 1.5},
                row=2,
                col=1,
            )
    fig.add_hline(y=CUTOFF, line={"color": INK_MUTED, "dash": "dot", "width": 1}, row=1, col=1)
    fig.add_hline(y=0.0, line={"color": INK_MUTED, "width": 0.8}, row=2, col=1)
    fig.update_yaxes(title_text="FFR", row=1, col=1)
    fig.update_yaxes(title_text="Sur. − sol.", row=2, col=1)
    fig.update_xaxes(title_text="Distance from start (mm)", row=2, col=1)
    fig.update_layout(**{**_LAYOUT, "height": 420})
    return fig


def overlay_image(
    image: np.ndarray,
    prob: np.ndarray | None,
    qca: QCAResult | None = None,
    stents: list[Stent] | None = None,
    points: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """RGB image: vessel mask tinted blue, centreline, stents in orange, clicked points."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    if prob is not None:
        mask = prob >= THRESHOLD
        tint = np.array([42, 120, 214], np.float32)  # SERIES_1
        rgb[mask] = 0.65 * rgb[mask] + 0.35 * tint
    out = rgb.astype(np.uint8)
    if qca is not None:
        pts = np.round(qca.centerline.points[:, ::-1]).astype(np.int32)  # (x, y)
        cv2.polylines(out, [pts], False, (255, 214, 10), 1, cv2.LINE_AA)
        s_mm = qca.s_mm
        for st in stents or []:
            sel = (s_mm >= m_to_mm(st.s_start)) & (s_mm <= m_to_mm(st.s_end))
            if sel.sum() >= 2:
                cv2.polylines(out, [pts[sel]], False, (235, 104, 52), 3, cv2.LINE_AA)
    for k, (r, c) in enumerate(points or []):
        cv2.circle(out, (int(c), int(r)), 5, (27, 175, 122) if k == 0 else (235, 104, 52), -1)
    return out


# --- two views (P5c) --------------------------------------------------------------------------

TWO_VIEW_CONFIG = Path("configs/two_view_synthetic.yaml")
TWO_VIEW_SEED = 7001  # the first case of reports/two_view_synthetic.md
GUIDE_DEPTH_M = 0.12  # epipolar guide: the ray from SOD − this to SOD + this
GUIDE = (240, 239, 236)  # light, visible on the grey angiogram
PATH = (255, 214, 10)
STENT = (235, 104, 52)
START = (27, 175, 122)


@dataclass(frozen=True, eq=False)
class TwoViewExample:
    """A synthetic two-view case for the app (not clinical data): both views as image inputs
    with their projection geometry, the rendered vessel masks (used instead of the
    segmentation, which was trained on clinical angiograms) and the true landmarks."""

    views: tuple[ImageInput, ImageInput]
    probs: tuple[np.ndarray, np.ndarray]
    landmarks: tuple[tuple[tuple[int, int], tuple[int, int]], ...]


def two_view_example(seed: int = TWO_VIEW_SEED, config: Path = TWO_VIEW_CONFIG) -> TwoViewExample:
    """Render the synthetic two-view example (``vstent.imaging.synthetic3d``)."""
    from vstent.imaging.synthetic3d import TwoViewSynthConfig, render_view, sample_case

    cfg = load_config(config, TwoViewSynthConfig)
    case = sample_case(seed, cfg)
    views, probs, marks = [], [], []
    for k in range(2):
        v = render_view(case, k, cfg)
        g = v.geometry
        note = (f"Synthetic example (not clinical data): {g.label()}, SID {g.sid_m * 1e3:.0f} mm, "
                f"SOD {g.sod_m * 1e3:.0f} mm, pixel size {g.iso_pixel_m * 1e3:.4f} mm at the "
                "isocentre.")  # fmt: skip
        sha = hashlib.sha256(v.image.tobytes()).hexdigest()
        detector_mm = float(np.mean(g.pixel_m)) * 1e3
        views.append(ImageInput(v.image, f"Synthetic two-view example, view {k + 1}", sha,
                                g.iso_pixel_m * 1e3, note, detector_mm, g))  # fmt: skip
        probs.append(v.prob)
        ends = np.round(g.project(case.curves[0][[0, -1]])).astype(int)
        marks.append(((int(ends[0, 0]), int(ends[0, 1])), (int(ends[1, 0]), int(ends[1, 1]))))
    return TwoViewExample(tuple(views), tuple(probs), tuple(marks))


def epipolar_guide(g1: ProjectionGeometry, point, g2: ProjectionGeometry) -> np.ndarray:
    """(2, 2) end points (row, col) in view 2 of the epipolar line of ``point`` (row, col) of
    view 1: where the same landmark must lie in view 2."""
    ray = g1.rays([point])[0]
    depths = np.array([g1.sod_m - GUIDE_DEPTH_M, g1.sod_m + GUIDE_DEPTH_M])
    return g2.project(g1.source + depths[:, None] * ray)


def two_view_overlay(
    image: np.ndarray,
    prob: np.ndarray | None,
    k: int,
    analysis: TwoViewAnalysis | None = None,
    stents: list[Stent] | None = None,
    points: list[tuple[int, int]] | None = None,
    guides: list[np.ndarray] | None = None,
) -> np.ndarray:
    """View ``k`` (0 or 1) of a two-view case: the vessel mask, the 3D centreline projected
    back into this view (stents along it in orange), epipolar guide lines and the points."""
    out = overlay_image(image, prob)
    for seg in guides or []:
        (r0, c0), (r1, c1) = seg
        cv2.line(out, (int(c0), int(r0)), (int(c1), int(r1)), GUIDE, 1, cv2.LINE_AA)
    if analysis is not None:
        rec = analysis.reconstruction
        pts = np.round(rec.geometries[k].project(rec.points)[:, ::-1]).astype(np.int32)
        cv2.polylines(out, [pts], False, PATH, 1, cv2.LINE_AA)
        s_mm = m_to_mm(rec.s - analysis.lumen.offset_m)  # profile coordinates
        for st in stents or []:
            sel = (s_mm >= m_to_mm(st.s_start)) & (s_mm <= m_to_mm(st.s_end))
            if sel.sum() >= 2:
                cv2.polylines(out, [pts[sel]], False, STENT, 3, cv2.LINE_AA)
    for j, (r, c) in enumerate(points or []):
        cv2.circle(out, (int(c), int(r)), 5, START if j == 0 else STENT, -1)
    return out


def two_view_caption(analysis: TwoViewAnalysis) -> str:
    """One-line summary of a two-view reconstruction for the app."""
    rec = analysis.reconstruction
    single = analysis.single_view_lengths_mm
    return (
        f"Two views: {rec.geometries[0].label()} and {rec.geometries[1].label()}, "
        f"{rec.separation_deg:.0f}° apart. 3D path {m_to_mm(rec.length):.1f} mm (single views "
        f"{single[0]:.1f} and {single[1]:.1f} mm: foreshortened). Mean ray gap "
        f"{rec.gap_m * 1e3:.2f} mm, residual {rec.residual_px.max():.2f} px, view 2 shifted "
        f"{np.hypot(*rec.shift_px):.1f} px (motion between the runs); length change with "
        f"smoother fitting {100 * rec.smoothing_change:+.1f}%."
    )


# --- report ----------------------------------------------------------------------------------


@dataclass
class CaseState:
    """What the report needs to know about the current case."""

    mode: str
    params: PhysicsParams
    pa: float
    pv: float
    profile: VesselProfile
    pre: Curves
    stents: list[Stent] = field(default_factory=list)
    post: Curves | None = None
    qca: QCAResult | None = None
    image: ImageInput | None = None
    points: list[tuple[int, int]] | None = None
    description: str = ""
    warnings: list[str] = field(default_factory=list)
    plans: PlanResult | None = None
    uncertainty: InputUncertainty | None = None
    pre_assessment: Assessment | None = None
    post_assessment: Assessment | None = None
    plan_assessments: list[Assessment] = field(default_factory=list)
    edge: float = DEFAULT_EDGE  # stent edge blending (m), the planner's edge_mm
    two_view: TwoViewAnalysis | None = None  # P5c: the 3D analysis (profile = its lumen)
    image2: ImageInput | None = None
    points2: list[tuple[int, int]] | None = None
    flow: dict[str, Any] | None = None  # P7: the measured flow velocity (flow_dict)
    consensus: dict[str, Any] | None = None  # P4g: Consensus.as_dict() of a cine run


def _stent_dict(st: Stent) -> dict[str, float]:
    return {
        "start_mm": round(m_to_mm(st.s_start), 3),
        "length_mm": round(m_to_mm(st.length), 3),
        "diameter_mm": round(m_to_mm(st.diameter), 3),
        "expansion": st.expansion,
    }


def case_report(state: CaseState, models: Models) -> dict[str, Any]:
    """JSON-serialisable case report with the disclaimer and every assumption.

    No image pixels are included (only the image's SHA-256 and size).
    """
    p = state.profile
    report: dict[str, Any] = {
        "disclaimer": BANNER,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "software": {"vstent": __version__, **git_info()},
        "models": models.describe(),
        "input": {"mode": state.mode, "description": state.description},
        "assumptions": {
            "flow_model": "reduced-order: dP = a·Q + b·Q², Q from Pa − Pv = dP + R_micro·Q; "
            "R_micro calibrated so the reference vessel carries v_hyp at Pa − Pv; post-PCI "
            "keeps the pre-PCI R_micro, Pa and Pv",
            "physics": state.params.model_dump(mode="json"),
            "pa_mmhg": round(pa_to_mmhg(state.pa), 2),
            "pv_mmhg": round(pa_to_mmhg(state.pv), 2),
            "r_micro_pa_s_per_m3": state.pre.r_micro,
            "single_vessel_no_side_branches": True,
            "stent_model": "lumen widened to the stent radius, half-cosine edges of "
            f"{m_to_mm(state.edge):g} mm inside the stent; no recoil or malapposition",
        },
        "vessel": {
            "length_mm": round(m_to_mm(p.length), 2),
            "mld_mm": round(m_to_mm(p.mld()[0]), 3),
            "max_ds_pct": round(float(p.percent_stenosis().max()), 1),
            "lesions": [
                {
                    "start_mm": round(m_to_mm(le.s_start), 2),
                    "end_mm": round(m_to_mm(le.s_end), 2),
                    "mld_mm": round(m_to_mm(le.mld), 3),
                    "max_ds_pct": round(le.max_ds_pct, 1),
                }
                for le in p.lesions()
            ],
        },
        "pre_pci": {
            "ffr_distal_solver": round(state.pre.distal_solver, 4),
            "ffr_distal_surrogate": None
            if state.pre.surrogate is None
            else round(state.pre.distal_surrogate, 4),
        },
        "stents": [_stent_dict(st) for st in state.stents],
        "post_pci": None
        if state.post is None
        else {
            "ffr_distal_solver": round(state.post.distal_solver, 4),
            "ffr_distal_surrogate": None
            if state.post.surrogate is None
            else round(state.post.distal_surrogate, 4),
        },
        "warnings": list(state.warnings),
    }
    if state.image is not None:
        report["input"]["image"] = {
            "name": state.image.name,
            "sha256": state.image.sha256,
            "shape": list(state.image.image.shape),
            "note": state.image.note,
        }
        report["input"]["points_row_col"] = [list(map(int, pt)) for pt in state.points or []]
    if state.qca is not None:
        report["assumptions"]["calibration_mm_per_px"] = state.qca.mm_per_px
        report["assumptions"]["calibration_assumed"] = state.qca.calibration_assumed
        report["assumptions"]["lumen_edges"] = LUMEN_METHOD_LABELS[state.qca.lumen_method]
        report["assumptions"]["lumen_size_factor"] = state.qca.size_factor  # P5b
    if state.two_view is not None:
        tv = state.two_view
        rec = tv.reconstruction
        report["assumptions"]["calibration"] = "projection geometry of both views (P5)"
        report["assumptions"]["lumen_edges"] = LUMEN_METHOD_LABELS[tv.qca[0].lumen_method]
        report["two_view"] = {
            "views": [g.label() for g in rec.geometries],
            "separation_deg": round(rec.separation_deg, 1),
            "length_3d_mm": round(m_to_mm(rec.length), 2),
            "single_view_lengths_mm": [round(x, 2) for x in tv.single_view_lengths_mm],
            "mean_ray_gap_mm": round(rec.gap_m * 1e3, 3),
            "residual_px": [round(float(x), 3) for x in rec.residual_px],
            "view2_shift_px": [round(float(x), 2) for x in rec.shift_px],
            "smoothing_change": round(rec.smoothing_change, 4),
            "size_factors": [round(q.size_factor, 3) for q in tv.qca],
        }
        if state.image2 is not None:
            report["input"]["image2"] = {
                "name": state.image2.name,
                "sha256": state.image2.sha256,
                "shape": list(state.image2.image.shape),
                "note": state.image2.note,
            }
            report["input"]["points2_row_col"] = [list(map(int, p)) for p in state.points2 or []]
    if state.consensus is not None:
        report["consensus"] = state.consensus
    if state.flow is not None:
        report["flow"] = state.flow
        report["assumptions"]["hyperemic_velocity_m_s"] = state.params.v_hyp
    if state.plans is not None:
        pr = state.plans
        report["plans"] = {
            "backend": pr.backend,
            "target_ffr": pr.target,
            "message": pr.message,
            "top": [
                {
                    "stents": [_stent_dict(st) for st in pl.stents],
                    "total_length_mm": round(m_to_mm(pl.total_length), 2),
                    "ffr": round(pl.ffr, 4),
                    "ffr_solver_check": None if pl.ffr_solver is None else round(pl.ffr_solver, 4),
                    "meets_target": pl.meets_target,
                    "uncertainty": a.to_dict() if a is not None else None,
                }
                for pl, a in zip(
                    pr.plans,
                    state.plan_assessments or [None] * len(pr.plans),
                    strict=True,
                )
            ],
        }
    if state.pre_assessment is not None:
        mc = state.uncertainty
        report["uncertainty"] = {
            "note": UNCERTAINTY_NOTE,
            "input_assumptions_varied": [SOURCE_LABELS[s] for s in mc.sources] if mc else [],
            "input_samples": mc.n_samples if mc else 0,
            "pre_pci": state.pre_assessment.to_dict(),
            "post_pci": None if state.post_assessment is None else state.post_assessment.to_dict(),
        }
    return report


def report_json(state: CaseState, models: Models) -> bytes:
    return json.dumps(case_report(state, models), indent=2).encode("utf-8")


def report_png(state: CaseState) -> bytes:
    """PNG with the diameter profile and FFR curves (no image pixels), with the disclaimer."""
    from matplotlib.figure import Figure

    from vstent.plotting import style_axes

    fig = Figure(figsize=(10, 4), dpi=130, facecolor="#fcfcfb")
    fig.text(0.06, 0.94, "vstent case report", fontsize=11)
    fig.text(0.06, 0.02, BANNER, fontsize=6.5, color=INK_MUTED)
    ax_d, ax_f = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18, wspace=0.25)
    p, s = state.profile, m_to_mm(state.profile.s)
    style_axes(ax_d)
    ax_d.plot(s, m_to_mm(p.diameter), color=SERIES_1, lw=1.6, label="Lumen")
    ax_d.plot(
        s, m_to_mm(2 * p.reference_radius()), color=INK_MUTED, ls="--", lw=1, label="Reference"
    )
    for st in state.stents:
        ax_d.axvspan(m_to_mm(st.s_start), m_to_mm(st.s_end), color=SERIES_2, alpha=0.15, lw=0)
    ax_d.set_xlabel("Distance (mm)", fontsize=8)
    ax_d.set_ylabel("Diameter (mm)", fontsize=8)
    ax_d.set_ylim(bottom=0)
    ax_d.legend(frameon=False, fontsize=7)
    style_axes(ax_f)
    for curves, color, label in ((state.pre, INK_SECONDARY, "pre"), (state.post, SERIES_2, "post")):
        if curves is None:
            continue
        ax_f.plot(m_to_mm(curves.s), curves.solver, color=color, lw=1.8, label=f"Solver {label}")
        if curves.surrogate is not None:
            ax_f.plot(
                m_to_mm(curves.s_surrogate),
                curves.surrogate,
                color=color,
                lw=1.4,
                ls="--",
                label=f"Surrogate {label}",
            )
    ax_f.axhline(CUTOFF, color=INK_MUTED, lw=0.8, ls=":")
    ax_f.set_xlabel("Distance (mm)", fontsize=8)
    ax_f.set_ylabel("FFR", fontsize=8)
    ax_f.legend(frameon=False, fontsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#fcfcfb")
    return buf.getvalue()
