"""App logic without Streamlit: inputs, FFR before / after virtual stents, figures, report.

Everything here is plain Python so it can be unit-tested and timed; ``streamlit_app.py`` only
wires widgets to these functions. Units: SI internally, mm / mmHg for display.
"""

import hashlib
import io
import json
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
from vstent.imaging.qca import QCAResult
from vstent.imaging.segment import THRESHOLD
from vstent.inference.models import SEG_MODELS, SURROGATE_MODELS, latest_model_dir
from vstent.inference.onnx_runtime import (
    ONNX_FILE,
    SEG_ONNX_FILE,
    SegmentationRunner,
    SurrogateRunner,
)
from vstent.physics.geometry import VesselProfile, cosine_narrowing
from vstent.physics.solver import PhysicsParams, solve_ffr_batch
from vstent.planning.optimizer import PlannerConfig, PlanResult, StentCatalog
from vstent.planning.stent import DEFAULT_EDGE, Stent, apply_stents
from vstent.plotting import BAND, INK_MUTED, INK_SECONDARY, SERIES_1, SERIES_2
from vstent.units import m_to_mm, pa_to_mmhg

CUTOFF = 0.80
BANNER = (
    f"{DISCLAIMER} This research prototype is not a medical device, is not validated, and "
    "must never be used to make decisions about real patients."
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


@dataclass
class ImageInput:
    """A greyscale angiogram and what is known about its pixel size."""

    image: np.ndarray  # (H, W) uint8
    name: str
    sha256: str
    pixel_spacing_mm: float | None = None  # from DICOM ImagerPixelSpacing, if present
    note: str = ""


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float64)
    lo, hi = np.percentile(arr, (0.5, 99.5))
    return np.clip((arr - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)


def read_image(data: bytes, name: str) -> ImageInput:
    """PNG / JPG / DICOM bytes → greyscale uint8 image (DICOM: first frame)."""
    digest = hashlib.sha256(data).hexdigest()
    is_dicom = name.lower().endswith((".dcm", ".dicom")) or data[128:132] == b"DICM"
    if is_dicom:
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        arr = ds.pixel_array
        frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        if frames > 1:
            arr = arr[0]
        if arr.ndim == 3:  # colour
            arr = cv2.cvtColor(_to_uint8(arr), cv2.COLOR_RGB2GRAY)
        spacing = ds.get("ImagerPixelSpacing")
        mm = float(spacing[0]) if spacing else None
        note = (
            "DICOM ImagerPixelSpacing is measured at the detector; geometric magnification is "
            "not corrected."
            if mm
            else "DICOM without ImagerPixelSpacing."
        )
        if frames > 1:
            note += f" First of {frames} frames used."
        return ImageInput(_to_uint8(arr), name, digest, mm, note)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"{name}: not a readable PNG / JPG / DICOM image")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return ImageInput(_to_uint8(img), name, digest)


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
    """FFR pullbacks of one geometry. Positions in m; surrogate on its own 256-point grid."""

    s: np.ndarray
    solver: np.ndarray
    s_surrogate: np.ndarray | None
    surrogate: np.ndarray | None
    r_micro: float
    q: float

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
    s_sur = ffr_sur = None
    if runner is not None:
        s_sur, ffr_sur = runner.predict_profile(profile, pa, pv, rm)
    return Curves(profile.s, res.ffr[0], s_sur, ffr_sur, rm, float(res.q[0]))


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


def ffr_figure(pre: Curves, post: Curves | None = None) -> go.Figure:
    """FFR pullbacks (solver solid, surrogate dashed; post-PCI orange) + surrogate − solver."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06
    )
    for curves, color, label in ((pre, INK_SECONDARY, "pre-PCI"), (post, SERIES_2, "post-PCI")):
        if curves is None:
            continue
        s = m_to_mm(curves.s)
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
            f"{m_to_mm(DEFAULT_EDGE):g} mm inside the stent; no recoil or malapposition",
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
                }
                for pl in pr.plans
            ],
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
