"""The guided workflow of the app (no Streamlit): one cine run, step by step.

1. Load a cine run (upload, or a demo run).
2. Choose the frame: the end-diastolic (ED) frames in the opacified window, the recommended one
   first (:func:`frame_choices`).
3. Vessel and lesions: an automatic path (:func:`auto_path`) the user can change; lesions
   numbered on the image (:func:`lesion_marks`).
4. Virtual FFR: the median over the run's ED frames (P4g), in plain words (:func:`verdict`,
   :func:`stability`).
5. Stent: the planner's suggestion, adjustable, before and after.
6. Report.

Research use only. Units: SI inside, mm and mmHg on screen.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go

from vstent.imaging.centerline import main_trunk_endpoints
from vstent.imaging.image_io import RunInput, read_run, read_run_bytes
from vstent.imaging.qca import QCAResult
from vstent.imaging.segment import THRESHOLD
from vstent.inference.consensus import opacified_frames
from vstent.planning.stent import Stent
from vstent.units import m_to_mm

DEMO_SEED = 9500  # a synthetic beating-heart case outside every report's seeds
CADICA_EXAMPLE = Path("data/raw/cadica/CADICA/selectedVideos/p24/v15/input")
DEMOS = {
    "synthetic": "Demo: synthetic beating heart (with ECG)",
    "cadica": "Demo: real angiogram (CADICA patient 24, CC BY 4.0)",
}
CUTOFF = 0.80
AUTO_PATH_MM = 100.0  # an automatic path stops here: coronary arteries are rarely longer in 2D
MARK_MIN_DS = 50.0  # narrowings marked on the image (milder ones are counted, not marked)

# colours (RGB) on the angiogram
VESSEL = np.array([42, 120, 214], np.float32)
PATH = (255, 214, 10)
STENT = (235, 104, 52)
START = (27, 175, 122)
LESION = (220, 38, 38)


def demo_run(kind: str) -> RunInput:
    """A demo cine run: ``synthetic`` (rendered beating heart with ECG and C-arm geometry) or
    ``cadica`` (a real run from CADICA, if the dataset is present; no ECG, no pixel size)."""
    if kind == "cadica":
        if not CADICA_EXAMPLE.exists():
            raise ValueError("CADICA is not downloaded (run `vstent data cadica`)")
        return replace(read_run(CADICA_EXAMPLE), name="CADICA patient 24, run 15")
    from vstent.imaging.synthetic_cine import render_run, run_dicom

    cc, cfg = demo_case()
    data = run_dicom(cc, 0, render_run(cc, 0, probs=False).frames, cfg)
    return read_run_bytes(data, "synthetic beating heart")


def demo_case():
    """The synthetic demo's beating-heart case and its sampling config."""
    from vstent.imaging.synthetic_cine import CineSynthConfig, sample_cine_case

    cfg = CineSynthConfig(fps_choices=(15.0,), ecg_prob=1.0, plateau_beats=(2.5, 3.0))
    return sample_cine_case(DEMO_SEED, cfg), cfg


def run_summary(run: RunInput, rf) -> list[str]:
    """Plain-language facts about the run: frames, heart rhythm source, pixel size."""
    out = [f"{len(run)} frames" + (f" at {1000.0 / run.frame_time_ms:.0f} frames/s"
                                   if run.frame_time_ms else "")]  # fmt: skip
    src = {"R-wave pointer": "heartbeats from the X-ray system's ECG markers",
           "ECG": "heartbeats from the recorded ECG",
           "images": "heartbeats found from the images (no ECG in the file)"}  # fmt: skip
    out.append(src.get(rf.ed.source, "heartbeats not found") if rf.ed.found else
               "heartbeats not found: the most opacified frame is used")  # fmt: skip
    if run.pixel_spacing_mm:
        out.append(f"pixel size {run.pixel_spacing_mm:.3f} mm (from the DICOM geometry)")
    else:
        out.append("no pixel size in the file: 0.30 mm assumed (set it under Settings)")
    return out


def frame_choices(rf, limit: int = 6) -> list[int]:
    """ED frames on which the vessels are fully opacified (the frames the multi-frame median
    can use, :func:`~vstent.inference.consensus.opacified_frames`), in time order, at most
    ``limit`` around the recommended one (:attr:`RunFrames.analysis_frame`), always included."""
    lo, hi = opacified_frames(rf, None)
    frames = [f for f in rf.ed.frames if lo <= f <= hi] if rf.ed.found else []
    rec = rf.analysis_frame
    if rec not in frames:
        frames.append(rec)
    frames = sorted(set(frames))
    while len(frames) > limit:  # drop the frame farthest from the recommended one
        frames.remove(max(frames, key=lambda f: (abs(f - rec), f)))
    return frames


def thumbnail(image: np.ndarray, width: int = 160) -> np.ndarray:
    """A small greyscale copy of a frame."""
    h, w = image.shape
    return cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def auto_path(prob: np.ndarray, mm_per_px: float | None) -> tuple[tuple[int, int], tuple[int, int]]:
    """An automatic path ((row, col) start, end): from the vessel tree's widest end along the
    main trunk, at most :data:`AUTO_PATH_MM` (at the pixel size; 0.30 mm assumed if unknown)."""
    a, b = main_trunk_endpoints(prob >= THRESHOLD, AUTO_PATH_MM / (mm_per_px or 0.30))
    return tuple(map(int, a)), tuple(map(int, b))


@dataclass(frozen=True)
class LesionMark:
    """A lesion on the analysed path, numbered from the start. Position (row, col) px at its
    narrowest point; lengths and diameters in mm."""

    number: int
    row: int
    col: int
    ds_pct: float
    length_mm: float
    mld_mm: float
    at_mm: float
    start_mm: float
    end_mm: float

    @property
    def severity(self) -> str:
        if self.ds_pct >= 70:
            return "severe"
        return "moderate" if self.ds_pct >= 50 else "mild"

    def text(self) -> str:
        return (f"Lesion {self.number}: {self.ds_pct:.0f}% narrowing ({self.severity}), "
                f"{self.length_mm:.0f} mm long, {self.at_mm:.0f} mm from the start; narrowest "
                f"{self.mld_mm:.2f} mm")  # fmt: skip


def lesion_marks(qca: QCAResult, min_ds: float = MARK_MIN_DS) -> list[LesionMark]:
    """The path's lesions (:meth:`VesselProfile.lesions`) with at least ``min_ds`` % diameter
    stenosis, numbered from the start."""
    pts, s_mm = qca.centerline.points, qca.s_mm
    out = []
    lesions = [les for les in qca.profile.lesions() if les.max_ds_pct >= min_ds]
    for k, les in enumerate(lesions, 1):
        at = m_to_mm(les.s_mld)
        i = int(np.argmin(np.abs(s_mm - at)))
        out.append(LesionMark(k, int(round(pts[i, 0])), int(round(pts[i, 1])),
                              float(les.max_ds_pct), m_to_mm(les.length), m_to_mm(les.mld), at,
                              m_to_mm(les.s_start), m_to_mm(les.s_end)))  # fmt: skip
    return out


@dataclass(frozen=True)
class Verdict:
    label: str
    color: str  # a Streamlit badge colour
    sentence: str


def verdict(ffr: float, borderline: bool) -> Verdict:
    """What the virtual FFR means, in plain words. ``borderline``: its uncertainty interval
    contains the 0.80 cut-off (the grey-zone flag)."""
    if borderline:
        return Verdict("Borderline", "orange",
                       f"Close to the {CUTOFF:.2f} cut-off: the true value could lie on either "
                       "side. A pressure-wire measurement would decide.")  # fmt: skip
    if ffr <= CUTOFF:
        return Verdict("Significant", "red",
                       f"Below the {CUTOFF:.2f} cut-off: the narrowing probably limits blood "
                       "flow.")  # fmt: skip
    return Verdict("Not significant", "green",
                   f"Above the {CUTOFF:.2f} cut-off: the narrowing probably does not limit "
                   "blood flow.")  # fmt: skip


def mild_count(qca: QCAResult, min_ds: float = MARK_MIN_DS) -> int:
    """Narrowings of 30% to ``min_ds`` % on the path (not marked)."""
    return sum(les.max_ds_pct < min_ds for les in qca.profile.lesions())


def stability(cons) -> tuple[bool, str]:
    """(stable, sentence) about the heartbeats the FFR was measured on (a P4g consensus)."""
    used = cons.used
    left = [f for f in cons.frames if f.error]
    note = ""
    if left:
        s = "s" if len(left) > 1 else ""
        note = (f" {len(left)} other heartbeat{s} could not be measured (frame {left[0].frame}: "
                f"{left[0].error}).")  # fmt: skip
    if len(used) == 1:
        return True, f"Measured on one heartbeat only ({cons.source}).{note}"
    vals = sorted(f.values.ffr for f in used)
    frames = ", ".join(str(f.frame) for f in used)
    head = f"Measured on {len(used)} heartbeats (frames {frames}): {vals[0]:.2f} to {vals[-1]:.2f}."
    if cons.flagged:
        warn = (" The heartbeats disagree about the 0.80 cut-off: check the frame and the "
                "vessel path before relying on this result.")  # fmt: skip
        return False, head + warn + note
    return True, head + " Consistent." + note


def overlay(
    image: np.ndarray,
    prob: np.ndarray | None,
    qca: QCAResult | None = None,
    lesions: list[LesionMark] | None = None,
    stents: list[Stent] | None = None,
    points: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """RGB angiogram: vessels tinted blue, the path yellow, stents orange, lesions as numbered
    red circles, the start (green) and end (orange) points."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    if prob is not None:
        mask = prob >= THRESHOLD
        rgb[mask] = 0.65 * rgb[mask] + 0.35 * VESSEL
    out = rgb.astype(np.uint8)
    scale = max(1, round(max(image.shape) / 512))
    if qca is not None:
        pts = np.round(qca.centerline.points[:, ::-1]).astype(np.int32)
        cv2.polylines(out, [pts], False, PATH, scale, cv2.LINE_AA)
        for st in stents or []:
            sel = (qca.s_mm >= m_to_mm(st.s_start)) & (qca.s_mm <= m_to_mm(st.s_end))
            if sel.sum() >= 2:
                cv2.polylines(out, [pts[sel]], False, STENT, 4 * scale, cv2.LINE_AA)
    for m in lesions or []:
        cv2.circle(out, (m.col, m.row), 11 * scale, LESION, 2 * scale, cv2.LINE_AA)
        at = (m.col + 12 * scale, m.row - 8 * scale)
        cv2.putText(out, str(m.number), at, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, LESION,
                    2 * scale, cv2.LINE_AA)  # fmt: skip
    for k, (r, c) in enumerate(points or []):
        cv2.circle(out, (int(c), int(r)), 6 * scale, START if k == 0 else STENT, -1, cv2.LINE_AA)
    return out


def stent_text(stents: list[Stent], ffr_after: float | None = None) -> str:
    """'3.00 × 18 mm from 25 to 43 mm' for each stent, with the FFR after."""
    parts = [f"{m_to_mm(s.diameter):.2f} × {m_to_mm(s.length):.0f} mm stent from "
             f"{m_to_mm(s.s_start):.0f} to {m_to_mm(s.s_end):.0f} mm along the path"
             for s in stents]  # fmt: skip
    text = " + ".join(parts)
    return text if ffr_after is None else f"{text}: virtual FFR after stenting {ffr_after:.2f}"


def ffr_chart(pre, post=None, lesions: list[LesionMark] | None = None) -> go.Figure:
    """Virtual FFR along the path (solver), before and after stenting, with the 0.80 cut-off
    and the lesions shaded. ``pre`` / ``post``: :class:`~vstent.app.core.Curves`."""
    fig = go.Figure()
    for m in lesions or []:
        fig.add_vrect(x0=m.start_mm, x1=m.end_mm, fillcolor="rgba(220,38,38,0.10)", line_width=0,
                      annotation_text=str(m.number), annotation_position="top left")  # fmt: skip
    fig.add_hline(y=CUTOFF, line_dash="dash", line_color="#898781",
                  annotation_text="0.80 cut-off", annotation_position="bottom right")  # fmt: skip
    fig.add_scatter(x=m_to_mm(pre.s), y=pre.solver, name="Now", line={"color": "#52514e",
                                                                       "width": 2.5})  # fmt: skip
    if post is not None:
        fig.add_scatter(x=m_to_mm(post.s), y=post.solver, name="After the stent",
                        line={"color": "#eb6834", "width": 2.5})  # fmt: skip
    low = min(float(np.min(pre.solver)), CUTOFF) - 0.05
    fig.update_layout(template="plotly_white", height=300, margin={"l": 50, "r": 10, "t": 20,
                      "b": 40}, legend={"orientation": "h", "y": -0.3},
                      xaxis_title="Distance along the path (mm)", yaxis_title="Virtual FFR",
                      yaxis_range=[max(low, 0.0), 1.02])  # fmt: skip
    return fig
