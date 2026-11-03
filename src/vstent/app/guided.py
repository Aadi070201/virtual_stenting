"""The guided workflow of the app (no Streamlit): one cine run, step by step.

0. A case: from the library (:func:`case_library`: synthetic beating hearts, CADICA's real
   runs if downloaded) or an uploaded run.
1. Frame: scroll the end-diastolic (ED) frames, the recommended one marked
   (:func:`frame_label`).
2. Vessel and lesions: an automatic path (:func:`auto_path`) the user can change; lesions
   numbered on the image (:func:`lesion_marks`) and shown enlarged with the measured edges
   (:func:`lesion_zoom`); the experts' boxes on CADICA's keyframes.
3. Virtual FFR: the median over the run's ED frames (P4g), read against the FFR guide
   (:data:`FFR_GUIDE`, :func:`verdict`).
4. Stent: the planner's suggestion, adjustable, before and after.
5. Report.

Research use only. Units: SI inside, mm and mmHg on screen.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vstent.imaging.centerline import branch_ends, main_trunk_endpoints, target_endpoints
from vstent.imaging.image_io import RunInput, read_run, read_run_bytes
from vstent.imaging.qca import QCAResult
from vstent.imaging.segment import THRESHOLD
from vstent.inference.consensus import opacified_frames
from vstent.planning.stent import Stent
from vstent.units import m_to_mm

DEMO_SEED = 9500  # synthetic library seeds start here, outside every report's seeds
N_SYNTHETIC = 10
CUTOFF = 0.80
GREY_LOW = 0.75  # the grey zone of wire FFR: 0.76–0.80
POST_TARGET = 0.90  # after a stent: a good result (and the planner's default target)
# The guide shown in the app (wire-FFR literature: FAME, ESC 2019 chronic coronary syndromes).
FFR_GUIDE = (
    ("green", "above 0.80", "Not significant: no stent needed"),
    ("orange", "0.76–0.80", "Grey zone: significant, stent usually advised"),
    ("red", "0.75 or below", "Significant: stent advised"),
    ("blue", "0.90 or above after a stent", "Good stent result"),
)
HEX = {"green": "#1baf7a", "orange": "#e08a00", "red": "#dc2626", "blue": "#2a78d6"}
# CADICA runs in the library: severe lesions first; occlusions (no valid FFR) and runs without a
# marked lesion last
LIBRARY_ORDER = ("p90_98", "p99", "p70_90", "p50_70", "p20_50", "p0_20", "p100", None)
AUTO_PATH_MM = 250.0  # a safety cap: the main-trunk path normally ends at the vessel's end
MARK_MIN_DS = 50.0  # narrowings marked red (significant by diameter)
MILD_MIN_DS = 30.0  # narrowings of 30–50% are marked amber (shown, so none is hidden)

# colours (RGB) on the angiogram
VESSEL = np.array([42, 120, 214], np.float32)
PATH = (255, 214, 10)
STENT = (235, 104, 52)
START = (27, 175, 122)
LESION = (220, 38, 38)
MILD = (245, 158, 11)
FAINT = (150, 190, 235)
FAINT_T = 0.15  # display only: vesselness vessels joined to the model's (D62; not measured)
EXPERT = (34, 211, 238)


@dataclass(frozen=True)
class Case:
    """A cine run of the library. ``boxes``: the experts' lesion boxes (CADICA) as
    (frame index, x, y, w, h, grade) in px, (x, y) the top-left corner."""

    key: str
    label: str
    kind: str  # "synthetic" or "cadica"
    path: Path | None = None
    seed: int | None = None
    boxes: tuple[tuple[int, float, float, float, float, str], ...] = ()


def _grade(category: str) -> str:
    from vstent.data.cadica import CATEGORIES

    lo, hi = CATEGORIES[category]
    return f"{lo:g}%" if lo == hi else f"{lo:g}–{hi:g}%"


def case_library(raw_dir: str | Path | None = None, n_synthetic: int = N_SYNTHETIC) -> list[Case]:
    """Synthetic beating hearts (with ECG and C-arm geometry), then CADICA's selected runs if the
    dataset is present (CC BY 4.0; no ECG, no pixel size), the most severe lesions first."""
    from vstent.data import cadica

    cases = [Case(f"synthetic:{DEMO_SEED + k}", f"Synthetic {k + 1} · beating heart with ECG",
                  "synthetic", seed=DEMO_SEED + k) for k in range(n_synthetic)]  # fmt: skip
    raw = Path(raw_dir) if raw_dir is not None else cadica.RAW_DIR
    if not (raw / cadica.ROOT / "selectedVideos").is_dir():
        return cases
    rank = list(cadica.CATEGORIES)  # the worst lesion of a run by this order
    boxes = defaultdict(list)
    for b in cadica.load_lesion_boxes(raw):
        boxes[(b.patient, b.video)].append(b)
    try:
        sides = cadica.load_artery_groups(raw)
    except (OSError, ValueError):
        sides = {}
    real = []
    for r in cadica.load_selected_runs(raw):
        names = sorted(p.name for p in r.frames_dir.iterdir())
        index = {n: k for k, n in enumerate(names)}
        bs = [b for b in boxes.get((r.patient, r.video), []) if b.image.name in index]
        worst = max(bs, key=lambda b: rank.index(b.category)) if bs else None
        side = {"left": "left coronary", "right": "right coronary"}.get(
            sides.get(f"{r.patient}_{r.video}", ""), ""
        )
        if worst is None:
            text = "no lesion marked"
        elif worst.category == "p100":
            text = "total occlusion"
        else:
            text = f"worst lesion {_grade(worst.category)}"
        label = " · ".join(x for x in (f"CADICA {r.patient} run {r.video[1:]}", side, text) if x)
        real.append((LIBRARY_ORDER.index(worst.category if worst else None), int(r.patient[1:]),
                     Case(f"cadica:{r.patient}_{r.video}", label, "cadica", path=r.frames_dir,
                          boxes=tuple((index[b.image.name], b.x, b.y, b.w, b.h,
                                       _grade(b.category)) for b in bs))))  # fmt: skip
    return cases + [c for *_, c in sorted(real, key=lambda x: x[:2])]


def load_case(case: Case) -> RunInput:
    """The case's run: a rendered synthetic beating heart, or CADICA's frames."""
    if case.kind == "cadica":
        return replace(read_run(case.path), name=case.label)
    from vstent.imaging.synthetic_cine import render_run, run_dicom

    cc, cfg = synthetic_case(case.seed)
    data = run_dicom(cc, 0, render_run(cc, 0, probs=False).frames, cfg)
    return read_run_bytes(data, case.label)


def synthetic_case(seed: int = DEMO_SEED):
    """A synthetic library case (beating heart, ECG) and its sampling config."""
    from vstent.imaging.synthetic_cine import CineSynthConfig, sample_cine_case

    cfg = CineSynthConfig(fps_choices=(15.0,), ecg_prob=1.0, plateau_beats=(2.5, 3.0))
    return sample_cine_case(seed, cfg), cfg


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


def frame_label(f: int, rf, all_frames: bool = False, expert: frozenset = frozenset()) -> str:
    """A frame on the scroll bar: its number, ★ for the recommended one, ◆ where experts marked a
    lesion, ED (when every frame is shown) and faint marks."""
    lo, hi = opacified_frames(rf, None)
    tags = ["★"] if f == rf.analysis_frame else []
    if f in expert:
        tags.append("◆")
    if all_frames and rf.ed.found and f in rf.ed.frames:
        tags.append("ED")
    if not lo <= f <= hi:
        tags.append("faint")
    return " ".join([str(f), *tags])


def severity(grade: str) -> float:
    """The lower end of an expert grade such as "70–90%" or "99%" (for ordering boxes)."""
    return float(grade.split("–")[0].rstrip("%"))


def box_path(prob: np.ndarray, box: tuple[float, float, float, float],
             mm_per_px: float | None = None):  # fmt: skip
    """(start, end) of a path through an expert box (x, y, w, h) as a clinician draws it: from
    the vessel tree's proximal end, through the vessel in the box, to that vessel's end
    (:func:`target_path`), so the healthy proximal vessel sets the reference. If the box's
    vessel is not connected to the tree in the segmentation, the vessel path through the box
    extended by the box size on every side
    (:func:`~vstent.validation.stenosis_grading.path_through_box`). ``None`` if no segmented
    vessel crosses the box."""
    from scipy.ndimage import distance_transform_edt

    from vstent.imaging.centerline import vessel_graph
    from vstent.validation.stenosis_grading import path_through_box

    mask = prob >= THRESHOLD
    path = path_through_box(vessel_graph(mask), box)
    if path is None or len(path) < 5:
        return None
    x, y, w, h = box
    inside = [p for p in path if x <= p[1] <= x + w and y <= p[0] <= y + h]
    if inside:
        try:
            return target_path(prob, inside[len(inside) // 2], mm_per_px)
        except ValueError:
            pass  # not connected to the tree: the path through the box alone
    a, b = tuple(map(int, path[0])), tuple(map(int, path[-1]))
    width = distance_transform_edt(mask)
    return (a, b) if width[a] >= width[b] else (b, a)


def thumbnail(image: np.ndarray, width: int = 160) -> np.ndarray:
    """A small greyscale copy of a frame."""
    h, w = image.shape
    return cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def target_path(prob: np.ndarray, point, mm_per_px: float | None):
    """The path to the vessel the user clicked (``point`` (row, col)): from the proximal end
    through the click to the end of that vessel (:func:`~vstent.imaging.centerline.
    target_endpoints`)."""
    a, b = target_endpoints(prob >= THRESHOLD, point, AUTO_PATH_MM / (mm_per_px or 0.30))
    return tuple(map(int, a)), tuple(map(int, b))


def auto_path(prob: np.ndarray, mm_per_px: float | None) -> tuple[tuple[int, int], tuple[int, int]]:
    """An automatic path ((row, col) start, end): from the vessel tree's widest end along the
    main trunk, at most :data:`AUTO_PATH_MM` (at the pixel size; 0.30 mm assumed if unknown)."""
    a, b = main_trunk_endpoints(prob >= THRESHOLD, AUTO_PATH_MM / (mm_per_px or 0.30))
    return tuple(map(int, a)), tuple(map(int, b))


SEEK_MIN_MM = 30.0  # P4k-B: candidate paths at least this long
SEEK_MIN_DS = 50.0  # ... chosen over the main trunk only with a lesion this deep
SEEK_MIN_REF_MM = 2.0  # ... where the reference diameter is at least this (FAVOR II)


def lesion_score(qca: QCAResult, min_ref_mm: float = SEEK_MIN_REF_MM) -> float:
    """The path's deepest lesion (%DS) where the reference diameter is ≥ ``min_ref_mm``
    (0 if none)."""
    s_mm = m_to_mm(qca.profile.s)
    ref_mm = 2e3 * np.broadcast_to(qca.profile.reference_radius(), s_mm.shape)
    deep = [float(les.max_ds_pct) for les in qca.profile.lesions()
            if np.interp(m_to_mm(les.s_mld), s_mm, ref_mm) >= min_ref_mm]  # fmt: skip
    return max(deep, default=0.0)


def seeking_path(prob: np.ndarray, mm_per_px: float | None, measure) -> tuple[tuple, tuple, float]:
    """P4k-B's lesion-seeking path: from the tree's proximal end to every vessel end ≥
    :data:`SEEK_MIN_MM` away, each measured with ``measure(start, end)`` (the product's
    :func:`~vstent.inference.pipeline.analyse_vessel`); the one with the highest
    :func:`lesion_score` if it is ≥ :data:`SEEK_MIN_DS`, else the main trunk
    (:func:`auto_path`). Returns (start, end, score)."""
    a, ends = branch_ends(prob >= THRESHOLD, SEEK_MIN_MM / (mm_per_px or 0.30))
    best, best_score = None, -1.0
    for e in ends:
        try:
            score = lesion_score(measure(a, e).qca)
        except ValueError:
            continue
        if score > best_score:
            best, best_score = e, score
    if best is not None and best_score >= SEEK_MIN_DS:
        return tuple(map(int, a)), tuple(map(int, best)), best_score
    start, end = auto_path(prob, mm_per_px)
    return start, end, max(best_score, 0.0)


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
    def severe(self) -> bool:
        """At least :data:`MARK_MIN_DS` % diameter stenosis (marked red; milder: amber)."""
        return self.ds_pct >= MARK_MIN_DS

    @property
    def severity(self) -> str:
        if self.ds_pct >= 70:
            return "severe"
        return "moderate" if self.ds_pct >= 50 else "mild"

    def text(self) -> str:
        return (f"Lesion {self.number}: {self.ds_pct:.0f}% narrowing ({self.severity}), "
                f"{self.length_mm:.0f} mm long, {self.at_mm:.0f} mm from the start; narrowest "
                f"{self.mld_mm:.2f} mm")  # fmt: skip


def faint_vessels(image: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Vessels the segmentation model left out: Frangi vesselness components joined to its mask
    (:func:`~vstent.imaging.vesselness.added_mask`, :data:`FAINT_T`). Shown, never measured: as
    a measurement change they were not adopted (D62)."""
    from vstent.imaging.vesselness import added_mask, vesselness

    return added_mask(prob, vesselness(image), FAINT_T)


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


def verdict(ffr: float) -> Verdict:
    """The FFR guide's band of a (virtual) FFR value (:data:`FFR_GUIDE`)."""
    ffr = round(float(ffr), 2)  # as shown
    if ffr > CUTOFF:
        return Verdict("Not significant", "green", "No stent needed")
    if ffr > GREY_LOW:
        return Verdict("Grey zone", "orange", "Significant; stent usually advised")
    return Verdict("Significant", "red", "Stent advised")


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
    boxes: list[tuple[float, float, float, float, str]] | None = None,
    contours: bool = True,
    faint: np.ndarray | None = None,
    beyond: QCAResult | None = None,
) -> np.ndarray:
    """RGB angiogram: every segmented vessel tinted and outlined blue; the path as its two
    measured lumen contours (yellow; ``contours=False``: its centreline); lesions filled
    between the contours (red ≥ 50%, amber 30–50%), with ticks at their ends and their
    numbers; stents orange; the start (green) and end (orange) points; experts' boxes
    (x, y, w, h, grade) in cyan. ``faint``: further vessels from the vesselness filter (bool
    (H, W); :func:`faint_vessels`), outlined light blue for display only. ``beyond``: the whole
    measured path when ``qca`` was cut at the wire position (D68); its rest is drawn dashed and
    the point where the FFR is read is marked white."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    scale = max(1, round(max(image.shape) / 512))
    outline = None
    if prob is not None:
        mask = prob >= THRESHOLD
        rgb[mask] = 0.75 * rgb[mask] + 0.25 * VESSEL
        outline, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_NONE)  # fmt: skip
    out = rgb.astype(np.uint8)
    if faint is not None and faint.any():
        extra, _ = cv2.findContours(faint.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, extra, -1, FAINT, 1, cv2.LINE_AA)
    if outline:
        cv2.drawContours(out, outline, -1, tuple(int(v) for v in VESSEL), 1, cv2.LINE_AA)
    for n, (x, y, w, h, grade) in enumerate(boxes or []):
        p0, p1 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(out, p0, p1, EXPERT, scale, cv2.LINE_AA)
        cv2.putText(out, f"{chr(65 + n)} {grade}", (p0[0], max(10, p0[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, EXPERT, scale, cv2.LINE_AA)  # fmt: skip
    if qca is not None:
        pts = np.round(qca.centerline.points[:, ::-1]).astype(np.int32)
        if contours:
            left, right = (np.round(e[:, ::-1]).astype(np.int32) for e in edge_lines(qca))
            fill = out.copy()
            for m in lesions or []:
                sel = (qca.s_mm >= m.start_mm) & (qca.s_mm <= m.end_mm)
                if sel.sum() >= 2:
                    cv2.fillPoly(fill, [np.vstack([left[sel], right[sel][::-1]])],
                                 LESION if m.severe else MILD)  # fmt: skip
                    for k in np.flatnonzero(sel)[[0, -1]]:  # ticks at the lesion's ends
                        cv2.line(out, tuple(map(int, left[k])), tuple(map(int, right[k])),
                                 (255, 255, 255), scale, cv2.LINE_AA)  # fmt: skip
            out = cv2.addWeighted(fill, 0.45, out, 0.55, 0)
            for edge in (left, right):
                cv2.polylines(out, [edge], False, PATH, scale, cv2.LINE_AA)
        else:
            cv2.polylines(out, [pts], False, PATH, scale, cv2.LINE_AA)
        for k, st in enumerate(stents or []):
            sel = (qca.s_mm >= m_to_mm(st.s_start)) & (qca.s_mm <= m_to_mm(st.s_end))
            if sel.sum() >= 2:
                cv2.polylines(out, [pts[sel]], False, STENT, 4 * scale, cv2.LINE_AA)
                if len(stents) > 1:  # S1, S2… beside each stent's mid-point
                    mid = pts[sel][len(pts[sel]) // 2]
                    at = (int(mid[0]) + 8 * scale, int(mid[1]) + 14 * scale)
                    cv2.putText(out, f"S{k + 1}", at, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale,
                                STENT, 2 * scale, cv2.LINE_AA)  # fmt: skip
    if qca is not None and beyond is not None:  # D68: the path beyond the wire position
        rest = np.round(beyond.centerline.points[len(qca.centerline.points) - 1 :, ::-1])
        rest = rest.astype(np.int32)
        for i in range(0, len(rest) - 1, 6):  # dashes
            cv2.polylines(out, [rest[i : i + 4]], False, PATH, scale, cv2.LINE_AA)
        end = tuple(int(v) for v in np.round(qca.centerline.points[-1, ::-1]))
        cv2.circle(out, end, 5 * scale, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(out, "FFR", (end[0] + 7 * scale, end[1] + 4 * scale),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), scale,
                    cv2.LINE_AA)  # fmt: skip
    for m in lesions or []:
        color = LESION if m.severe else MILD
        past = qca is not None and m.at_mm > float(qca.s_mm[-1])  # beyond the FFR point
        if not (contours and qca is not None) or past:
            cv2.circle(out, (m.col, m.row), 11 * scale, color, 2 * scale, cv2.LINE_AA)
        at = (m.col + 12 * scale, m.row - 8 * scale)
        cv2.putText(out, str(m.number), at, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color,
                    2 * scale, cv2.LINE_AA)  # fmt: skip
    for k, (r, c) in enumerate(points or []):
        cv2.circle(out, (int(c), int(r)), 6 * scale, START if k == 0 else STENT, -1, cv2.LINE_AA)
    return out


def stent_text(stents: list[Stent], ffr_after: float | None = None) -> str:
    """'3.00 × 18 mm from 25 to 43 mm' for each stent (numbered S1, S2… when several), with
    the FFR after."""
    parts = [f"{f'S{k + 1}: ' if len(stents) > 1 else ''}{m_to_mm(s.diameter):.2f} × "
             f"{m_to_mm(s.length):.0f} mm stent from {m_to_mm(s.s_start):.0f} to "
             f"{m_to_mm(s.s_end):.0f} mm along the path"
             for k, s in enumerate(stents)]  # fmt: skip
    text = " + ".join(parts)
    return text if ffr_after is None else f"{text}: virtual FFR after stenting {ffr_after:.2f}"


MAX_OWN_STENTS = 3  # stents the user can place by hand


def own_stent_defaults(plan: list[Stent], lesions: list[LesionMark], n: int,
                       length_mm: float) -> list[tuple[float, float, float]]:  # fmt: skip
    """Starting (diameter, length, start) in mm for ``n`` hand-placed stents: the plan's
    stents first, then one 18 mm stent on each marked lesion not yet covered (most severe
    first), then 2 mm past the last stent."""
    out = [(m_to_mm(s.diameter), m_to_mm(s.length), m_to_mm(s.s_start)) for s in plan[:n]]

    def covered(at_mm: float) -> bool:
        return any(s0 <= at_mm <= s0 + ln for _, ln, s0 in out)

    for m in sorted(lesions, key=lambda m: -m.ds_pct):
        if len(out) >= n:
            break
        if not covered(m.at_mm):
            out.append((3.0, 18.0, max(0.0, m.at_mm - 9.0)))
    while len(out) < n:
        after = max((s0 + ln for _, ln, s0 in out), default=-2.0) + 2.0
        out.append((3.0, 18.0, max(0.0, min(after, length_mm - 18.0))))
    return out


def overlap_notes(stents: list[Stent]) -> list[str]:
    """'S1 and S2 overlap by 4 mm' for each pair of stents that overlap."""
    out = []
    for i in range(len(stents)):
        for j in range(i + 1, len(stents)):
            mm = m_to_mm(min(stents[i].s_end, stents[j].s_end)
                         - max(stents[i].s_start, stents[j].s_start))  # fmt: skip
            if mm > 0:
                out.append(f"S{i + 1} and S{j + 1} overlap by {mm:.0f} mm.")
    return out


PPG_WINDOW_MM = 20.0  # Collet et al., JACC 2019: the largest FFR drop over any 20 mm
PPG_FD_PER_MM = 0.0015  # ... length with functional disease: an FFR drop ≥ this per mm
PPG_FOCAL = 0.78  # angiography-derived (QFR) PPG: Shin et al. 2021; PANDA III


def ppg(curves) -> float | None:
    """Pullback pressure gradient index of the virtual FFR along the path (Collet et al.,
    JACC 2019): (largest FFR drop over 20 mm ÷ the path's FFR drop + 1 − share of the path
    with a drop ≥ 0.0015 per mm) ÷ 2. Near 1 focal, near 0 diffuse; ``None`` without a drop.
    ``curves``: :class:`~vstent.app.core.Curves`."""
    s_mm = m_to_mm(np.asarray(curves.s, float))
    f = np.asarray(curves.solver, float)
    total = float(f[0] - f[-1])
    length = float(s_mm[-1] - s_mm[0])
    if total <= 1e-6 or length <= 0:
        return None
    ahead = np.interp(np.minimum(s_mm + PPG_WINDOW_MM, s_mm[-1]), s_mm, f)
    max20 = float(np.max(f - ahead))
    drop = ffr_drop_per_mm(curves)
    ds = np.gradient(s_mm)
    diseased = float(ds[drop >= PPG_FD_PER_MM].sum())
    return float(np.clip(0.5 * (max20 / total + 1.0 - diseased / length), 0.0, 1.0))


def ppg_text(curves) -> str | None:
    """The disease pattern along the path from :func:`ppg` (research; display only)."""
    value = ppg(curves)
    if value is None:
        return None
    kind = "focal" if value >= PPG_FOCAL else "diffuse"
    hint = ("a stent at the lesion should restore most of the FFR" if kind == "focal"
            else "a stent may not restore the FFR fully")  # fmt: skip
    return f"Disease pattern: {kind} (PPG {value:.2f}): {hint}."


def diffuse_note(post, lesions: list[LesionMark]) -> str | None:
    """Why stenting the marked lesions leaves the FFR low: most of the remaining pressure loss
    is spread along the path. ``post``: :class:`~vstent.app.core.Curves` after stenting."""
    if post is None or post.distal_solver > CUTOFF:
        return None
    share = loss_split(post, lesions)
    if share is None or share > 0.5:
        return None
    return (f"After stenting, {100 * (1 - share):.0f}% of the remaining pressure loss is spread "
            "along the path, not at the lesions: a stent cannot fix it. A path that runs into "
            "thin distal branches adds such loss; try ending it earlier (Target).")  # fmt: skip


def ffr_chart(pre, post=None, lesions: list[LesionMark] | None = None) -> go.Figure:
    """Virtual FFR along the path (solver), before and after stenting, with the 0.80 cut-off
    and the lesions shaded. ``pre`` / ``post``: :class:`~vstent.app.core.Curves`."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.06)  # fmt: skip
    for m in lesions or []:
        fig.add_vrect(x0=m.start_mm, x1=m.end_mm, fillcolor="rgba(220,38,38,0.10)", line_width=0,
                      annotation_text=str(m.number), annotation_position="top left",
                      row="all", col=1)  # fmt: skip
    fig.add_hline(y=CUTOFF, line_dash="dash", line_color="#898781", row=1, col=1,
                  annotation_text="0.80 cut-off", annotation_position="bottom right")  # fmt: skip
    series = [("Now", pre, "#52514e")]
    if post is not None:
        series.append(("After the stent", post, "#eb6834"))
    for name, c, color in series:
        fig.add_scatter(x=m_to_mm(c.s), y=c.solver, name=name, legendgroup=name,
                        line={"color": color, "width": 2.5}, row=1, col=1)  # fmt: skip
        fig.add_scatter(x=m_to_mm(c.s), y=ffr_drop_per_mm(c), name=name, legendgroup=name,
                        showlegend=False, line={"color": color, "width": 1.5}, row=2,
                        col=1)  # fmt: skip
    low = min(float(np.min(pre.solver)), CUTOFF) - 0.05
    fig.update_layout(template="plotly_white", height=360, margin={"l": 50, "r": 10, "t": 20,
                      "b": 40}, legend={"orientation": "h", "y": -0.22})  # fmt: skip
    fig.update_yaxes(title_text="Virtual FFR", range=[max(low, 0.0), 1.02], row=1, col=1)
    fig.update_yaxes(title_text="Loss per mm", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title_text="Distance along the path (mm)", row=2, col=1)
    return fig


TUBE_SIDES = 24
TUBE_RINGS = 300  # at most this many rings along the path
# FFR colours of the 3D lumen: red ≤ 0.75, orange at 0.80, green from 0.85 (range 0.60–1.00)
FFR_SCALE = ((0.0, HEX["red"]), (0.375, HEX["red"]), (0.5, HEX["orange"]),
             (0.625, HEX["green"]), (1.0, HEX["green"]))  # fmt: skip


def _transport_frames(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals and binormals along a 3D curve ``p`` (n, 3), carried along without twist
    (parallel transport), so the tube does not wring at bends."""
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n = np.cross(t[0], ref)
    n /= np.linalg.norm(n)
    normals = [n]
    for ti in t[1:]:
        m = n - np.dot(n, ti) * ti
        norm = np.linalg.norm(m)
        n = m / norm if norm > 1e-9 else n
        normals.append(n)
    nn = np.array(normals)
    return nn, np.cross(t, nn)


def tube_mesh(points_mm: np.ndarray, radius_mm: np.ndarray, values: np.ndarray, **kw) -> go.Mesh3d:
    """A closed-side tube of ``radius_mm`` around ``points_mm`` (n, 3), coloured by ``values``
    (one per point)."""
    nn, bb = _transport_frames(points_mm)
    th = np.linspace(0.0, 2.0 * np.pi, TUBE_SIDES, endpoint=False)
    ring = points_mm[:, None, :] + radius_mm[:, None, None] * (
        np.cos(th)[None, :, None] * nn[:, None, :] + np.sin(th)[None, :, None] * bb[:, None, :]
    )
    v = ring.reshape(-1, 3)
    a = np.arange(len(points_mm) - 1)[:, None] * TUBE_SIDES
    b = np.arange(TUBE_SIDES)[None, :]
    v00, v01 = (a + b).ravel(), (a + (b + 1) % TUBE_SIDES).ravel()
    v10, v11 = v00 + TUBE_SIDES, v01 + TUBE_SIDES
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=np.r_[v00, v00], j=np.r_[v10, v11],
                     k=np.r_[v11, v01], intensity=np.repeat(values, TUBE_SIDES),
                     flatshading=False, **kw)  # fmt: skip


def path_mm(qca: QCAResult, s_m: np.ndarray) -> np.ndarray:
    """(n, 3) positions (mm) of the arc lengths ``s_m`` (m) on the analysed path, in the image
    plane: x to the right, y up, z = 0 (one view does not measure depth)."""
    c = qca.centerline
    u_c = c.s / c.s[-1] if c.s[-1] > 0 else np.linspace(0, 1, len(c.s))
    u = np.asarray(s_m, float) / max(float(qca.profile.s[-1]), 1e-12)
    row, col = np.interp(u, u_c, c.points[:, 0]), np.interp(u, u_c, c.points[:, 1])
    return np.c_[col * qca.mm_per_px, -row * qca.mm_per_px, np.zeros_like(row)]


def vessel_3d(
    qca: QCAResult,
    curves,
    lesions: list[LesionMark] | None = None,
    stents: list[Stent] | None = None,
    radius_m: np.ndarray | None = None,
) -> go.Figure:
    """The lumen as a 3D tube along the analysed path, coloured by the virtual FFR of
    ``curves`` (:class:`~vstent.app.core.Curves`; after the stent if ``stents``, with
    ``radius_m`` the stented radius), the lesions marked and numbered, the stents drawn as
    see-through sleeves. One view: the tube lies in the image plane (depth not measured)."""
    s = np.asarray(curves.s, float)
    r_mm = 1e3 * np.asarray(qca.profile.r if radius_m is None else radius_m, float)
    idx = np.unique(np.linspace(0, len(s) - 1, min(len(s), TUBE_RINGS)).round().astype(int))
    p = path_mm(qca, s[idx])
    ffr = np.asarray(curves.solver, float)[idx]
    s_mm = m_to_mm(s[idx])
    fig = go.Figure()
    fig.add_trace(tube_mesh(p, r_mm[idx], ffr, cmin=0.6, cmax=1.0, colorscale=FFR_SCALE,
                            colorbar={"title": "FFR", "len": 0.6, "thickness": 12},
                            hoverinfo="skip", name="Lumen",
                            lighting={"ambient": 0.55, "diffuse": 0.7, "specular": 0.15},
                            ))  # fmt: skip
    fig.add_scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", name="Path",
                      line={"color": "rgba(60,60,60,0.25)", "width": 2}, showlegend=False,
                      customdata=np.c_[s_mm, 2.0 * r_mm[idx], ffr],
                      hovertemplate="%{customdata[0]:.1f} mm · D %{customdata[1]:.2f} mm · "
                      "FFR %{customdata[2]:.2f}<extra></extra>")  # fmt: skip
    for st_ in stents or []:
        lo, hi = m_to_mm(st_.s_start), m_to_mm(st_.s_end)
        k = np.flatnonzero((s_mm >= lo) & (s_mm <= hi))
        if len(k) >= 2:
            sleeve = np.full(len(k), 0.5e3 * st_.diameter * 1.06)
            fig.add_trace(tube_mesh(p[k], sleeve, np.zeros(len(k)), opacity=0.28,
                                    colorscale=((0, HEX["orange"]), (1, HEX["orange"])),
                                    showscale=False, hoverinfo="skip",
                                    name=f"Stent {m_to_mm(st_.diameter):.2f} × "
                                    f"{m_to_mm(st_.length):.0f} mm"))  # fmt: skip
    marks = lesions or []
    if marks:
        at = path_mm(qca, np.array([m.at_mm * 1e-3 for m in marks]))
        lift = np.array([0.0, 0.0, 1.0]) * (1.5 * float(np.max(r_mm)) + 1.0)
        fig.add_scatter3d(x=at[:, 0] + lift[0], y=at[:, 1] + lift[1], z=at[:, 2] + lift[2],
                          mode="markers+text", name="Lesions", showlegend=False,
                          marker={"size": 6, "color": HEX["red"], "symbol": "diamond"},
                          text=[f"{m.number}: {m.ds_pct:.0f}%" for m in marks],
                          textposition="top center", textfont={"color": HEX["red"], "size": 13},
                          hovertext=[f"Lesion {m.number}: {m.ds_pct:.0f}% · MLD {m.mld_mm:.2f} mm"
                                     f" · {m.length_mm:.1f} mm long" for m in marks],
                          hoverinfo="text")  # fmt: skip
    fig.update_layout(
        height=430, margin={"l": 0, "r": 0, "t": 10, "b": 0}, showlegend=False,
        scene={"aspectmode": "data", "xaxis": {"title": "x (mm)", "showbackground": False},
               "yaxis": {"title": "y (mm)", "showbackground": False},
               "zaxis": {"title": "", "showticklabels": False, "showbackground": False},
               "camera": {"eye": {"x": 0.0, "y": -0.7, "z": 1.8},
                          "up": {"x": 0.0, "y": 1.0, "z": 0.0}}},
    )  # fmt: skip
    return fig


def ffr_drop_per_mm(curves) -> np.ndarray:
    """The FFR lost per mm along the path (1/mm), from a :class:`~vstent.app.core.Curves`."""
    return np.maximum(-np.gradient(curves.solver, m_to_mm(curves.s)), 0.0)


def lesion_table(qca: QCAResult, pre, lesions: list[LesionMark]) -> list[dict[str, str]]:
    """One row per marked lesion: %DS, minimal lumen and reference diameters (mm), length (mm)
    and the FFR lost across it. ``pre``: :class:`~vstent.app.core.Curves`."""
    s_mm = m_to_mm(qca.profile.s)
    ref_mm = 2e3 * qca.profile.reference_radius()
    curve_mm = m_to_mm(pre.s)
    rows = []
    for m in lesions:
        drop = float(np.interp(m.start_mm, curve_mm, pre.solver)
                     - np.interp(m.end_mm, curve_mm, pre.solver))  # fmt: skip
        ref = float(np.interp(m.at_mm, s_mm, ref_mm))
        rows.append({"Lesion": str(m.number), "Stenosis": f"{m.ds_pct:.0f}%",
                     "Minimal lumen": f"{m.mld_mm:.2f} mm", "Reference": f"{ref:.2f} mm",
                     "Length": f"{m.length_mm:.1f} mm",
                     "FFR drop": f"{max(drop, 0.0):.2f}"})  # fmt: skip
    return rows


def edge_lines(qca: QCAResult) -> tuple[np.ndarray, np.ndarray]:
    """The measured lumen edges (n, 2) (row, col) px on each side of the centreline; where an
    edge was not found, half the measured diameter."""
    pts, nrm, d = qca.centerline.points, qca.centerline.normals, qca.diameters
    left = np.where(np.isnan(d.left_px), -d.diameter_px / 2, d.left_px)
    right = np.where(np.isnan(d.right_px), d.diameter_px / 2, d.right_px)
    return pts + nrm * left[:, None], pts + nrm * right[:, None]


def lesion_zoom(
    image: np.ndarray, qca: QCAResult | None, mark: LesionMark, *, edges: bool = True,
    zoom: int = 4, half_px: int | None = None,
) -> np.ndarray:  # fmt: skip
    """An enlarged RGB view around a lesion's narrowest point: the angiogram, and if ``edges``
    the measured lumen edges (yellow) with the narrowest point marked (red)."""
    half = half_px or max(24, round(0.07 * max(image.shape)))
    pad = np.pad(image, half, mode="edge")
    r, c = mark.row + half, mark.col + half
    crop = pad[r - half : r + half, c - half : c + half]
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    out = cv2.cvtColor(big, cv2.COLOR_GRAY2RGB)
    if edges and qca is not None:
        origin = np.array([mark.row - half, mark.col - half], float)
        for line in edge_lines(qca):
            xy = np.round((line - origin) * zoom)[:, ::-1].astype(np.int32)
            cv2.polylines(out, [xy], False, PATH, 1, cv2.LINE_AA)
        cv2.circle(out, (half * zoom, half * zoom), 3 * zoom, LESION, 1, cv2.LINE_AA)
    return out


def contrast_state(rf, f: int) -> tuple[str, float]:
    """How full of contrast frame ``f`` is: (state, % of the run's peak vessel area). States:
    no contrast yet (before the arrival), filling, full (the opacification plateau, the frames
    the multi-frame median uses), washing out."""
    op = rf.opacification
    pct = 100.0 * float(op.smoothed[f]) / max(float(op.smoothed.max()), 1.0)
    lo, hi = opacified_frames(rf, None)
    if f < op.arrival:
        return "no contrast yet", pct
    if lo <= f <= hi:
        return "full", pct
    return ("filling", pct) if f < lo else ("washing out", pct)


def frame_chart(rf, frame: int) -> go.Figure:
    """Contrast per frame (% of peak vessel area), the full-contrast frames shaded, the
    end-diastolic frames (diamonds), the recommended one (★) and ``frame`` (line)."""
    op = rf.opacification
    t = np.arange(op.smoothed.size)
    peak = max(float(op.smoothed.max()), 1.0)
    lo, hi = opacified_frames(rf, None)
    fig = go.Figure()
    fig.add_vrect(x0=lo - 0.5, x1=hi + 0.5, fillcolor="rgba(27,175,122,0.10)", line_width=0,
                  annotation_text="full contrast", annotation_position="top left")  # fmt: skip
    fig.add_scatter(x=t, y=100 * op.smoothed / peak, name="contrast", hoverinfo="x+y",
                    line={"color": HEX["blue"], "width": 2})  # fmt: skip
    eds = [f for f in (rf.ed.frames if rf.ed.found else ()) if 0 <= f < t.size]
    fig.add_scatter(x=eds, y=[100 * op.smoothed[f] / peak for f in eds], mode="markers",
                    name="end-diastole", marker={"color": HEX["orange"], "symbol": "diamond",
                                                 "size": 9})  # fmt: skip
    r = rf.analysis_frame
    fig.add_scatter(x=[r], y=[100 * op.smoothed[r] / peak], mode="markers+text", text=["★"],
                    textposition="top center", name="recommended",
                    marker={"color": HEX["green"], "size": 12})  # fmt: skip
    fig.add_vline(x=frame, line={"color": "#52514e", "width": 1.5, "dash": "dash"})
    fig.update_layout(template="plotly_white", height=190, showlegend=True,
                      margin={"l": 40, "r": 10, "t": 10, "b": 30}, xaxis_title="frame",
                      yaxis_title="contrast %", yaxis_range=[0, 110],
                      legend={"orientation": "h", "y": 1.25, "x": 0})  # fmt: skip
    return fig


def contrast_text(tr, frame: int, frame_time_ms: float | None) -> str:
    """Contrast along the analysed vessel (P4k-I): its state on ``frame`` and the TIMI frame
    count (Gibson 1996, at 30 fps) if the frame rate is known."""
    from vstent.imaging.cine import path_state, timi_frame_count

    if tr is None:
        return "Contrast along the vessel: not measured on this run."
    parts = [f"Contrast along the vessel: {path_state(tr, frame) or 'unknown'} on frame {frame}"]
    tfc = timi_frame_count(tr, frame_time_ms)
    if tfc is not None:
        parts.append(f"TIMI frame count {tfc:.0f} (at 30 fps)")
    elif not frame_time_ms:
        parts.append("TIMI frame count needs the frame rate")
    else:
        parts.append("TIMI frame count: the contrast was not seen reaching the end of the path")
    return " · ".join(parts) + "."


def loss_split(pre, lesions: list[LesionMark]) -> float | None:
    """Share of the path's pressure loss (1 − distal FFR) that happens inside the marked
    lesions (``pre``: :class:`~vstent.app.core.Curves`); ``None`` without a loss."""
    total = 1.0 - float(pre.solver[-1])
    if total <= 1e-6:
        return None
    s_mm = m_to_mm(pre.s)
    focal = 0.0
    for m in lesions:
        a = float(np.interp(m.start_mm, s_mm, pre.solver))
        b = float(np.interp(m.end_mm, s_mm, pre.solver))
        focal += max(a - b, 0.0)
    return min(focal / total, 1.0)


def no_plan_reason(profile, plans, catalog, config) -> str:
    """Why the planner proposed no stent, in plain words, with what to try."""
    ref_mm = 2e3 * float(np.median(profile.reference_radius()))
    smallest = min(catalog.diameters_mm)
    if not plans.lesions:
        return ("The pressure loss is spread along the whole path (a small or long vessel, or "
                "diffuse disease), not at one narrowing, so no stent can fix it. Check that the "
                "path stays in the main vessel (Change path) and that the pixel size is right "
                "(Settings).")  # fmt: skip
    if smallest > config.max_diameter_ratio * ref_mm:
        return (f"The vessel measures {ref_mm:.1f} mm, smaller than the smallest stent "
                f"({smallest:g} mm) allows. With an assumed pixel size this is often too small: "
                "set the pixel size under Settings.")  # fmt: skip
    return ("The narrowing reaches the end of the path, so there is no healthy vessel for the "
            "stent to land on. Extend the path a little beyond the lesion "
            "(Change path).")  # fmt: skip


@dataclass(frozen=True)
class BoxCheck:
    """One expert box against the analysis: ``status`` "found" (a marked lesion inside it),
    "read mild" (the path crosses it but reads < 50%) or "off the path"; ``ds``: our largest
    %DS inside it (``None`` off the path)."""

    letter: str
    grade: str
    status: str
    ds: float | None
    lesion: int | None


def detection_check(
    qca: QCAResult | None, lesions: list[LesionMark], boxes, margin_px: float = 4.0
) -> tuple[list[BoxCheck], list[LesionMark]]:
    """Every expert box (x, y, w, h, grade) checked against the analysed path and its marked
    lesions; and the marked lesions inside no box (extra)."""
    checks, inside_any = [], set()
    pts = None if qca is None else qca.centerline.points
    ds = None if qca is None else 100.0 * (1.0 - qca.profile.r / qca.profile.reference_radius())
    for n, (x, y, w, h, grade) in enumerate(boxes):
        x0, y0, x1, y1 = x - margin_px, y - margin_px, x + w + margin_px, y + h + margin_px
        hit = [m for m in lesions if m.severe and x0 <= m.col <= x1 and y0 <= m.row <= y1]
        inside_any.update(m.number for m in hit)
        on = None
        if pts is not None:
            on = (pts[:, 1] >= x0) & (pts[:, 1] <= x1) & (pts[:, 0] >= y0) & (pts[:, 0] <= y1)
        best = float(ds[on].max()) if on is not None and on.any() else None
        if hit:
            top = max(hit, key=lambda m: m.ds_pct)
            checks.append(BoxCheck(chr(65 + n), grade, "found", top.ds_pct, top.number))
        elif best is not None:
            checks.append(BoxCheck(chr(65 + n), grade, "read mild", max(best, 0.0), None))
        else:
            checks.append(BoxCheck(chr(65 + n), grade, "off the path", None, None))
    return checks, [m for m in lesions if m.severe and m.number not in inside_any]
