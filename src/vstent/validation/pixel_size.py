"""P4r: the true pixel size on CARDIAG (acquisition geometry), checked against the experts'
catheter masks, and how well a catheter of assumed size calibrates the scale (docs/PLAN.md,
P4r). Torch-free.

CARDIAG is read inside ``CARDIAG.zip``: each sample's ``metadata.json``, ``catheter.png`` and
the size of ``img.png``. Nothing is extracted and no model is run.

Post hoc (added after the catheter check failed; not pre-registered): the experts' masks of the
left main and the proximal LAD against their normal diameters (Dodge et al., Circulation 1992).
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

ASSUMED_MM = 0.30  # the product's assumed pixel size (mm per pixel)
CATHETER_MM = 2.0  # 6 F
FRENCH_MM = (5 / 3, 7 / 3)  # 5–7 F (1 F = 1/3 mm)
MIN_SKELETON = 20  # skeleton pixels for a width
# post hoc: SYNTAX segment masks and normal diameters (mm; Dodge et al., Circulation 1992)
SEGMENTS = {"05": ("left main", 4.5), "06": ("proximal LAD", 3.7)}
ARCHIVE = Path("data/raw/cardiag/CARDIAG.zip")


@dataclass(frozen=True)
class Sample:
    """One CARDIAG frame: ``mm_per_px`` at the isocentre (imager spacing × SOD / SID, ``None``
    if the geometry is missing), the PNG's shape, and the experts' catheter width (pixels,
    ``None`` if not measurable) with its skeleton length (pixels); ``segments``: the width
    (pixels) of each :data:`SEGMENTS` mask present and measurable (post hoc)."""

    name: str
    centre: str
    patient: str
    mm_per_px: float | None
    shape: tuple[int, int] | None
    catheter_px: float | None
    skeleton_px: int
    segments: tuple[tuple[str, float], ...] = ()


def _numbers(value) -> list[float]:
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, list | tuple):
        return [x for v in value for x in _numbers(v)]
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))]


def isocentre_mm(meta: dict[str, Any]) -> float | None:
    """Pixel size at the isocentre (mm per pixel): the mean imager spacing (mm) × SOD / SID;
    ``None`` if a value is missing or not positive."""
    try:
        spacing = _numbers(meta["spacing"])
        sod, sid = _numbers(meta["sod"])[0], _numbers(meta["sid"])[0]
    except (KeyError, IndexError, ValueError):
        return None
    if not spacing or min(spacing) <= 0 or sod <= 0 or sid <= 0:
        return None
    return float(np.mean(spacing)) * sod / sid


def mask_width_px(mask: np.ndarray) -> tuple[float | None, int]:
    """A tube mask's width (pixels; the catheter, a vessel segment): the median of 2·d − 1 over
    the mask's skeleton (d: the Euclidean distance to the background); ``None`` below
    :data:`MIN_SKELETON` skeleton pixels. Also the skeleton's length (pixels)."""
    m = np.asarray(mask) > 0
    if not m.any():
        return None, 0
    skel = skeletonize(m)
    n = int(skel.sum())
    if n < MIN_SKELETON:
        return None, n
    d = ndimage.distance_transform_edt(m)
    return float(np.median(2 * d[skel] - 1)), n


def _png(zf: zipfile.ZipFile, name: str) -> np.ndarray | None:
    data = np.frombuffer(zf.read(name), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def read_samples(archive: str | Path = ARCHIVE) -> list[Sample]:
    """Every CARDIAG sample folder with a ``metadata.json``, in name order."""
    out = []
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        folders = sorted(n.split("/")[0] for n in names
                         if n.count("/") == 1 and n.endswith("/metadata.json"))  # fmt: skip
        for f in folders:
            meta = json.loads(zf.read(f"{f}/metadata.json"))
            img = _png(zf, f"{f}/img.png") if f"{f}/img.png" in names else None
            cat = _png(zf, f"{f}/catheter.png") if f"{f}/catheter.png" in names else None
            width, n = mask_width_px(cat) if cat is not None else (None, 0)
            seg = []
            for k in SEGMENTS:
                if f"{f}/{k}.png" in names and (w := mask_width_px(_png(zf, f"{f}/{k}.png"))[0]):
                    seg.append((k, w))
            out.append(Sample(f, str(meta.get("center_id", "")), str(meta.get("patient_id", "")),
                              isocentre_mm(meta), None if img is None else img.shape[:2],
                              width, n, tuple(seg)))  # fmt: skip
    return out


def _spread(x) -> dict[str, float | int | None]:
    x = np.asarray([v for v in x if v is not None], float)
    if not x.size:
        return {"n": 0, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max())}  # fmt: skip


def summarise(samples: list[Sample], input_rel: float = 0.10) -> dict[str, Any]:
    """A, B and C of P4r; ``input_rel``: the input intervals' relative pixel-size range."""
    true = [s for s in samples if s.mm_per_px is not None]
    mm = np.array([s.mm_per_px for s in true], float)
    lo, hi = ASSUMED_MM * (1 - input_rel), ASSUMED_MM * (1 + input_rel)
    centres = sorted({s.centre for s in true})
    a = {
        "frames": len(samples), "with_geometry": len(true),
        "patients": len({s.patient for s in true}), "centres": len(centres),
        "mm_per_px": _spread(mm),
        "by_centre": [{"centre": f"C{k + 1}", **_spread([s.mm_per_px for s in true
                                                          if s.centre == c])}
                      for k, c in enumerate(centres)],
        "overstatement": _spread(ASSUMED_MM / mm) if mm.size else _spread([]),
        "inside_input_range": float(np.mean((mm >= lo) & (mm <= hi))) if mm.size else None,
        "input_range": [lo, hi],
        "shapes": {f"{h}×{w}": n for (h, w), n in Counter(s.shape for s in samples
                                                          if s.shape).most_common()},
    }  # fmt: skip
    both = [s for s in true if s.catheter_px is not None]
    width_mm = np.array([s.catheter_px * s.mm_per_px for s in both], float)
    b = {
        "measured": len(both),
        "no_catheter": sum(s.skeleton_px == 0 for s in samples),
        "too_short": sum(0 < s.skeleton_px < MIN_SKELETON for s in samples),
        "width_px": _spread([s.catheter_px for s in both]),
        "width_mm": _spread(width_mm),
        "inside_french": (
            float(np.mean((width_mm >= FRENCH_MM[0]) & (width_mm <= FRENCH_MM[1])))
            if width_mm.size
            else None
        ),  # fmt: skip
    }
    med = b["width_mm"]["median"]
    b["scale_holds"] = med is not None and FRENCH_MM[0] <= med <= FRENCH_MM[1]
    err = np.array([CATHETER_MM / s.catheter_px / s.mm_per_px - 1 for s in both], float)
    c: dict[str, Any] = {"frames": int(err.size)}
    if err.size:
        c.update(median_abs_error=float(np.median(np.abs(err))),
                 median_error=float(np.median(err)),
                 within_10=float(np.mean(np.abs(err) <= 0.10)),
                 within_20=float(np.mean(np.abs(err) <= 0.20)))  # fmt: skip
        c["worth_building"] = bool(c["median_abs_error"] <= 0.10 and c["within_20"] >= 0.70)
    else:
        c["worth_building"] = False
    vessels = []
    for k, (label, normal) in SEGMENTS.items():
        px = [(dict(s.segments)[k], s.mm_per_px) for s in true if k in dict(s.segments)]
        mm_a = [w * m for w, m in px]
        vessels.append({"segment": k, "label": label, "normal_mm": normal,
                        "width_px": _spread([w for w, _m in px]), "width_mm": _spread(mm_a),
                        "factor": _spread([normal / x for x in mm_a])})  # fmt: skip
    cat_factor = _spread([CATHETER_MM / x for x in width_mm]) if width_mm.size else _spread([])
    return {"a": a, "b": b, "c": c, "post_hoc": {"vessels": vessels, "catheter": cat_factor}}


def _f(x, fmt=".3f") -> str:
    return "–" if x is None else format(x, fmt)


def report_lines(s: dict[str, Any], git: dict[str, Any]) -> list[str]:
    """The Markdown report."""
    a, b, c = s["a"], s["b"], s["c"]
    m, o = a["mm_per_px"], a["overstatement"]
    w, wp = b["width_mm"], b["width_px"]
    lo, hi = a["input_range"]
    verdict = ("met: an automatic catheter calibration is worth building" if c["worth_building"]
               else "not met: no catheter calibration is built")  # fmt: skip
    lines = [
        "# P4r: the true pixel size (CARDIAG)", "",
        "> **Research use only — not for clinical decisions.**", "",
        "Generated by `vstent pixel-size`. Do not edit by hand. Pre-registered in docs/PLAN.md, "
        "P4r. CARDIAG (Zenodo 19958730, CC BY 4.0), read inside its zip; no model is run.", "",
        f"{a['frames']} frames ({a['with_geometry']} with geometry), {a['patients']} patients, "
        f"{a['centres']} centres. Image sizes: "
        + ", ".join(f"{k} ({n})" for k, n in a["shapes"].items()) + ".", "",
        "## A. True pixel size at the isocentre (imager spacing × SOD / SID)", "",
        "| | Median | Interquartile range | Range |", "|---|---:|---:|---:|",
        f"| mm per pixel | {_f(m['median'])} | {_f(m['q1'])}–{_f(m['q3'])} | "
        f"{_f(m['min'])}–{_f(m['max'])} |",
        f"| Assumed 0.30 mm ÷ true (sizes overstated by) | {_f(o['median'], '.2f')}× | "
        f"{_f(o['q1'], '.2f')}–{_f(o['q3'], '.2f')}× | {_f(o['min'], '.2f')}–"
        f"{_f(o['max'], '.2f')}× |", "",
        "By centre (hashed in the dataset; numbered here):", "",
        "| Centre | Frames | Median mm per pixel | Interquartile range |",
        "|---|---:|---:|---:|",
        *(f"| {r['centre']} | {r['n']} | {_f(r['median'])} | {_f(r['q1'])}–{_f(r['q3'])} |"
          for r in a["by_centre"]), "",
        f"Frames inside the input intervals' pixel-size range ({lo:.2f}–{hi:.2f} mm): "
        f"{_f(a['inside_input_range'] and 100 * a['inside_input_range'], '.0f')}%.", "",
        "## B. Scale check: the experts' catheter masks", "",
        f"{b['measured']} frames measured ({b['no_catheter']} without a catheter mask, "
        f"{b['too_short']} with fewer than {MIN_SKELETON} skeleton pixels).", "",
        "| | Median | Interquartile range | Range |", "|---|---:|---:|---:|",
        f"| Catheter width, pixels | {_f(wp['median'], '.1f')} | {_f(wp['q1'], '.1f')}–"
        f"{_f(wp['q3'], '.1f')} | {_f(wp['min'], '.1f')}–{_f(wp['max'], '.1f')} |",
        f"| Catheter width, mm (pixels × A) | {_f(w['median'], '.2f')} | {_f(w['q1'], '.2f')}–"
        f"{_f(w['q3'], '.2f')} | {_f(w['min'], '.2f')}–{_f(w['max'], '.2f')} |", "",
        f"Widths inside 5–7 F ({FRENCH_MM[0]:.2f}–{FRENCH_MM[1]:.2f} mm): "
        f"{_f(b['inside_french'] and 100 * b['inside_french'], '.0f')}%. "
        f"**Scale {'holds' if b['scale_holds'] else 'does not hold'}** (median width "
        f"{'inside' if b['scale_holds'] else 'outside'} 5–7 F).", "",
        "## C. Catheter calibration (6 F = 2.0 mm assumed)", "",
        f"On {c['frames']} frames, against A: median absolute error "
        f"{_f(c.get('median_abs_error') and 100 * c['median_abs_error'], '.1f')}% (median signed "
        f"{_f(c.get('median_error') and 100 * c['median_error'], '+.1f')}%); within 10%: "
        f"{_f(c.get('within_10') and 100 * c['within_10'], '.0f')}%; within 20%: "
        f"{_f(c.get('within_20') and 100 * c['within_20'], '.0f')}%.", "",
        f"**Rule (P4r):** median absolute error ≤ 10% and ≥ 70% of frames within 20% → "
        f"{verdict}.", "",
        "## Post hoc: vessel widths (added after B failed; not pre-registered)", "",
        "After the catheter check failed, one frame was looked at (frame 000: the proximal "
        "right coronary artery about 11 pixels wide, 2.3 mm at A). The experts' masks of two "
        "proximal segments are measured as the catheter was, against their normal diameters "
        "(Dodge et al., Circulation 1992). Factor = normal ÷ width at A: about 1 if A is the "
        "image's scale, about 2 if the images were halved after the spacing was recorded.", "",
        "| Segment | Frames | Width, pixels | Width at A, mm | Normal, mm | Factor (median, IQR) |",
        "|---|---:|---:|---:|---:|---:|",
        *(f"| {v['segment']} ({v['label']}) | {v['width_px']['n']} | "
          f"{_f(v['width_px']['median'], '.1f')} | {_f(v['width_mm']['median'], '.2f')} | "
          f"{v['normal_mm']:.1f} | {_f(v['factor']['median'], '.2f')} "
          f"({_f(v['factor']['q1'], '.2f')}–{_f(v['factor']['q3'], '.2f')}) |"
          for v in s["post_hoc"]["vessels"]),
        f"| Catheter (6 F) | {b['measured']} | {_f(wp['median'], '.1f')} | "
        f"{_f(w['median'], '.2f')} | {CATHETER_MM:.1f} | "
        f"{_f(s['post_hoc']['catheter']['median'], '.2f')} "
        f"({_f(s['post_hoc']['catheter']['q1'], '.2f')}–"
        f"{_f(s['post_hoc']['catheter']['q3'], '.2f')}) |", "",
        "Normal diameters are population means (men and women differ; diseased segments are "
        "narrower), so the factor is approximate. The paper (arXiv:2607.22139) gives neither the "
        "original image size nor the grid the spacing refers to.", "",
        "**Limits:** the isocentre is not the vessel's depth (P5: about 4% median error at the "
        "vessel). The catheter's French size is not recorded (6 F assumed; 5–7 F span ± 17%), "
        "and the catheter sits near the aortic root. The width rule (2·d − 1 on the skeleton) "
        "reads even widths one pixel narrow. One vendor (Siemens) and one region (northern "
        "Poland): the product's default is not changed from this report.", "",
        f"Commit `{git.get('git_commit')}` (dirty: {git.get('git_dirty')}).",
    ]  # fmt: skip
    return lines
