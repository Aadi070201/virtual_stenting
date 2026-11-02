"""Multi-frame consensus for a cine run (P4g). Torch-free.

One automatic frame does not measure a lesion stably (P4f, D57). For a cine run the vessel is
therefore measured on up to K end-diastolic (ED) frames on which the analysed path is fully
opacified (:func:`consensus_frames`): from the frame the contrast has filled the path (its
transit, P7) to the end of the opacification plateau (before the washout). The user's two
points, placed on one frame, move to every other frame with the heart (affine transforms,
:mod:`vstent.imaging.motion`). The consensus is the
median distal FFR (solver) over the frames, reported with every frame's value. The frame whose
FFR is closest to the median gives the profile the rest of the analysis uses (uncertainty,
planner, figure). A consensus is flagged as unstable when its frames' FFR values lie on both
sides of 0.80. K was chosen on the CADICA development patients (docs/PLAN.md, P4g; D58).

The same function serves a single path (:func:`vessel_values`) and a vessel tree
(:func:`tree_values`). Used by ``vstent analyse``, the app and ``vstent validate``. A single
image is measured once.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from vstent.imaging.centerline import Centerline
from vstent.imaging.cine import ContrastTransit
from vstent.imaging.motion import apply, plateau
from vstent.imaging.run_analysis import RunFrames
from vstent.inference.pipeline import TreeAnalysis, VesselAnalysis

CONSENSUS_K = 3  # frames; set from the P4g development result (D58)
THRESHOLD = 0.80
# A frame whose traced path is this much longer or shorter than on the reference frame has
# most likely followed another vessel: it is listed with the reason and left out.
MAX_LENGTH_CHANGE = 0.30
FILLED_QUANTILE = 95  # the path is filled when this share (%) of its points is filled


def straddles(ffr, threshold: float = THRESHOLD) -> bool:
    """FFR values on both sides of ``threshold`` (some ≤, some >)."""
    v = np.asarray(ffr, float)
    return bool(v.size and v.min() <= threshold < v.max())


@dataclass(frozen=True)
class Values:
    """What the consensus reads from one frame's analysis: distal FFR (solver; surrogate if
    any), maximum %DS, path length (mm) and centreline of the path between the points."""

    ffr: float
    ds: float
    length_mm: float
    ffr_surrogate: float | None = None
    centerline: Centerline | None = None


def vessel_values(a: VesselAnalysis) -> Values:
    """A single path (:func:`~vstent.inference.pipeline.analyse_vessel`)."""
    return Values(a.ffr_distal_solver, a.qca.max_percent_stenosis, float(a.qca.s_mm[-1]),
                  a.ffr_distal_surrogate, a.qca.centerline)  # fmt: skip


def tree_values(a: TreeAnalysis) -> Values:
    """A vessel tree (:func:`~vstent.inference.pipeline.analyse_tree`): its main path."""
    main = a.qca.branches[0]
    return Values(float(a.result.ffr[0][-1]), main.max_percent_stenosis, float(main.s_mm[-1]),
                  centerline=main.centerline)  # fmt: skip


def filled_frame(transit: ContrastTransit | None) -> int | None:
    """The first frame on which the contrast has filled the path: :data:`FILLED_QUANTILE` % of
    its points at 90% of their plateau contrast (``None`` if the transit was not measured)."""
    if transit is None:
        return None
    f = transit.filled[np.isfinite(transit.filled)]
    return int(math.ceil(np.percentile(f, FILLED_QUANTILE))) if f.size else None


@dataclass(frozen=True, eq=False)
class FrameResult:
    """One frame of a consensus: the points moved there ((row, col) px) and its analysis and
    values, or why it was left out (``error``)."""

    frame: int
    start: tuple[int, int]
    end: tuple[int, int]
    analysis: Any = None
    values: Values | None = None
    error: str = ""

    @property
    def ffr(self) -> float | None:
        """Distal FFR (solver), ``None`` if left out."""
        return None if self.values is None else self.values.ffr


@dataclass(frozen=True, eq=False)
class Consensus:
    """The frames of a consensus (the reference frame first) and their median.

    Attributes:
        frames: every frame tried, the reference frame (the user's points) first.
        k: the number of frames asked for.
        source: where the frames came from (the ED frames' source, or why only one).
    """

    frames: tuple[FrameResult, ...]
    k: int
    source: str
    warning: str = ""

    @property
    def used(self) -> list[FrameResult]:
        """The frames measured and kept."""
        return [f for f in self.frames if f.values is not None]

    @property
    def ffr(self) -> float:
        """Median distal FFR (solver) of the frames used."""
        return float(np.median([f.values.ffr for f in self.used]))

    @property
    def ffr_surrogate(self) -> float | None:
        """Median distal FFR of the surrogate, if every frame used has one."""
        v = [f.values.ffr_surrogate for f in self.used]
        return None if None in v else float(np.median(v))

    @property
    def ds(self) -> float:
        """Median of the frames' maximum %DS."""
        return float(np.median([f.values.ds for f in self.used]))

    @property
    def chosen(self) -> FrameResult:
        """The frame used whose FFR is closest to the median (the earliest tried on a tie)."""
        m = self.ffr
        return min(self.used, key=lambda f: abs(f.values.ffr - m))

    @property
    def flagged(self) -> bool:
        """The frames' FFR values lie on both sides of 0.80."""
        return straddles([f.values.ffr for f in self.used])

    def note(self, label: str = "Distal FFR") -> str:
        """One line for the user; ``label`` names the FFR (e.g. the tree's main path)."""
        n = len(self.used)
        vals = ", ".join(f"frame {f.frame}: {f.values.ffr:.3f}" for f in self.used)
        if n > 1:
            head = f"{label}, median of {n} end-diastolic frames: {self.ffr:.3f} ({vals})"
        else:
            head = f"{label} on one frame only ({self.source}): {self.ffr:.3f}"
        out = [head + "."]
        if self.warning:
            out.append(self.warning)
        if n > 1:
            out.append(f"Frame {self.chosen.frame} is closest to the median; its profile is used "
                       "below.")  # fmt: skip
        if self.flagged:
            out.append(f"Unstable: the frames fall on both sides of {THRESHOLD:.2f}; check the "
                       "frames and the vessel outline before relying on this result.")  # fmt: skip
        out += [f"Frame {f.frame} left out: {f.error}." for f in self.frames if f.error]
        if 1 < len(self.frames) < self.k:
            out.append(f"{len(self.frames)} of the {self.k} frames asked for were available "
                       f"({self.source}).")  # fmt: skip
        return " ".join(out)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (case reports, predictions)."""
        return {
            "k": self.k, "source": self.source, "warning": self.warning,
            "ffr": round(self.ffr, 4),
            "ds_pct": round(self.ds, 1), "chosen_frame": self.chosen.frame,
            "unstable": self.flagged,
            "frames": [{"frame": f.frame, "start": list(f.start), "end": list(f.end),
                        "ffr": None if f.values is None else round(f.values.ffr, 4),
                        "ds_pct": None if f.values is None else round(f.values.ds, 1),
                        "left_out": f.error} for f in self.frames],
        }  # fmt: skip


def opacified_frames(rf: RunFrames, filled: int | None) -> tuple[int, int]:
    """First and last frame on which the analysed path is fully opacified: from ``filled``
    (:func:`filled_frame`; the start of the opacification plateau if unknown) to the end of
    the plateau (the frames with ≥ 90% of the peak vessel area, :func:`motion.plateau`)."""
    lo, hi = plateau(rf.opacification.smoothed)
    return (lo if filled is None else max(lo, filled)), hi


def consensus_frames(
    rf: RunFrames, reference: int, k: int = CONSENSUS_K, filled: int | None = None
) -> tuple[list[int], str]:
    """``reference`` (where the points were placed), then the other ED frames on which the
    path is fully opacified (:func:`opacified_frames`) with the most vessel (smoothed area), up
    to ``k`` frames; and the source of the frames."""
    op, ed = rf.opacification, rf.ed
    if k <= 1:
        return [reference], "one frame asked for"
    if not ed.found:
        return [reference], "end-diastolic frames unknown"
    lo, hi = opacified_frames(rf, filled)
    others = sorted((f for f in ed.frames if lo <= f <= hi and f != reference),
                    key=lambda f: (-op.smoothed[f], f))  # fmt: skip
    frames = [reference, *others[: k - 1]]
    source = (f"end-diastolic frames from the {ed.source}" if len(frames) > 1 else
              "no other end-diastolic frame with the path fully opacified")  # fmt: skip
    return frames, source


def _moved(warp: np.ndarray, point, shape) -> tuple[int, int]:
    r, c = apply(warp, np.asarray([point], float))[0]
    return (int(np.clip(round(r), 0, shape[0] - 1)), int(np.clip(round(c), 0, shape[1] - 1)))


def run_consensus(
    rf: RunFrames,
    reference: int,
    start,
    end,
    segment: Callable[[np.ndarray], np.ndarray],
    analyse: Callable[[np.ndarray, np.ndarray, tuple, tuple], Any],
    *,
    k: int = CONSENSUS_K,
    values: Callable[[Any], Values] = vessel_values,
    reference_analysis: Any = None,
) -> Consensus:
    """The consensus of the vessel between ``start`` and ``end`` ((row, col) px on frame
    ``reference``) over up to ``k`` ED frames.

    ``segment``: frame → probability map. ``analyse(prob, image, start, end)``: one frame's
    analysis with the caller's pixel size, pressures, lumen and models; ``values`` reads it
    (:func:`vessel_values` or :func:`tree_values`). ``reference_analysis``: the reference
    frame's analysis if already done. Raises ``ValueError`` if the reference frame cannot be
    measured.
    """
    start, end = tuple(map(int, start)), tuple(map(int, end))
    ref = reference_analysis
    if ref is None:
        img = rf.run.frames[reference]
        ref = analyse(segment(img), img, start, end)
    ref_values = values(ref)
    filled = None
    if k > 1 and rf.ed.found and ref_values.centerline is not None:
        filled = filled_frame(rf.transit(ref_values.centerline, reference))
    frames, source = consensus_frames(rf, reference, k, filled)
    lo, hi = opacified_frames(rf, filled)
    warning = ""
    if filled is not None and reference < filled:
        warning = (f"On frame {reference} the contrast has not yet filled the whole path (filled "
                   f"from frame {filled}).")  # fmt: skip
    elif k > 1 and not lo <= reference <= hi:
        warning = f"Frame {reference} is outside the fully opacified frames ({lo}–{hi})."
    out = [FrameResult(reference, start, end, ref, ref_values)]
    warps = rf.warps(reference) if len(frames) > 1 else None
    for f in frames[1:]:
        img = rf.run.frames[f]
        a, b = _moved(warps[f], start, img.shape), _moved(warps[f], end, img.shape)
        try:
            res = analyse(segment(img), img, a, b)
        except ValueError as exc:
            out.append(FrameResult(f, a, b, error=f"the vessel could not be traced ({exc})"))
            continue
        v = values(res)
        if abs(v.length_mm / ref_values.length_mm - 1.0) > MAX_LENGTH_CHANGE:
            out.append(FrameResult(f, a, b, error=f"path {v.length_mm:.0f} mm against "
                                   f"{ref_values.length_mm:.0f} mm on frame {reference}: the "
                                   "moved points probably found another vessel"))  # fmt: skip
            continue
        out.append(FrameResult(f, a, b, res, v))
    return Consensus(tuple(out), k, source, warning)
