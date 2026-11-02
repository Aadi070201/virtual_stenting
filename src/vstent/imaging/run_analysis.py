"""One cine run with the product's steps (P7). Torch-free.

:func:`analyse_frames` segments every frame once. That gives the opacification curve (P4d),
the vessel tree's spread, the ED frames (R-wave pointer, ECG, then the images;
:mod:`vstent.imaging.cardiac`) and the analysis frame: the ED frame in the opacified window with
the most vessel. :meth:`RunFrames.velocity` then measures the flow velocity along a centreline
drawn on a frame of the run, with its points following the heart
(:mod:`vstent.imaging.motion`). Used by ``vstent analyse``, the app and the P7 report.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from vstent.imaging.cardiac import EndDiastole, ed_analysis_frame, end_diastole, tree_spread
from vstent.imaging.centerline import Centerline
from vstent.imaging.cine import ContrastTransit, Opacification, contrast_transit
from vstent.imaging.cine import opacification as opacification_curve
from vstent.imaging.flow import FlowVelocity, flow_velocity
from vstent.imaging.image_io import RunInput
from vstent.imaging.motion import plateau, small_map, track_affine_maps

# Vessel pixels counted for the run's curves (opacification, tree spread): confident ones only.
# On the CADICA development runs, 0.9 put the analysis frame inside the clinicians' selected
# span more often than the segmentation threshold of 0.5 (D55); QCA keeps 0.5.
RUN_LEVEL = 0.9


@dataclass(eq=False)
class RunFrames:
    """What one pass over a run's frames gives.

    Attributes:
        run: the run.
        area: (T,) vessel pixels per frame.
        spread: (T,) vessel-tree spread per frame (px; NaN where too little vessel).
        small: half-size smoothed probability maps (motion tracking).
        opacification: the opacification curve (P4d).
        ed: the ED frames and their source.
        analysis_frame: the frame to analyse (0-based).
        frame_note: why that frame.
    """

    run: RunInput
    area: np.ndarray
    spread: np.ndarray
    small: list[np.ndarray]
    opacification: Opacification
    ed: EndDiastole
    analysis_frame: int
    frame_note: str
    _warps: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def warps(self, frame: int) -> np.ndarray:
        """(T, 2, 3) transforms from ``frame``'s (row, col) to every frame's (cached)."""
        if frame not in self._warps:
            self._warps[frame] = track_affine_maps(
                self.small, self.area, frame, window=plateau(self.opacification.smoothed),
                period=self.ed.period_frames,
            )  # fmt: skip
        return self._warps[frame]

    def transit(self, centerline: Centerline, frame: int) -> ContrastTransit | None:
        """Contrast arrival along ``centerline`` (drawn on ``frame``), following the heart."""
        return contrast_transit(self.run.frames, centerline, warps=self.warps(frame))

    def velocity(
        self,
        centerline: Centerline,
        frame: int,
        distance_m: Callable[[np.ndarray], np.ndarray] | None,
        *,
        single_view: bool,
        hyperemic_run: bool = False,
    ) -> tuple[FlowVelocity, ContrastTransit | None]:
        """The flow velocity along ``centerline`` (drawn on ``frame``) and its transit;
        ``distance_m`` maps the centreline's arc length (px) to metres along the path."""
        tr = self.transit(centerline, frame)
        fv = flow_velocity(tr, self.run.frame_time_ms, distance_m, ed_frames=self.ed.frames,
                           period_frames=self.ed.period_frames, single_view=single_view,
                           hyperemic_run=hyperemic_run)  # fmt: skip
        return fv, tr


def pixel_distance(mm_per_px: float | None) -> Callable[[np.ndarray], np.ndarray] | None:
    """Arc length (px) → m at a calibrated pixel size (``None`` if not calibrated)."""
    if not mm_per_px:
        return None
    m = float(mm_per_px) * 1e-3
    return lambda s: np.asarray(s, float) * m


def analyse_frames(run: RunInput, segment: Callable[[np.ndarray], np.ndarray]) -> RunFrames:
    """Segment every frame of ``run`` once (``segment``: frame → probability map) and find
    the opacification, the ED frames and the analysis frame."""
    area, spread, small = [], [], []
    for f in run.frames:
        prob = segment(f)
        mask = prob >= RUN_LEVEL
        area.append(int(mask.sum()))
        spread.append(tree_spread(mask))
        small.append(small_map(prob))
    area_a, spread_a = np.array(area), np.array(spread, float)
    op = opacification_curve(area_a)
    ed = end_diastole(len(run), run.frame_time_ms, pointer=run.r_wave_frames, ecg=run.ecg,
                      spread=spread_a, smoothed_area=op.smoothed)  # fmt: skip
    frame, note = ed_analysis_frame(op, ed)
    return RunFrames(run, area_a, spread_a, small, op, ed, frame, note)
