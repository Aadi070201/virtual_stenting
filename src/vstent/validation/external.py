"""The product on one cine run, as the external checks use it (P4k-J AngioCAD, P4o
CardioSYNTAX): the run's analysis frame, the automatic path, the product's QCA and physics
(``configs/product.yaml``, the app's settings), the median over up to five heartbeats. A run is
**flagged** if it has a marked lesion ≥ 50% or a virtual FFR ≤ 0.80. Torch-free.
"""

from __future__ import annotations

from typing import Any

_STATE: dict[str, Any] = {}


def product_state(seg_dir: str) -> dict[str, Any]:
    """The segmenter, the product settings, physics parameters and lesion reader (loaded once
    per process)."""
    if not _STATE:
        from vstent.app.core import load_settings
        from vstent.inference.onnx_runtime import SegmentationRunner
        from vstent.inference.product import product_config, product_params, product_reader

        cfg = product_config()
        _STATE.update(
            seg=SegmentationRunner.from_dir(seg_dir), cfg=cfg,
            params=product_params(load_settings()[0], cfg),  # as the app
            reader=product_reader(cfg.lesion_reader),
        )  # fmt: skip
    return _STATE


PROFILE_STEP_MM = 1.0


def profile_of(analysis, step_mm: float = PROFILE_STEP_MM) -> dict[str, Any]:
    """The analysed vessel of a :class:`~vstent.inference.pipeline.VesselAnalysis` on a
    ``step_mm`` grid: arc length, lumen and reference diameter (mm), the solver's FFR pullback,
    and each lesion's start, narrowest point and end (mm), %DS, MLD and reference there (mm)."""
    import numpy as np

    from vstent.units import m_to_mm

    q = analysis.qca
    s = q.s_mm
    grid = np.arange(0.0, float(s[-1]) + 1e-9, step_mm)
    ref = np.broadcast_to(q.reference_diameter_mm, s.shape)
    ffr = np.interp(grid, m_to_mm(np.asarray(analysis.solver.s)), analysis.solver.ffr)
    lesions = [[m_to_mm(les.s_start), m_to_mm(les.s_mld), m_to_mm(les.s_end),
                float(les.max_ds_pct), m_to_mm(les.mld),
                float(np.interp(m_to_mm(les.s_mld), s, ref))]
               for les in q.lesions]  # fmt: skip
    r3 = lambda a: [round(float(x), 3) for x in a]  # noqa: E731
    return {"s_mm": r3(grid), "d_mm": r3(np.interp(grid, s, q.diameter_mm)),
            "ref_mm": r3(np.interp(grid, s, ref)), "ffr": r3(ffr),
            "lesions": [r3(les) for les in lesions]}  # fmt: skip


PRODUCT = "product"


def product_on_run(run, seg_dir: str, *, detail: bool = False, gate=PRODUCT, rf=None):
    """(measurements, the run's frame analysis) of one :class:`~vstent.imaging.image_io.RunInput`
    through the product's steps. Raises if the run cannot be measured. With ``detail``, the
    measurements also hold the chosen heartbeat's :func:`profile_of` (``"profile"``). ``gate``:
    a :class:`~vstent.imaging.lesion_gate.LesionGate` or ``None`` in place of the product's
    (P4q); ``rf``: the run's frame analysis, if already done (it does not depend on ``gate``)."""
    from vstent.app.guided import MARK_MIN_DS, auto_path, lesion_marks
    from vstent.imaging.run_analysis import analyse_frames
    from vstent.inference.consensus import run_consensus
    from vstent.inference.pipeline import analyse_vessel
    from vstent.inference.product import product_gate

    w = product_state(seg_dir)
    seg, cfg, params, reader = w["seg"], w["cfg"], w["params"], w["reader"]
    if gate == PRODUCT:
        gate = product_gate(cfg)
    rf = rf if rf is not None else analyse_frames(run, seg.predict)
    frame = rf.analysis_frame
    image = run.frames[frame]
    prob = seg.predict(image)
    start, end = auto_path(prob, run.pixel_spacing_mm)

    def one(p, im, a, b):
        return analyse_vessel(p, a, b, params=params, mm_per_px=run.pixel_spacing_mm, image=im,
                              lesion_reader=reader, wire=cfg.wire,
                              lesion_min_ref_mm=cfg.lesion_min_ref_mm,
                              lesion_gate=gate)  # fmt: skip

    cons = run_consensus(rf, frame, start, end, seg.predict, one,
                         reference_analysis=one(prob, image, start, end))  # fmt: skip
    q = cons.chosen.analysis.qca
    marks = lesion_marks(q, 0.0)
    severe = [m for m in marks if m.ds_pct >= MARK_MIN_DS]
    out = {
        "frames": len(run), "frame": int(frame), "chosen": int(cons.chosen.frame),
        "start": list(map(int, cons.chosen.start)), "end": list(map(int, cons.chosen.end)),
        "heartbeats": len(cons.used), "ffr": float(cons.ffr),
        "max_ds": float(max((m.ds_pct for m in marks), default=0.0)),
        "marked": len(severe), "length_mm": float(q.s_mm[-1]),
        "gate": None if cons.chosen.analysis.gate is None else list(cons.chosen.analysis.gate),
    }  # fmt: skip
    out["flagged"] = bool(severe) or round(out["ffr"], 2) <= 0.80
    if detail:
        out["profile"] = profile_of(cons.chosen.analysis)
    return out, rf
