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


def product_on_run(run, seg_dir: str):
    """(measurements, the run's frame analysis) of one :class:`~vstent.imaging.image_io.RunInput`
    through the product's steps. Raises if the run cannot be measured."""
    from vstent.app.guided import MARK_MIN_DS, auto_path, lesion_marks
    from vstent.imaging.run_analysis import analyse_frames
    from vstent.inference.consensus import run_consensus
    from vstent.inference.pipeline import analyse_vessel

    w = product_state(seg_dir)
    seg, cfg, params, reader = w["seg"], w["cfg"], w["params"], w["reader"]
    rf = analyse_frames(run, seg.predict)
    frame = rf.analysis_frame
    image = run.frames[frame]
    prob = seg.predict(image)
    start, end = auto_path(prob, run.pixel_spacing_mm)

    def one(p, im, a, b):
        return analyse_vessel(p, a, b, params=params, mm_per_px=run.pixel_spacing_mm, image=im,
                              lesion_reader=reader, wire=cfg.wire,
                              lesion_min_ref_mm=cfg.lesion_min_ref_mm)  # fmt: skip

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
    }  # fmt: skip
    out["flagged"] = bool(severe) or round(out["ffr"], 2) <= 0.80
    return out, rf
