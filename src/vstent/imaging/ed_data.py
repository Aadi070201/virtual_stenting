"""Synthetic training data for the learned end-diastolic frame model (P4l, M2). Torch-free.

Every case is a synthetic beating heart (``synthetic_cine``) with 7.5, 10, 15 or 30 fps; both of
its runs are used. Per run (``data/derived/ed/<split>/<seed>_<k>.npz``):
- ``frames``: (T, 128, 128) uint8, area-resized from the rendered 512 px frames;
- ``ed``: the true ED frames (the frame nearest each R wave);
- ``well``: the well-opacified frames as P7 scores them (segmented vessel area ≥ half its
  peak);
- ``m0``: the image-based ED frames of the product now (vessel-tree spread, M0);
- ``fps``.

Generation skips runs already written (resumable) and runs in worker processes, each with its
own segmentation model.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("data/derived/ed")
SPLITS = {"train": range(20000, 20250), "val": range(20250, 20300), "test": range(20300, 20360)}
SIZE = 128
FPS_CHOICES = (7.5, 10.0, 15.0, 30.0)

_SEG = None


def _segmenter(seg_dir: str):
    global _SEG
    if _SEG is None:
        from vstent.inference.onnx_runtime import SegmentationRunner

        _SEG = SegmentationRunner.from_dir(seg_dir)
    return _SEG


def small_frames(frames: np.ndarray) -> np.ndarray:
    """(T, 128, 128) uint8: a run's frames area-resized for the ED model (training and real
    runs alike)."""
    return np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in frames])


def make_case(seed: int, split: str, seg_dir: str, out_dir: str | Path = OUT_DIR) -> int:
    """Both runs of one case; returns the number of runs written (0 if already there)."""
    from vstent.imaging.cardiac import ED_AREA_FRACTION, ed_from_spread
    from vstent.imaging.image_io import read_run_bytes
    from vstent.imaging.run_analysis import analyse_frames
    from vstent.imaging.synthetic_cine import (
        CineSynthConfig,
        render_run,
        run_dicom,
        sample_cine_case,
    )

    out = Path(out_dir) / split
    paths = [out / f"{seed}_{k}.npz" for k in range(2)]
    if all(p.exists() for p in paths):
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cfg = CineSynthConfig(fps_choices=FPS_CHOICES)
    cc = sample_cine_case(seed, cfg)
    seg = _segmenter(seg_dir)
    written = 0
    for k, path in enumerate(paths):
        if path.exists():
            continue
        frames = render_run(cc, k, probs=False).frames
        loaded = read_run_bytes(run_dicom(cc, k, frames, cfg), f"case{seed}_run{k}.dcm")
        rf = analyse_frames(loaded, seg.predict)
        op = rf.opacification
        m0 = ed_from_spread(rf.spread, op.smoothed, loaded.frame_time_ms)
        np.savez_compressed(path, frames=small_frames(frames), ed=np.array(cc.ed_frames(k), int),
                            well=op.smoothed >= ED_AREA_FRACTION * op.smoothed.max(),
                            m0=np.array(m0.frames, int), fps=float(cc.runs[k].fps))  # fmt: skip
        written += 1
    return written


def generate(split: str, seg_dir: str, *, workers: int = 1, out_dir: str | Path = OUT_DIR,
             progress: Callable[[int, int], None] | None = None) -> int:  # fmt: skip
    """Every case of ``split`` (resumable); returns the runs written."""
    from concurrent.futures import ProcessPoolExecutor

    seeds = list(SPLITS[split])
    if workers <= 1:
        total = 0
        for i, s in enumerate(seeds):
            total += make_case(s, split, seg_dir, out_dir)
            if progress:
                progress(i + 1, len(seeds))
        return total
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(make_case, s, split, seg_dir, str(out_dir)) for s in seeds]
        for i, f in enumerate(futs):
            total += f.result()
            if progress:
                progress(i + 1, len(seeds))
    return total


def load_split(split: str, out_dir: str | Path = OUT_DIR) -> list[dict]:
    """Every run of ``split``: frames, ed, well, m0, fps, name."""
    runs = []
    for path in sorted((Path(out_dir) / split).glob("*.npz")):
        z = np.load(path)
        runs.append({"name": path.stem, "frames": z["frames"], "ed": tuple(z["ed"].tolist()),
                     "well": z["well"], "m0": tuple(z["m0"].tolist()),
                     "fps": float(z["fps"])})  # fmt: skip
    return runs
