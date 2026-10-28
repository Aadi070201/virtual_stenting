"""ONNX Runtime inference for the FFR surrogate (no torch needed).

Inputs go through :func:`vstent.surrogate.features.prepare_inputs`, the same pre-processing as
training. A model directory (``models/surrogate/<run-name>/``) holds ``surrogate.onnx`` and
``norm_stats.json``; the ONNX file carries metadata that is checked against the statistics.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort

from vstent.physics.geometry import VesselProfile
from vstent.surrogate.features import NORM_STATS, NormStats, RawInputs, prepare_inputs

ONNX_FILE = "surrogate.onnx"
INPUT_NAMES = ("x_point", "x_global")
OUTPUT_NAME = "ffr"
CHUNK_SIZE = 16


def default_workers() -> int:
    """Half the logical CPUs (≈ physical cores on SMT machines), at least 1."""
    return max(1, (os.cpu_count() or 1) // 2)


class SurrogateRunner:
    """FFR pullback prediction with an exported surrogate.

    Large batches are split into chunks of ``chunk_size`` vessels that run in parallel on
    ``workers`` threads, each ONNX call single-threaded: activations of a small chunk stay in
    the CPU cache, which is ~2.5x faster than one multi-threaded call on the whole batch
    (DECISIONS D17). Results do not depend on the chunking.

    Args:
        model_path: the ``.onnx`` file.
        stats_path: normalisation statistics; default ``norm_stats.json`` next to the model.
        workers: parallel chunk workers; ``None`` = :func:`default_workers`.
        chunk_size: vessels per ONNX call; ``None`` = the whole batch in one call.
        intra_op_threads: ONNX Runtime threads per call; ``None`` = 1 when ``workers`` > 1,
            else the ONNX Runtime default (all physical cores).
        providers: ONNX Runtime execution providers.
    """

    def __init__(
        self,
        model_path: str | Path,
        stats_path: str | Path | None = None,
        *,
        workers: int | None = None,
        chunk_size: int | None = CHUNK_SIZE,
        intra_op_threads: int | None = None,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.model_path = Path(model_path)
        self.stats = NormStats.load(stats_path or self.model_path.with_name(NORM_STATS))
        self.workers = default_workers() if workers is None else workers
        self.chunk_size = chunk_size
        if intra_op_threads is None and self.workers > 1:
            intra_op_threads = 1
        self.intra_op_threads = intra_op_threads
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads is not None:
            opts.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=list(providers))
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self._check_metadata()
        self._pool = ThreadPoolExecutor(self.workers) if self.workers > 1 else None

    @classmethod
    def from_dir(cls, model_dir: str | Path, **kwargs) -> "SurrogateRunner":
        """Load ``surrogate.onnx`` + ``norm_stats.json`` from a model directory."""
        return cls(Path(model_dir) / ONNX_FILE, **kwargs)

    def _check_metadata(self) -> None:
        expected = {
            "feature_version": str(self.stats.feature_version),
            "n_points": str(self.stats.n_points),
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(
                    f"{self.model_path}: {key} = {self.metadata.get(key)!r} in the model, "
                    f"{value!r} in the normalisation statistics"
                )

    @property
    def n_points(self) -> int:
        """Number of centreline points the model expects."""
        return self.stats.n_points

    def close(self) -> None:
        """Stop the worker threads (the runner can still predict, serially)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _run(self, x_point: np.ndarray, x_global: np.ndarray) -> np.ndarray:
        feeds = {
            INPUT_NAMES[0]: np.ascontiguousarray(x_point, dtype=np.float32),
            INPUT_NAMES[1]: np.ascontiguousarray(x_global, dtype=np.float32),
        }
        return self.session.run([OUTPUT_NAME], feeds)[0]

    def predict_features(self, x_point: np.ndarray, x_global: np.ndarray) -> np.ndarray:
        """FFR (B, N), float32, from standardised features (B, 4, N) and (B, 5)."""
        n = x_point.shape[0]
        if self.chunk_size is None or n <= self.chunk_size:
            return self._run(x_point, x_global)
        starts = range(0, n, self.chunk_size)

        def chunk(i: int) -> np.ndarray:
            return self._run(x_point[i : i + self.chunk_size], x_global[i : i + self.chunk_size])

        parts = self._pool.map(chunk, starts) if self._pool is not None else map(chunk, starts)
        return np.concatenate(list(parts))

    def predict(self, raw: RawInputs) -> np.ndarray:
        """FFR pullback (B, N), dimensionless, for vessels in SI units (see ``RawInputs``)."""
        return self.predict_features(*prepare_inputs(raw, self.stats))

    def predict_profile(
        self, profile: VesselProfile, pa: float, pv: float, r_micro: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """One vessel: returns (s (m), FFR) on the model's N points along the centreline.

        The profile is resampled to N points. Pressures in Pa, R_micro in Pa·s/m³.
        """
        raw = RawInputs.from_profile(profile, pa, pv, r_micro, n=self.n_points)
        s = np.linspace(profile.s[0], profile.s[-1], self.n_points)
        return s, self.predict(raw)[0]
