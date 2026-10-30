"""ONNX Runtime inference for the tree FFR surrogate (P3; no torch needed).

Inputs go through :func:`vstent.surrogate.tree_features.prepare_tree_inputs`, the same
pre-processing as in training. A model directory (``models/tree_surrogate/<run-name>/``) holds
``tree_surrogate.onnx`` and ``tree_norm_stats.json``; the ONNX file's metadata is checked when
it is loaded. Large batches run as small chunks on parallel single-threaded calls, as for the
single-vessel surrogate (DECISIONS D17).
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort

from vstent.inference.onnx_runtime import default_workers
from vstent.physics.tree import VesselTree
from vstent.surrogate.tree_features import (
    TREE_NORM_STATS,
    TreeNormStats,
    TreeRawInputs,
    prepare_tree_inputs,
)

TREE_ONNX_FILE = "tree_surrogate.onnx"
TREE_MODELS = Path("models/tree_surrogate")
TREE_INPUT_NAMES = ("x_point", "x_branch", "parent", "attach", "mask")
TREE_OUTPUT = "ffr"
TREE_CHUNK = 4  # trees per ONNX call when a batch is split across threads


class TreeSurrogateRunner:
    """FFR at every sample of every branch, predicted by an exported tree surrogate.

    Args:
        model_path: ``tree_surrogate.onnx``.
        stats_path: normalisation statistics; default ``tree_norm_stats.json`` next to it.
        workers: parallel chunk workers; ``None`` = all logical CPUs.
        chunk_size: trees per ONNX call; ``None`` = the whole batch in one call.
        providers: ONNX Runtime execution providers.
    """

    def __init__(
        self,
        model_path: str | Path,
        stats_path: str | Path | None = None,
        *,
        workers: int | None = None,
        chunk_size: int | None = TREE_CHUNK,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.model_path = Path(model_path)
        self.stats = TreeNormStats.load(stats_path or self.model_path.with_name(TREE_NORM_STATS))
        self.workers = default_workers() if workers is None else workers
        self.chunk_size = chunk_size
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.workers > 1:
            opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=list(providers))
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        for key, value in {
            "tree_feature_version": str(self.stats.feature_version),
            "n_points": str(self.stats.n_points),
            "max_branches": str(self.stats.max_branches),
        }.items():
            if self.metadata.get(key) != value:
                raise ValueError(
                    f"{self.model_path}: {key} = {self.metadata.get(key)!r} in the model, "
                    f"{value!r} in the normalisation statistics"
                )
        self._pool = ThreadPoolExecutor(self.workers) if self.workers > 1 else None

    @classmethod
    def from_dir(cls, model_dir: str | Path, **kwargs) -> "TreeSurrogateRunner":
        """Load ``tree_surrogate.onnx`` + ``tree_norm_stats.json`` from a model directory."""
        return cls(Path(model_dir) / TREE_ONNX_FILE, **kwargs)

    @property
    def n_points(self) -> int:
        return self.stats.n_points

    @property
    def max_branches(self) -> int:
        return self.stats.max_branches

    def close(self) -> None:
        """Stop the worker threads (the runner can still predict, serially)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _run(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        feeds = {
            name: np.ascontiguousarray(x, dtype=np.float32)
            for name, x in zip(TREE_INPUT_NAMES, inputs, strict=True)
        }
        return self.session.run([TREE_OUTPUT], feeds)[0]

    def predict_features(self, *inputs: np.ndarray) -> np.ndarray:
        """FFR (B, M, N), float32, from the five standardised input arrays."""
        n = inputs[0].shape[0]
        if self.chunk_size is None or n <= self.chunk_size:
            return self._run(inputs)
        starts = range(0, n, self.chunk_size)

        def chunk(i: int) -> np.ndarray:
            return self._run([x[i : i + self.chunk_size] for x in inputs])

        mapper = self._pool.map if self._pool is not None else map
        return np.concatenate(list(mapper(chunk, starts)))

    def predict(self, raw: TreeRawInputs) -> np.ndarray:
        """FFR (B, M, N) for trees in SI units (see ``TreeRawInputs``); padded branches are
        meaningless (use ``raw.mask``)."""
        return self.predict_features(*prepare_tree_inputs(raw, self.stats))

    def predict_tree(
        self, tree: VesselTree, pa: float, pv: float, r_outlet: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """One tree: positions (m) and FFR of every branch on the model's N-point grid.

        Pressures in Pa; ``r_outlet`` (Pa·s/m³) one per branch, e.g. from the tree solver's
        calibration. Raises ``ValueError`` for a tree with more branches than the model takes.
        """
        raw = TreeRawInputs.from_trees([tree], pa, pv, [r_outlet], n=self.n_points,
                                       max_branches=self.max_branches)  # fmt: skip
        ffr = self.predict(raw)[0]
        s = [np.linspace(0.0, raw.length[0, k], self.n_points) for k in range(tree.n_branches)]
        return s, [ffr[k] for k in range(tree.n_branches)]


def floor_warnings(ffr: list[np.ndarray], names: Sequence[str], pa: float, pv: float) -> list[str]:
    """Branches whose surrogate FFR falls below the physical minimum Pv/Pa (zero flow): the
    tree is far from what the model learned (the solver's value is the one to trust)."""
    floor = pv / pa
    return [
        f"Tree surrogate FFR in {name} reaches {f.min():.2f}, below the physical minimum "
        f"Pv/Pa = {floor:.2f}: this tree is far from what the model learned"
        for name, f in zip(names, ffr, strict=True)
        if f.min() < floor
    ]
