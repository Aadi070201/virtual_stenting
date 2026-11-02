"""The learned lesion reader in ONNX Runtime (P4i). Torch-free.

``models/lesion/<run>/lesion.onnx`` is the mean of the 5 fold models (inputs: patches
(n, 2, 64, 32), scalars (n, 3); outputs: %DS, significance logit). Its inputs come from
:func:`vstent.imaging.lesion_patch.lesion_patch`, the same function as in training.
"""

from pathlib import Path

import numpy as np

ONNX_FILE = "lesion.onnx"
LESION_MODELS = Path("models/lesion")


class LesionRunner:
    """Predicted %DS of lesion patches."""

    def __init__(self, path: str | Path):
        import onnxruntime as ort

        self.path = Path(path)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "LesionRunner":
        return cls(Path(model_dir) / ONNX_FILE)

    def predict(self, patches: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        """(n,) %DS."""
        if len(patches) == 0:
            return np.zeros(0)
        return self.predict_both(patches, scalars)[0]

    def predict_both(self, patches: np.ndarray, scalars: np.ndarray):
        """(n,) %DS and (n,) probability of a significant lesion (≥ 50%)."""
        feeds = {"patch": np.asarray(patches, np.float32),
                 "scalars": np.asarray(scalars, np.float32)}  # fmt: skip
        ds, logit = self.session.run(["ds", "sig_logit"], feeds)
        return ds, 1.0 / (1.0 + np.exp(-logit))
