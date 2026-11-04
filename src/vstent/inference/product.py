"""The product's FFR pipeline settings (``configs/product.yaml``): which learned lesion reader
measures the lesions' depth, which outflow model the solver uses, whether the FFR is read where
a pressure wire would sit, and whether a lesion detector must confirm each narrowing (P4q).
Kept apart from ``configs/physics.yaml``, which also drives the synthetic data and the physics
checks. Each setting changes only through a pre-registered decision (docs/DECISIONS.md).
Torch-free.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from vstent.config import StrictModel, load_config
from vstent.physics.solver import PhysicsParams

PRODUCT_CONFIG = Path("configs/product.yaml")


class ProductConfig(StrictModel):
    """The product's FFR pipeline (defaults: the product before P4k-C3)."""

    lesion_reader: str | None = None  # an exported reader directory (models/lesion/<run>)
    outflow: Literal["none", "murray"] = "none"
    wire: bool = True  # D68
    lesion_min_ref_mm: float | None = None  # P4n: the learned depth only in vessels ≥ this
    lesion_detector: tuple[str, ...] | None = None  # P4q: detector dirs (the mean of their maps)
    detector_threshold: float = 0.5  # P4q: detections are the pixels ≥ this


@lru_cache(maxsize=4)
def product_config(path: str | Path = PRODUCT_CONFIG) -> ProductConfig:
    """``configs/product.yaml``, or the defaults if it is missing."""
    path = Path(path)
    return load_config(path, ProductConfig) if path.exists() else ProductConfig()


def product_params(params: PhysicsParams, cfg: ProductConfig | None = None) -> PhysicsParams:
    """``params`` with the product's outflow model."""
    cfg = cfg or product_config()
    return params.model_copy(update={"outflow": cfg.outflow})


@lru_cache(maxsize=4)
def product_reader(reader_dir: str | None):
    """The product's lesion reader (loaded once), or ``None``."""
    if not reader_dir:
        return None
    from vstent.inference.lesion_runtime import LesionRunner

    return LesionRunner.from_dir(reader_dir)


@lru_cache(maxsize=4)
def lesion_gate(detector_dirs: tuple[str, ...] | None, threshold: float):
    """P4q's gate: the mean of the detectors' lesion maps, detections ≥ ``threshold`` (loaded
    once), or ``None`` without detectors."""
    if not detector_dirs:
        return None
    import numpy as np

    from vstent.imaging.lesion_gate import LesionGate
    from vstent.inference.onnx_runtime import MapRunner

    runners = [MapRunner.from_dir(d) for d in detector_dirs]

    def predict(image: np.ndarray) -> np.ndarray:
        return np.mean([r.predict(image)[0] for r in runners], axis=0)

    return LesionGate(predict, threshold)


def product_gate(cfg: ProductConfig | None = None):
    """The product's lesion gate (P4q), or ``None``."""
    cfg = cfg or product_config()
    return lesion_gate(cfg.lesion_detector, cfg.detector_threshold)
