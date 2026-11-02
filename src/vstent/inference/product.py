"""The product's FFR pipeline settings (``configs/product.yaml``): which learned lesion reader
measures the lesions' depth, which outflow model the solver uses, and whether the FFR is read
where a pressure wire would sit. Kept apart from ``configs/physics.yaml``, which also drives
the synthetic data and the physics checks. Each setting changes only through a pre-registered
decision (docs/DECISIONS.md). Torch-free.
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
