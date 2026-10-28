"""Locating exported models (``models/<kind>/<run-name>/``)."""

from pathlib import Path

SURROGATE_MODELS = Path("models/surrogate")
SEG_MODELS = Path("models/seg")


def latest_model_dir(root: str | Path, filename: str) -> Path | None:
    """Most recently modified ``root/<run>/`` that contains ``filename``, or ``None``."""
    dirs = [p.parent for p in Path(root).glob(f"*/{filename}")]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None
