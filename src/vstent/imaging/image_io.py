"""Reading angiogram images (PNG / JPG / DICOM) into greyscale uint8. Torch-free.

The single image loader for the app and ``vstent analyse``, so both segment exactly the same
pixels. Images that are not 8-bit (e.g. 12- or 16-bit DICOM) are stretched between their 0.5th
and 99.5th percentiles.
"""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ImageInput:
    """A greyscale angiogram and what is known about its pixel size."""

    image: np.ndarray  # (H, W) uint8
    name: str
    sha256: str
    pixel_spacing_mm: float | None = None  # from DICOM ImagerPixelSpacing, if present
    note: str = ""


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Array → uint8: 8-bit unchanged, otherwise stretched between the 0.5th and 99.5th
    percentiles."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float64)
    lo, hi = np.percentile(arr, (0.5, 99.5))
    return np.clip((arr - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)


def read_image(data: bytes, name: str, frame: int | None = None) -> ImageInput:
    """PNG / JPG / DICOM bytes → greyscale uint8 image. ``frame``: the frame of a multi-frame
    DICOM (0-based; default the first). A single-frame image only has frame 0."""
    digest = hashlib.sha256(data).hexdigest()
    is_dicom = name.lower().endswith((".dcm", ".dicom")) or data[128:132] == b"DICM"
    if is_dicom:
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        arr = ds.pixel_array
        frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        _check_frame(name, frame, frames)
        if frames > 1:
            arr = arr[frame or 0]
        if arr.ndim == 3:  # colour
            arr = cv2.cvtColor(to_uint8(arr), cv2.COLOR_RGB2GRAY)
        spacing = ds.get("ImagerPixelSpacing")
        mm = float(spacing[0]) if spacing else None
        note = (
            "DICOM ImagerPixelSpacing is measured at the detector; geometric magnification is "
            "not corrected."
            if mm
            else "DICOM without ImagerPixelSpacing."
        )
        if frames > 1:
            note += f" Frame {frame or 0} of {frames} used (0-based)."
        return ImageInput(to_uint8(arr), name, digest, mm, note)
    _check_frame(name, frame, 1)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"{name}: not a readable PNG / JPG / DICOM image")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return ImageInput(to_uint8(img), name, digest)


def read_image_file(path: str | Path, frame: int | None = None) -> ImageInput:
    """:func:`read_image` for a file on disk."""
    path = Path(path)
    return read_image(path.read_bytes(), path.name, frame)


def _check_frame(name: str, frame: int | None, frames: int) -> None:
    if frame is not None and not 0 <= frame < frames:
        raise ValueError(f"{name}: frame {frame} requested, the image has {frames} frame(s)")
