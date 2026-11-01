"""Reading angiogram images (PNG / JPG / DICOM) into greyscale uint8. Torch-free.

The single image loader for the app and ``vstent analyse``, so both segment exactly the same
pixels. Images that are not 8-bit (e.g. 12- or 16-bit DICOM) are stretched between their 0.5th
and 99.5th percentiles.

DICOM pixel size (P5): ImagerPixelSpacing is measured at the detector, where the vessels are
magnified by SID / SOD. The pixel size used for measurement is corrected to the isocentre
(:mod:`vstent.imaging.projection`); without SID / SOD the detector spacing is kept for the
record but not offered as a calibration.
"""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vstent.imaging.projection import DicomProjection, ProjectionGeometry, read_projection


@dataclass
class ImageInput:
    """A greyscale angiogram and what is known about its pixel size."""

    image: np.ndarray  # (H, W) uint8
    name: str
    sha256: str
    pixel_spacing_mm: float | None = None  # DICOM, magnification-corrected to the isocentre
    note: str = ""
    detector_spacing_mm: float | None = None  # DICOM ImagerPixelSpacing (at the detector)
    geometry: ProjectionGeometry | None = None  # DICOM projection geometry (P5), if complete


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
        proj = read_projection(ds)
        note = proj.note
        if frames > 1:
            note += f" Frame {frame or 0} of {frames} used (0-based)."
        return ImageInput(to_uint8(arr), name, digest, proj.iso_spacing_mm, note,
                          proj.detector_spacing_mm, proj.geometry)  # fmt: skip
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


FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass
class RunInput:
    """A cine run: greyscale frames on one intensity scale (P4d). Frame-wise stretching would
    distort the contrast's arrival and washout, so non-8-bit runs are stretched once."""

    frames: np.ndarray  # (T, H, W) uint8
    name: str
    sha256: str
    frame_time_ms: float | None = None  # DICOM FrameTime (or 1000 / CineRate), if present
    pixel_spacing_mm: float | None = None  # magnification-corrected, as in ImageInput
    note: str = ""
    detector_spacing_mm: float | None = None
    geometry: ProjectionGeometry | None = None

    def __len__(self) -> int:
        return len(self.frames)

    def frame(self, t: int) -> ImageInput:
        """Frame ``t`` (0-based) as a single image for the single-frame pipeline."""
        if not 0 <= t < len(self.frames):
            raise ValueError(f"{self.name}: frame {t} requested, the run has {len(self)} frames")
        img = np.ascontiguousarray(self.frames[t])
        note = f"Frame {t} of {len(self)} (0-based). {self.note}".strip()
        return ImageInput(img, f"{self.name} frame {t}", hashlib.sha256(img.tobytes()).hexdigest(),
                          self.pixel_spacing_mm, note, self.detector_spacing_mm,
                          self.geometry)  # fmt: skip


def is_run(path: str | Path) -> bool:
    """A cine run: a folder of frames, or a DICOM file with more than one frame."""
    path = Path(path)
    if path.is_dir():
        return True
    return is_multiframe_dicom(path.read_bytes(), path.name)


def read_run(path: str | Path) -> RunInput:
    """A cine run from a multi-frame DICOM file or a folder of frame images (sorted by file
    name; all the same size)."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in FRAME_SUFFIXES)
        if not files:
            raise ValueError(f"{path}: no frame images ({', '.join(FRAME_SUFFIXES)})")
        digest, frames = hashlib.sha256(), []
        for f in files:
            data = f.read_bytes()
            digest.update(data)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"{f}: not a readable image")
            if img.ndim == 3:
                code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
                img = cv2.cvtColor(img, code)
            frames.append(img)
        if len({f.shape for f in frames}) > 1:
            raise ValueError(f"{path}: frames of different sizes")
        note = "Frame folder: frame time unknown."
        return RunInput(to_uint8(np.stack(frames)), path.name, digest.hexdigest(), note=note)
    return read_run_bytes(path.read_bytes(), path.name)


def is_multiframe_dicom(data: bytes, name: str) -> bool:
    """DICOM bytes with more than one frame (a cine run)."""
    if not (name.lower().endswith((".dcm", ".dicom")) or data[128:132] == b"DICM"):
        return False
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)
    return int(getattr(ds, "NumberOfFrames", 1) or 1) > 1


def read_run_bytes(data: bytes, name: str) -> RunInput:
    """A cine run from DICOM bytes (one intensity scale for all frames)."""
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array
    if int(getattr(ds, "NumberOfFrames", 1) or 1) == 1:
        arr = arr[None]
    if arr.ndim == 4:  # colour frames
        arr = np.stack([cv2.cvtColor(to_uint8(f), cv2.COLOR_RGB2GRAY) for f in arr])
    frame_time = ds.get("FrameTime")
    rate = ds.get("CineRate") or ds.get("RecommendedDisplayFrameRate")
    ms = float(frame_time) if frame_time else (1000.0 / float(rate) if rate else None)
    proj: DicomProjection = read_projection(ds)
    note = (
        proj.note if ms else f"DICOM without FrameTime / CineRate: frame time unknown. {proj.note}"
    )
    return RunInput(to_uint8(arr), name, hashlib.sha256(data).hexdigest(), ms,
                    proj.iso_spacing_mm, note, proj.detector_spacing_mm,
                    proj.geometry)  # fmt: skip
