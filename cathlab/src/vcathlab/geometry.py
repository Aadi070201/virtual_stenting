"""The virtual C-arm: positioner angles and distances → where a 3D point appears in the image,
and the pixel size at a point.

Copied from vstent's ``imaging/projection.py`` (P5, convention D49) on 2026-11-04 and trimmed;
it evolves here on its own (docs/DECISIONS.md, V1). Using the same convention means a DICOM
written here is read by vstent as intended; the convention itself is still to be confirmed on
real two-view DICOM.

**Coordinates.** The DICOM patient system: x towards the patient's left, y posterior, z
towards the head; metres; the isocentre at the origin.

**Angles** give the detector's position: primary α in the transaxial plane (0: detector
anterior, + LAO, − RAO); secondary β towards the head (+ cranial, − caudal). The detector
direction from the isocentre is u = (sin α cos β, −cos α cos β, sin β).

**Image axes.** At α = β = 0 columns run towards the patient's left and rows towards the feet.
The principal point is the image centre moved by ``shift_px`` (the table pan that frames the
artery).

**Pixel size at a point** at distance d from the source along the central ray: detector pixel ×
d / SID; at the isocentre (d = SOD) detector pixel × SOD / SID.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CArm:
    """One projection.

    Attributes:
        primary_deg: primary angle (degrees; + LAO, − RAO).
        secondary_deg: secondary angle (degrees; + cranial, − caudal).
        sid_m: source-to-detector distance (m).
        sod_m: source-to-isocentre distance (m).
        pixel_m: detector pixel spacing (m), the same for rows and columns.
        shape: image size (rows, cols).
        shift_px: principal point offset from the image centre (row, col) (px).
    """

    primary_deg: float
    secondary_deg: float
    sid_m: float
    sod_m: float
    pixel_m: float
    shape: tuple[int, int] = (512, 512)
    shift_px: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not 0.0 < self.sod_m < self.sid_m:
            raise ValueError(f"need 0 < SOD < SID, got SOD {self.sod_m} m, SID {self.sid_m} m")
        if self.pixel_m <= 0.0:
            raise ValueError("pixel_m must be positive")

    @property
    def iso_pixel_m(self) -> float:
        """Pixel size at the isocentre (m/px): detector pixel × SOD / SID."""
        return self.pixel_m * self.sod_m / self.sid_m

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit vectors: isocentre → detector, image down (rows), image right (columns)."""
        a, b = math.radians(self.primary_deg), math.radians(self.secondary_deg)
        u = np.array([math.sin(a) * math.cos(b), -math.cos(a) * math.cos(b), math.sin(b)])
        down = np.array([math.sin(a) * math.sin(b), -math.cos(a) * math.sin(b), -math.cos(b)])
        right = np.array([math.cos(a), math.sin(a), 0.0])
        return u, down, right

    @property
    def source(self) -> np.ndarray:
        """X-ray source position (m)."""
        return -self.sod_m * self.axes()[0]

    def depth(self, points) -> np.ndarray:
        """(n,) distance (m) from the source along the central ray of (n, 3) points (m)."""
        return (np.atleast_2d(np.asarray(points, float)) - self.source) @ self.axes()[0]

    def project(self, points) -> np.ndarray:
        """(n, 3) points (m) → (n, 2) image positions (row, col) (px)."""
        u, down, right = self.axes()
        d = np.atleast_2d(np.asarray(points, float)) - self.source
        depth = d @ u
        if np.any(depth <= 0.0):
            raise ValueError("a point lies behind the X-ray source")
        scale = self.sid_m / depth / self.pixel_m
        r0, c0 = self.centre_px()
        return np.stack([r0 + (d @ down) * scale, c0 + (d @ right) * scale], axis=1)

    def centre_px(self) -> tuple[float, float]:
        """The principal point (row, col) (px)."""
        return (
            (self.shape[0] - 1) / 2.0 + self.shift_px[0],
            (self.shape[1] - 1) / 2.0 + self.shift_px[1],
        )

    def ray_directions(self, points) -> np.ndarray:
        """(n, 3) unit directions of the rays from the source through (n, 3) points (m)."""
        d = np.atleast_2d(np.asarray(points, float)) - self.source
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def pixel_size_at(self, points) -> np.ndarray:
        """(n,) pixel size (m/px) at (n, 3) points (m): detector pixel × depth / SID."""
        return self.pixel_m * self.depth(points) / self.sid_m

    def label(self) -> str:
        """Angles as clinicians name them, e.g. ``LAO 30° CRA 20°``."""
        a, b = self.primary_deg, self.secondary_deg
        side = "LAO" if a > 0 else "RAO" if a < 0 else "AP"
        tilt = "CRA" if b > 0 else "CAU" if b < 0 else ""
        first = f"{side} {abs(a):.0f}°" if side != "AP" else "AP"
        return f"{first} {tilt} {abs(b):.0f}°" if tilt else first
