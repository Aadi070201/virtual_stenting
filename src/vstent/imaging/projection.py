"""C-arm projection geometry (P5): DICOM positioner angles and distances → where a 3D point
appears in the image, the ray through a pixel, and the pixel size at a point. Torch-free.

**Coordinates.** The DICOM patient system (PS3.3 C.7.6.2.1.1): x towards the patient's left,
y posterior, z towards the head; metres; the isocentre at the origin.

**Angles** (PS3.3 C.8.7.5.1.2) give the position of the image detector:
- Positioner Primary Angle α, in the transaxial plane: 0 perpendicular to the chest (detector
  anterior), +90° at the patient's left (LAO), −90° at the right (RAO);
- Positioner Secondary Angle β, in the sagittal plane: +90° cranial (CRA), negative caudal.

The standard gives no rotation order. Here (D49) the detector direction from the isocentre is
u = (sin α cos β, −cos α cos β, sin β): the primary rotation about the patient's long axis, then
the secondary towards the head about the rotated horizontal axis. This is to be confirmed on
real two-view DICOM; a mismatch shows as a large reprojection residual
(:mod:`vstent.imaging.twoview`).

**Image axes.** At α = β = 0 the patient faces the detector: columns run towards the patient's
left and rows towards the feet, as angiograms are displayed. The axes turn with the detector.
The principal point is the image centre, moved by ``shift_px`` (e.g. patient motion between two
runs). A flat-panel detector is assumed (no image-intensifier distortion).

**Pixel size at a point.** A point at distance d from the source along the central ray is
magnified by SID / d, so one detector pixel spans ``pixel × d / SID`` there. At the isocentre
(d = SOD) this is ``ImagerPixelSpacing × SOD / SID``: the magnification-corrected pixel size of
a single view, exact only for structures at the isocentre's depth.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from vstent.units import mm_to_m


@dataclass(frozen=True)
class ProjectionGeometry:
    """One X-ray projection.

    Attributes:
        primary_deg: Positioner Primary Angle (degrees; + LAO, − RAO).
        secondary_deg: Positioner Secondary Angle (degrees; + cranial, − caudal).
        sid_m: source-to-detector distance (m).
        sod_m: source-to-isocentre distance (m).
        pixel_m: detector pixel spacing (row, col) (m).
        shape: image size (rows, cols).
        shift_px: principal point offset from the image centre (row, col) (px).
    """

    primary_deg: float
    secondary_deg: float
    sid_m: float
    sod_m: float
    pixel_m: tuple[float, float]
    shape: tuple[int, int]
    shift_px: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not 0.0 < self.sod_m < self.sid_m:
            raise ValueError(f"need 0 < SOD < SID, got SOD {self.sod_m} m, SID {self.sid_m} m")
        if len(self.pixel_m) != 2 or min(self.pixel_m) <= 0.0:
            raise ValueError("pixel_m must be two positive spacings (row, col)")
        if len(self.shape) != 2 or min(self.shape) < 2:
            raise ValueError("shape must be (rows, cols)")
        object.__setattr__(self, "pixel_m", tuple(float(v) for v in self.pixel_m))
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        object.__setattr__(self, "shift_px", tuple(float(v) for v in self.shift_px))

    @property
    def magnification(self) -> float:
        """SID / SOD: the magnification at the isocentre."""
        return self.sid_m / self.sod_m

    @property
    def iso_pixel_m(self) -> float:
        """Pixel size at the isocentre (m/px; mean of rows and columns)."""
        return float(np.mean(self.pixel_m)) / self.magnification

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

    def _centre_px(self) -> tuple[float, float]:
        return (
            (self.shape[0] - 1) / 2.0 + self.shift_px[0],
            (self.shape[1] - 1) / 2.0 + self.shift_px[1],
        )

    def depth(self, points) -> np.ndarray:
        """(n,) distance (m) from the source along the central ray of (n, 3) points (m)."""
        return (np.atleast_2d(np.asarray(points, dtype=float)) - self.source) @ self.axes()[0]

    def project(self, points) -> np.ndarray:
        """(n, 3) points (m) → (n, 2) image positions (row, col) (px)."""
        u, down, right = self.axes()
        d = np.atleast_2d(np.asarray(points, dtype=float)) - self.source
        depth = d @ u
        if np.any(depth <= 0.0):
            raise ValueError("a point lies behind the X-ray source")
        scale = self.sid_m / depth
        r0, c0 = self._centre_px()
        rows = r0 + (d @ down) * scale / self.pixel_m[0]
        cols = c0 + (d @ right) * scale / self.pixel_m[1]
        return np.stack([rows, cols], axis=1)

    def rays(self, pixels) -> np.ndarray:
        """(n, 2) image positions (row, col) (px) → (n, 3) unit ray directions from the
        source."""
        u, down, right = self.axes()
        p = np.atleast_2d(np.asarray(pixels, dtype=float))
        r0, c0 = self._centre_px()
        v = (
            self.sid_m * u
            + ((p[:, 0] - r0) * self.pixel_m[0])[:, None] * down
            + ((p[:, 1] - c0) * self.pixel_m[1])[:, None] * right
        )
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def pixel_size_at(self, points) -> np.ndarray:
        """(n,) pixel size (m/px) at (n, 3) points (m): mean detector spacing × depth / SID."""
        return float(np.mean(self.pixel_m)) * self.depth(points) / self.sid_m

    def separation_deg(self, other: "ProjectionGeometry") -> float:
        """Angle (degrees) between the central rays of two views."""
        c = float(np.clip(self.axes()[0] @ other.axes()[0], -1.0, 1.0))
        return math.degrees(math.acos(c))

    def shifted(self, shift_px) -> "ProjectionGeometry":
        """The same view with the principal point moved by ``shift_px`` (row, col) (px)."""
        return replace(self, shift_px=(float(shift_px[0]), float(shift_px[1])))

    def label(self) -> str:
        """Angles as clinicians name them, e.g. ``LAO 30° CRA 20°``."""
        a, b = self.primary_deg, self.secondary_deg
        side = "LAO" if a > 0 else "RAO" if a < 0 else "AP"
        tilt = "CRA" if b > 0 else "CAU" if b < 0 else ""
        first = f"{side} {abs(a):.0f}°" if side != "AP" else "AP"
        return f"{first} {tilt} {abs(b):.0f}°" if tilt else first


@dataclass(frozen=True)
class DicomProjection:
    """What a DICOM image says about its projection.

    Attributes:
        detector_spacing_mm: ImagerPixelSpacing (mean of row and column; at the detector).
        magnification: SID / SOD (or EstimatedRadiographicMagnificationFactor).
        geometry: the full projection, when the angles and distances are all present.
        note: plain-language summary for reports.
    """

    detector_spacing_mm: float | None
    magnification: float | None
    geometry: ProjectionGeometry | None
    note: str

    @property
    def iso_spacing_mm(self) -> float | None:
        """Magnification-corrected pixel size at the isocentre (mm/px), if known."""
        if self.detector_spacing_mm is None or self.magnification is None:
            return None
        return self.detector_spacing_mm / self.magnification


def _number(ds, name: str) -> float | None:
    value = ds.get(name)
    try:
        out = float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None
    return out if out is not None and math.isfinite(out) else None


def read_projection(ds) -> DicomProjection:
    """Projection information of a pydicom dataset (single or multi-frame; for a run the
    angles of the first frame, as DICOM defines them).

    Uses ImagerPixelSpacing (0018,1164), DistanceSourceToDetector (0018,1110),
    DistanceSourceToPatient (0018,1111; the isocentre on cardiovascular systems),
    EstimatedRadiographicMagnificationFactor (0018,1114; when SOD is missing),
    PositionerPrimaryAngle (0018,1510), PositionerSecondaryAngle (0018,1511), Rows and Columns.
    Distances in the file are mm; the geometry is SI.
    """
    spacing = ds.get("ImagerPixelSpacing")
    pix = None
    if spacing is not None:
        try:
            values = [float(v) for v in (spacing if hasattr(spacing, "__len__") else [spacing])]
        except (TypeError, ValueError):
            values = []
        if values and all(math.isfinite(v) and v > 0 for v in values):
            pix = (values[0], values[1] if len(values) > 1 else values[0])
    sid = _number(ds, "DistanceSourceToDetector")
    sod = _number(ds, "DistanceSourceToPatient")
    emf = _number(ds, "EstimatedRadiographicMagnificationFactor")
    if sid and emf and emf > 1.0 and not sod:
        sod = sid / emf
    magnification = None
    if sid and sod and 0.0 < sod < sid:
        magnification = sid / sod
    elif emf and emf > 1.0:
        magnification = emf
    primary = _number(ds, "PositionerPrimaryAngle")
    secondary = _number(ds, "PositionerSecondaryAngle")
    rows, cols = ds.get("Rows"), ds.get("Columns")
    geometry = None
    if pix and magnification and sid and sod and None not in (primary, secondary) and rows and cols:
        geometry = ProjectionGeometry(
            primary,
            secondary,
            mm_to_m(sid),
            mm_to_m(sod),
            (mm_to_m(pix[0]), mm_to_m(pix[1])),
            (int(rows), int(cols)),
        )
    detector = float(np.mean(pix)) if pix else None
    if detector is None:
        note = "DICOM without ImagerPixelSpacing: pixel size unknown."
    elif magnification is None:
        note = (
            f"DICOM ImagerPixelSpacing {detector:.4f} mm is measured at the detector; without "
            "SID / SOD the geometric magnification cannot be corrected, so it is not used."
        )
    else:
        note = (
            f"Pixel size {detector / magnification:.4f} mm/px at the isocentre: "
            f"ImagerPixelSpacing {detector:.4f} mm ÷ magnification {magnification:.3f} "
            "(SID / SOD). Exact only at the isocentre's depth."
        )
    if geometry is not None:
        note += f" Projection {geometry.label()}, SID {sid:.0f} mm, SOD {sod:.0f} mm."
    return DicomProjection(detector, magnification, geometry, note)
