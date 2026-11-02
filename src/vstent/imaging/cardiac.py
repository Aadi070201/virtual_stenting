"""End-diastolic (ED) frames of a cine run (P7). Torch-free.

End-diastole is the ECG R wave: the ventricles are fullest, so the heart and the epicardial
coronary tree on it are largest. QCA is done on ED frames, and two runs analysed at their ED
frames show the heart at the same phase (two-view reconstruction, P5). Sources, in order:

1. the DICOM R-wave pointer (0028,6040): the frame numbers of the R waves;
2. an ECG waveform in the DICOM (WaveformSequence): R peaks (:func:`r_peaks`) mapped to
   frames with FrameDelay / FrameTime and the waveform's MultiplexGroupTimeOffset;
3. the images (:func:`ed_from_spread`): the spread of the segmented vessel tree (RMS distance
   of vessel pixels from their centroid) follows the heart's size. Over the well-opacified
   frames it is detrended, its period is found by autocorrelation, and its maxima, one per
   beat, are the ED frames.

Frames are 0-based; times are ms from the first frame unless named.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from vstent.imaging.cine import Opacification

ECG_BAND_HZ = (5.0, 15.0)  # QRS band of the R-peak detector
ECG_WINDOW_S = 0.15  # moving-window integration of the squared derivative
ECG_REFRACTORY_S = 0.25  # no two R waves closer than this
ECG_HEIGHT = 0.3  # integrated energy above this share of its 99th percentile
ECG_REFINE_S = 0.08  # the R wave: the extreme of the band-passed ECG this close to the peak
ED_AREA_FRACTION = 0.5  # image-based: frames whose smoothed area is ≥ this share of the peak
MIN_BPM, MAX_BPM = 40.0, 150.0  # heart rates searched when the frame time is known
MIN_PERIOD_FRAMES = 3  # and when it is not
MIN_SPREAD_PIXELS = 50  # fewer vessel pixels: no spread for that frame
PEAK_DISTANCE = 0.6  # ED maxima at least this share of the period apart


@dataclass(frozen=True)
class EcgSignal:
    """One ECG channel of a run: ``values`` (the file's units × channel sensitivity), sampled
    at ``fs_hz``, the first sample at ``start_ms`` from the first frame."""

    values: np.ndarray
    fs_hz: float
    start_ms: float

    def times_ms(self, idx: np.ndarray) -> np.ndarray:
        return self.start_ms + np.asarray(idx, float) * 1000.0 / self.fs_hz


@dataclass(frozen=True)
class EndDiastole:
    """End-diastolic frames of a run.

    Attributes:
        frames: ED frame indices, increasing.
        source: ``"R-wave pointer"``, ``"ECG"``, ``"images"`` or ``"none"``.
        period_frames: beat length (frames) if known.
        note: how they were found, or why not.
    """

    frames: tuple[int, ...]
    source: str
    period_frames: float | None
    note: str

    @property
    def found(self) -> bool:
        return bool(self.frames)


def _number(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def r_wave_pointer(ds, n_frames: int) -> tuple[int, ...] | None:
    """R-wave frames from the DICOM R Wave Pointer (0028,6040), or ``None``. DICOM numbers
    these frames from 0."""
    value = ds.get("RWavePointer")
    if value is None:
        return None
    values = list(value) if hasattr(value, "__iter__") else [value]
    frames = sorted({int(v) for v in values if _number(v) is not None and 0 <= int(v) < n_frames})
    return tuple(frames) or None


_DTYPES = {("SS", 16): "<i2", ("US", 16): "<u2", ("SB", 8): "i1", ("UB", 8): "u1",
           ("SL", 32): "<i4", ("UL", 32): "<u4"}  # fmt: skip


def _ecg_channel(definitions) -> tuple[int, float] | None:
    """Index and sensitivity of the first channel whose source names an ECG lead."""
    for k, ch in enumerate(definitions or []):
        meaning = " ".join(
            str(src.get("CodeMeaning", "")) for src in (ch.get("ChannelSourceSequence") or [])
        ).lower()
        if "lead" in meaning or "ecg" in meaning:
            return k, _number(ch.get("ChannelSensitivity")) or 1.0
    return None


def read_ecg(ds) -> EcgSignal | None:
    """The first ECG channel of the DICOM WaveformSequence, or ``None``. Its time origin is
    the first frame: MultiplexGroupTimeOffset − FrameDelay (both from the acquisition's
    reference time; 0 if absent)."""
    for item in ds.get("WaveformSequence") or []:
        n_ch = int(_number(item.get("NumberOfWaveformChannels")) or 0)
        n = int(_number(item.get("NumberOfWaveformSamples")) or 0)
        fs = _number(item.get("SamplingFrequency")) or 0.0
        dtype = _DTYPES.get(
            (str(item.get("WaveformSampleInterpretation", "SS")),
             int(_number(item.get("WaveformBitsAllocated")) or 16))
        )  # fmt: skip
        channel = _ecg_channel(item.get("ChannelDefinitionSequence"))
        raw = item.get("WaveformData")
        if not (n_ch and n and fs > 0 and dtype and channel and raw is not None):
            continue
        data = np.frombuffer(bytes(raw), dtype=dtype)
        if data.size < n * n_ch:
            continue
        k, sensitivity = channel
        values = data[: n * n_ch].reshape(n, n_ch)[:, k].astype(float) * sensitivity
        start = (_number(item.get("MultiplexGroupTimeOffset")) or 0.0) - (
            _number(ds.get("FrameDelay")) or 0.0
        )
        return EcgSignal(values, float(fs), float(start))
    return None


def r_peaks(values: np.ndarray, fs_hz: float) -> np.ndarray:
    """Sample indices of the R waves: band-pass ``ECG_BAND_HZ``, squared derivative
    integrated over ``ECG_WINDOW_S``, peaks at least ``ECG_REFRACTORY_S`` apart above
    ``ECG_HEIGHT`` of its 99th percentile, each moved to the extreme of the band-passed
    signal within ``ECG_REFINE_S`` (on the side, positive or negative, where the QRS is
    larger)."""
    x = np.asarray(values, float)
    if x.size < int(fs_hz):
        return np.zeros(0, int)
    nyq = fs_hz / 2.0
    b, a = butter(2, [ECG_BAND_HZ[0] / nyq, min(ECG_BAND_HZ[1] / nyq, 0.99)], btype="band")
    f = filtfilt(b, a, x - np.median(x))
    w = max(int(ECG_WINDOW_S * fs_hz), 1)
    energy = np.convolve(np.gradient(f) ** 2, np.ones(w) / w, mode="same")
    peaks, _ = find_peaks(energy, height=ECG_HEIGHT * np.percentile(energy, 99),
                          distance=max(int(ECG_REFRACTORY_S * fs_hz), 1))  # fmt: skip
    sign = 1.0 if np.percentile(f, 99.5) >= -np.percentile(f, 0.5) else -1.0
    half = max(int(ECG_REFINE_S * fs_hz), 1)
    out = []
    for p in peaks:
        lo, hi = max(p - half, 0), min(p + half + 1, f.size)
        out.append(lo + int(np.argmax(sign * f[lo:hi])))
    return np.unique(np.asarray(out, int))


def frames_at(times_ms: np.ndarray, n_frames: int, frame_time_ms: float) -> tuple[int, ...]:
    """The frame nearest each time, for times inside the run (±half a frame)."""
    idx = np.round(np.asarray(times_ms, float) / frame_time_ms).astype(int)
    keep = (idx >= 0) & (idx < n_frames)
    return tuple(sorted({int(i) for i in idx[keep]}))


def ed_from_ecg(ecg: EcgSignal, n_frames: int, frame_time_ms: float) -> EndDiastole:
    """ED frames: the frames at the ECG's R waves."""
    peaks = r_peaks(ecg.values, ecg.fs_hz)
    frames = frames_at(ecg.times_ms(peaks), n_frames, frame_time_ms)
    period = (float(np.median(np.diff(peaks))) * 1000.0 / ecg.fs_hz / frame_time_ms
              if peaks.size >= 2 else None)  # fmt: skip
    if not frames:
        return EndDiastole((), "none", period, "ECG present, but no R wave inside the run.")
    return EndDiastole(frames, "ECG", period, f"R waves of the DICOM ECG: frames {list(frames)}.")


def tree_spread(mask: np.ndarray) -> float:
    """RMS distance (px) of the vessel pixels from their centroid; NaN if fewer than
    ``MIN_SPREAD_PIXELS``."""
    rows, cols = np.nonzero(mask)
    if rows.size < MIN_SPREAD_PIXELS:
        return float("nan")
    return float(math.sqrt(rows.var() + cols.var()))


def _period(x: np.ndarray, lags: range) -> int | None:
    """Lag of the highest autocorrelation in ``lags`` (only lags with ≥ 1.5 cycles of data)."""
    x = x - x.mean()
    denom = float(x @ x)
    best, best_r = None, 0.0
    for lag in lags:
        if lag <= 0 or 1.5 * lag > x.size:
            continue
        r = float(x[:-lag] @ x[lag:]) / denom if denom > 0 else 0.0
        if r > best_r:
            best, best_r = lag, r
    return best


def ed_from_spread(
    spread: np.ndarray,
    smoothed_area: np.ndarray,
    frame_time_ms: float | None = None,
    *,
    area_fraction: float = ED_AREA_FRACTION,
) -> EndDiastole:
    """Image-based ED frames from the vessel tree's spread per frame (:func:`tree_spread`).

    Only the longest run of frames whose smoothed vessel area is ≥ ``area_fraction`` of its
    peak is used (contrast filling and washout change the spread too). The log-spread there
    is detrended with a straight line; the beat length is the lag of its highest
    autocorrelation (heart rate ``MIN_BPM``–``MAX_BPM`` when the frame time is known); ED
    frames are its maxima at least ``PEAK_DISTANCE`` beats apart. Without a period (too few
    frames), the single largest value is taken."""
    spread = np.asarray(spread, float)
    area = np.asarray(smoothed_area, float)
    if area.size == 0 or not np.any(area > 0):
        return EndDiastole((), "none", None, "No vessel segmented in the run.")
    ok = (area >= area_fraction * area.max()) & np.isfinite(spread) & (spread > 0)
    best, start = (0, -1), None
    for i, v in enumerate([*ok, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0] + 1:
                best = (start, i - 1)
            start = None
    a, b = best
    if b < a:
        return EndDiastole((), "none", None, "No well-opacified frames.")
    idx = np.arange(a, b + 1)
    y = np.log(spread[idx])
    if idx.size >= 3:
        y = y - np.polyval(np.polyfit(idx, y, 1), idx)
    if frame_time_ms:
        lo = max(int(math.floor(60_000.0 / MAX_BPM / frame_time_ms)), 2)
        hi = int(math.ceil(60_000.0 / MIN_BPM / frame_time_ms))
    else:
        lo, hi = MIN_PERIOD_FRAMES, idx.size
    period = _period(y, range(lo, hi + 1)) if idx.size >= 3 else None
    if period is None:
        frame = int(idx[int(np.argmax(y))])
        return EndDiastole((frame,), "images", None,
                           "Image-based: too few opacified frames for a heart rate; the frame "
                           "of the largest vessel-tree spread.")  # fmt: skip
    # maxima inside the stretch only: a value at its edge is not known to be a maximum
    peaks, _ = find_peaks(y, distance=max(math.ceil(PEAK_DISTANCE * period), 1))
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(y))])
    frames = tuple(int(idx[p]) for p in peaks)
    rate = f", {60_000.0 / (period * frame_time_ms):.0f} bpm" if frame_time_ms else ""
    return EndDiastole(frames, "images", float(period),
                       f"Image-based (vessel-tree spread, beat {period} frames{rate}): frames "
                       f"{list(frames)}.")  # fmt: skip


def end_diastole(
    n_frames: int,
    frame_time_ms: float | None,
    *,
    pointer: tuple[int, ...] | None = None,
    ecg: EcgSignal | None = None,
    spread: np.ndarray | None = None,
    smoothed_area: np.ndarray | None = None,
) -> EndDiastole:
    """ED frames from the best available source: R-wave pointer, ECG, then the images."""
    if pointer:
        period = float(np.median(np.diff(pointer))) if len(pointer) >= 2 else None
        if period is None and ecg is not None and frame_time_ms:
            period = ed_from_ecg(ecg, n_frames, frame_time_ms).period_frames
        return EndDiastole(tuple(pointer), "R-wave pointer", period,
                           f"DICOM R-wave pointer: frames {list(pointer)}.")  # fmt: skip
    if ecg is not None and frame_time_ms:
        ed = ed_from_ecg(ecg, n_frames, frame_time_ms)
        if ed.found:
            return ed
    if spread is not None and smoothed_area is not None:
        return ed_from_spread(spread, smoothed_area, frame_time_ms)
    return EndDiastole((), "none", None, "No ECG and no segmentation: ED frames unknown.")


def ed_analysis_frame(op: Opacification, ed: EndDiastole) -> tuple[int, str]:
    """The analysis frame: the ED frame inside the opacified window with the largest smoothed
    vessel area; P4d's most opacified frame if the window holds none."""
    lo, hi = op.window
    inside = [f for f in ed.frames if lo <= f <= hi]
    if inside:
        frame = max(inside, key=lambda f: (op.smoothed[f], -f))
        return frame, f"End-diastolic frame ({ed.source}) in the opacified window."
    reason = ("no end-diastolic frame lies in the opacified window" if ed.found
              else "end-diastolic frames unknown")  # fmt: skip
    return op.analysis_frame, f"The most opacified frame: {reason}."
