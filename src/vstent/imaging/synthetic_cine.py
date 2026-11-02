"""Synthetic beating-heart cine runs (P7 test data). Not clinical data.

A P5 case (:mod:`vstent.imaging.synthetic3d`: a tree on a heart-sized sphere, two C-arm views)
is animated and rendered frame by frame, one run per view:

- **heart cycle:** the tree is scaled about the heart's centre by a volume-like curve of the
  cycle phase (0 = ECG R wave = end-diastole, scale 1). It contracts to 1 − c at the end of
  systole (``systole_s`` after the R wave), then comes rapid filling, diastasis and the atrial
  kick back to 1 (a monotone spline through these points). It also twists about a random long
  axis, in proportion to the contraction;
- **breathing:** a slow sinusoidal 3D shift;
- **contrast:** a front advances along every branch, in arc length of the end-diastolic tree,
  from the main vessel's ostium at the arrival time; a side branch fills once the front
  passes its origin. Coronary flow is phasic: the front is ``diastolic_ratio`` times faster in
  diastole than in systole, with a cycle average of ``velocity_m_s`` (the true velocity). The
  lumen is fully filled ``front_ramp_mm`` behind the front (mixing). After the plateau the
  contrast washes out linearly;
- **ECG:** a synthetic lead (P, QRS and T waves, baseline wander, noise) whose R waves fall at
  phase 0.

The two runs of a case differ in cycle phase, frame rate, breathing and contrast timing. Ground
truth per run: the ED frames (the frame nearest each R wave), the frame time, the true velocity
and the tree's position at any time. :func:`run_dicom` writes a run as an XA DICOM with
FrameTime, FrameDelay and the C-arm geometry, and, for runs with an ECG, the R-wave pointer and
the waveform. Units: SI unless named; times in s unless named.
"""

import io
import math
from dataclasses import dataclass

import numpy as np
from pydantic import Field
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from vstent.config import StrictModel
from vstent.data.synth import Range
from vstent.imaging.synthetic import render_projection, render_tubes
from vstent.imaging.synthetic3d import MIN_SINE, TwoViewCase, TwoViewSynthConfig, sample_case
from vstent.units import mm_to_m

CURVE_STEP = 2  # render every 2nd point of the 0.05 mm curves (≤ 0.5 px apart)
TIME_STEP_S = 0.001  # integration step of the front position
RETRY_SEED = 1_000_003  # seed offset between layout attempts
ECG_GAIN_MV = 0.005  # mV per stored unit (ChannelSensitivity)


class CineSynthConfig(StrictModel):
    """Sampling of synthetic beating-heart runs (uniform over [low, high] unless noted). May be
    tuned; the physics comes from configs/physics.yaml."""

    name: str = "cine_synthetic"
    seed: int = 9100
    n_cases: int = Field(40, ge=1)
    scene: TwoViewSynthConfig = TwoViewSynthConfig()
    heart_rate_bpm: Range = (50.0, 110.0)
    fps_choices: tuple[float, ...] = (7.5, 15.0, 15.0, 30.0)  # drawn with equal probability
    contraction: Range = (0.08, 0.15)  # end-systolic radius reduction (share)
    systole_s: Range = (0.28, 0.34)  # R wave to end-systole
    twist_deg: Range = (3.0, 8.0)  # at end-systole
    breathing_mm: Range = (0.0, 4.0)  # amplitude
    breathing_period_s: Range = (3.0, 6.0)
    velocity_m_s: Range = (0.10, 0.35)  # cycle-average front speed (resting runs)
    diastolic_ratio: Range = (1.5, 2.5)  # diastolic / systolic front speed
    front_ramp_mm: Range = (5.0, 15.0)
    arrival_s: Range = (0.3, 1.0)  # contrast reaches the ostium
    plateau_beats: Range = (1.5, 3.0)  # fully opacified, after the whole tree has filled
    washout_s: Range = (0.8, 1.5)
    tail_s: float = Field(0.3, ge=0)
    ecg_prob: float = Field(0.5, ge=0, le=1)
    ecg_fs_hz: float = Field(500.0, gt=0)
    ecg_noise_mv: float = Field(0.03, ge=0)
    frame_delay_ms: Range = (0.0, 100.0)  # first frame after the acquisition's reference time
    ecg_lead_ms: Range = (0.0, 300.0)  # the ECG starts this long before the first frame
    layout_tries: int = Field(30, ge=1)


@dataclass(frozen=True)
class CineRun:
    """Timing of one run (view ``view``); times in s from its first frame."""

    view: int
    fps: float
    n_frames: int
    r0_s: float  # first R wave at or after the first frame; R waves every period
    arrival_s: float
    washout_start_s: float
    washout_s: float
    breathing_m: float
    breathing_period_s: float
    breathing_phase: float
    breathing_axis: np.ndarray
    has_ecg: bool
    frame_delay_ms: float
    ecg_lead_ms: float

    @property
    def frame_time_ms(self) -> float:
        return 1000.0 / self.fps

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps

    @property
    def end_s(self) -> float:
        return self.n_frames / self.fps


@dataclass(frozen=True, eq=False)
class CineCase:
    """A beating-heart case: the P5 case, the heart's motion, the flow and two runs."""

    seed: int
    case: TwoViewCase
    period_s: float
    contraction: float
    systole_s: float
    twist_rad: float
    axis: np.ndarray
    velocity_m_s: float
    diastolic_ratio: float
    front_ramp_m: float
    offsets_m: tuple[float, ...]  # path distance from the ostium to each branch's origin
    runs: tuple[CineRun, CineRun]

    @property
    def centre(self) -> np.ndarray:
        return self.case.centre_m

    def r_times(self, k: int, start_s: float, end_s: float) -> np.ndarray:
        """R-wave times of run ``k`` in [start_s, end_s] (s)."""
        run = self.runs[k]
        first = run.r0_s - self.period_s * math.ceil((run.r0_s - start_s) / self.period_s)
        t = np.arange(first, end_s + 1e-9, self.period_s)
        return t[(t >= start_s - 1e-9) & (t <= end_s + 1e-9)]

    def ed_frames(self, k: int) -> tuple[int, ...]:
        """True ED frames of run ``k``: the frame nearest each R wave inside the run."""
        run = self.runs[k]
        dt = 1.0 / run.fps
        idx = np.round(self.r_times(k, -dt / 2, run.end_s - dt / 2) * run.fps).astype(int)
        return tuple(sorted({int(i) for i in idx if 0 <= i < run.n_frames}))

    def phase(self, k: int, t) -> np.ndarray:
        return np.mod((np.asarray(t, float) - self.runs[k].r0_s) / self.period_s, 1.0)

    def scale(self, phase) -> np.ndarray:
        """Heart scale at cycle ``phase`` (1 at end-diastole)."""
        c, sys = self.contraction, self.systole_s / self.period_s
        knots = np.array([0.0, sys, min(sys + 0.12, 0.8), 0.9, 1.0])
        values = 1.0 - c * np.array([0.0, 1.0, 0.35, 0.12, 0.0])
        return PchipInterpolator(knots, values)(np.mod(phase, 1.0))

    def speed_factor(self, phase) -> np.ndarray:
        """Front speed / cycle average at ``phase``: systole slower, diastole faster."""
        sys = self.systole_s / self.period_s
        g_s = 1.0 / (sys + (1.0 - sys) * self.diastolic_ratio)
        return np.where(np.mod(phase, 1.0) < sys, g_s, self.diastolic_ratio * g_s)

    def front_m(self, k: int, t) -> np.ndarray:
        """Front position along the main vessel (m of end-diastolic arc length) at times ``t``
        (s) of run ``k``; 0 before the arrival."""
        run = self.runs[k]
        grid = np.arange(run.arrival_s, max(run.end_s, run.arrival_s) + 2 * TIME_STEP_S,
                         TIME_STEP_S)  # fmt: skip
        v = self.velocity_m_s * self.speed_factor(self.phase(k, grid))
        pos = np.concatenate([[0.0], np.cumsum(v[:-1] * TIME_STEP_S)])
        return np.interp(np.asarray(t, float), grid, pos, left=0.0)

    def fade(self, k: int, t: float) -> float:
        run = self.runs[k]
        return float(np.clip(1.0 - (t - run.washout_start_s) / run.washout_s, 0.0, 1.0))

    def position(self, k: int, t: float, points: np.ndarray) -> np.ndarray:
        """Points (m, end-diastolic) at time ``t`` of run ``k``: scaled about the heart's
        centre, twisted about its axis, shifted by breathing."""
        run = self.runs[k]
        phase = self.phase(k, t)
        s = float(self.scale(phase))
        angle = self.twist_rad * (1.0 - s) / self.contraction
        rot = Rotation.from_rotvec(angle * self.axis).as_matrix()
        c = self.centre
        shift = (
            run.breathing_m
            * math.sin(2 * math.pi * t / run.breathing_period_s + run.breathing_phase)
            * run.breathing_axis
        )
        return c + (np.asarray(points, float) - c) * s @ rot.T + shift

    def landmarks(self, k: int, frame: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """Main vessel start and end (row, col) in run ``k`` at ``frame`` (px)."""
        t = frame / self.runs[k].fps
        ends = self.case.views[k].project(self.position(k, t, self.case.curves[0][[0, -1]]))
        return (float(ends[0, 0]), float(ends[0, 1])), (float(ends[1, 0]), float(ends[1, 1]))


def _branch_offsets(case: TwoViewCase) -> tuple[float, ...]:
    tree = case.tree.tree
    out = [0.0]
    for k in range(1, tree.n_branches):
        parent = tree.parent[k]
        out.append(out[parent] + float(tree.branches[parent].s[tree.attach[k]]))
    return tuple(out)


def _inside(cc: CineCase, margin: float) -> bool:
    """The main vessel stays inside both views over a whole cycle and the breathing range."""
    main = cc.case.curves[0][::20]
    for k, g in enumerate(cc.case.views):
        run = cc.runs[k]
        for t in np.linspace(0.0, max(run.end_s, cc.period_s), 24):
            p = g.project(cc.position(k, float(t), main))
            if np.any(p < margin) or np.any(p > np.asarray(g.shape) - 1 - margin):
                return False
    return True


def sample_cine_case(seed: int, cfg: CineSynthConfig) -> CineCase:
    """One beating-heart case; the layout is redrawn until the main vessel stays inside both
    views over the cycle."""
    for attempt in range(cfg.layout_tries):
        case = sample_case(seed + RETRY_SEED * attempt, cfg.scene)
        rng = np.random.default_rng([seed, attempt])
        period = 60.0 / rng.uniform(*cfg.heart_rate_bpm)
        axis = rng.standard_normal(3)
        velocity = rng.uniform(*cfg.velocity_m_s)
        offsets = _branch_offsets(case)
        fill = max(o + float(b.s[-1]) for o, b in zip(offsets, case.tree.tree.branches,
                                                       strict=True)) / velocity  # fmt: skip
        runs = []
        for k in range(2):
            fps = float(rng.choice(cfg.fps_choices))
            arrival = rng.uniform(*cfg.arrival_s)
            washout_start = arrival + 1.5 * fill + rng.uniform(*cfg.plateau_beats) * period
            washout = rng.uniform(*cfg.washout_s)
            end = washout_start + washout + cfg.tail_s
            b_axis = rng.standard_normal(3)
            runs.append(CineRun(
                k, fps, int(math.ceil(end * fps)), float(rng.uniform(0.0, period)), arrival,
                washout_start, washout, mm_to_m(rng.uniform(*cfg.breathing_mm)),
                rng.uniform(*cfg.breathing_period_s), rng.uniform(0.0, 2 * math.pi),
                b_axis / np.linalg.norm(b_axis), bool(rng.random() < cfg.ecg_prob),
                rng.uniform(*cfg.frame_delay_ms), rng.uniform(*cfg.ecg_lead_ms),
            ))  # fmt: skip
        cc = CineCase(seed, case, period, rng.uniform(*cfg.contraction),
                      rng.uniform(*cfg.systole_s), math.radians(rng.uniform(*cfg.twist_deg)),
                      axis / np.linalg.norm(axis), velocity, rng.uniform(*cfg.diastolic_ratio),
                      mm_to_m(rng.uniform(*cfg.front_ramp_mm)), offsets,
                      (runs[0], runs[1]))  # fmt: skip
        if _inside(cc, cfg.scene.margin_px):
            return cc
    raise ValueError(f"seed {seed}: no layout keeps the vessel inside both views over the cycle")


@dataclass(frozen=True, eq=False)
class RenderedRun:
    """A rendered run: (T, H, W) uint8 frames and (T, H, W) float32 stand-in vessel
    probabilities (the filled reference lumen)."""

    frames: np.ndarray
    probs: np.ndarray


def render_run(cc: CineCase, k: int, *, probs: bool = True) -> RenderedRun:
    """Every frame of run ``k`` (the D39 imaging chain of the case)."""
    case, run, g = cc.case, cc.runs[k], cc.case.views[k]
    n = case.views[k].shape[0]
    rng = np.random.default_rng([cc.seed, k, 7])
    texture = gaussian_filter(rng.standard_normal((n, n)), 40.0)
    texture *= 0.04 / max(float(texture.std()), 1e-12)
    im = case.imaging
    fronts = cc.front_m(k, run.times)
    radii = [tuple(a[::CURVE_STEP] for a in case.radii(b)) for b in range(len(case.curves))]
    arcs = [s[::CURVE_STEP] for s in case.curve_s]
    frames, maps = [], []
    for j, t in enumerate(run.times):
        fade = cc.fade(k, float(t))
        tubes, refs = [], []
        for b, curve in enumerate(case.curves):
            conc = np.clip((fronts[j] - cc.offsets_m[b] - arcs[b]) / cc.front_ramp_m, 0.0, 1.0)
            conc = conc * fade
            if not np.any(conc > 0):
                continue
            x = cc.position(k, float(t), curve[::CURVE_STEP])
            p = g.project(x)
            size = g.pixel_size_at(x)
            r, r_ref = radii[b]
            tangent = np.gradient(x, axis=0)
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
            ray = x - g.source
            ray /= np.linalg.norm(ray, axis=1, keepdims=True)
            sine = np.maximum(np.linalg.norm(np.cross(tangent, ray), axis=1), MIN_SINE)
            tubes.append((p, r / size, r / size / sine * conc))
            if probs:
                filled = conc >= 0.5
                if filled.sum() >= 2:
                    refs.append((p[filled], 2.0 * r_ref[filled] / size[filled]))
        img = render_projection(
            (n, n), tubes, mm_per_px=g.iso_pixel_m * 1e3, mu_per_mm=im["mu_per_mm"],
            background=im["background"] + texture, blur_sigma=im["blur_px"],
            edge_amount=im["edge_amount"], edge_sigma=im["edge_sigma_px"],
            noise_std=im["noise_sd"], seed=int(rng.integers(2**31)),
        )  # fmt: skip
        frames.append(np.round(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))
        if probs:
            maps.append(render_tubes((n, n), refs, blur_sigma=0.0, noise_std=0.0) if refs
                        else np.zeros((n, n), np.float32))  # fmt: skip
    empty = np.zeros((0, n, n), np.float32)
    return RenderedRun(np.stack(frames), np.stack(maps) if probs else empty)


def ecg_waveform(cc: CineCase, k: int, fs_hz: float, noise_mv: float) -> tuple[np.ndarray, float]:
    """A synthetic ECG lead of run ``k`` (mV) and its start (ms from the first frame)."""
    run = cc.runs[k]
    start = -run.ecg_lead_ms / 1000.0
    t = np.arange(start, run.end_s, 1.0 / fs_hz)
    rng = np.random.default_rng([cc.seed, k, 11])
    sig = 0.1 * np.sin(2 * math.pi * 0.25 * t + rng.uniform(0, 2 * math.pi))
    t_wave = 0.25 * math.sqrt(cc.period_s)
    waves = ((-0.16, 0.12, 0.025), (-0.025, -0.10, 0.008), (0.0, 1.0, 0.012),
             (0.03, -0.25, 0.012), (t_wave, 0.30, 0.05))  # fmt: skip
    for r in cc.r_times(k, start - cc.period_s, run.end_s + cc.period_s):
        for dt, amp, sd in waves:
            sig += amp * np.exp(-0.5 * ((t - r - dt) / sd) ** 2)
    sig += rng.normal(0.0, noise_mv, t.size)
    return sig, start * 1000.0


def run_dicom(cc: CineCase, k: int, frames: np.ndarray, cfg: CineSynthConfig) -> bytes:
    """Run ``k`` as an XA multi-frame DICOM (8-bit) with FrameTime, FrameDelay, the C-arm
    geometry and, if the run has an ECG, the R-wave pointer and a one-lead waveform."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, XRayAngiographicImageStorage, generate_uid

    run, g = cc.runs[k], cc.case.views[k]
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = XRayAngiographicImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, generate_uid()
    ds.Modality = "XA"
    t, h, w = frames.shape
    ds.Rows, ds.Columns, ds.NumberOfFrames = h, w, t
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.FrameTime = f"{run.frame_time_ms:.4f}"
    ds.FrameDelay = f"{run.frame_delay_ms:.3f}"
    ds.FrameIncrementPointer = 0x00181063
    ds.ImagerPixelSpacing = [f"{g.pixel_m[0] * 1e3:.6f}", f"{g.pixel_m[1] * 1e3:.6f}"]
    ds.DistanceSourceToDetector = f"{g.sid_m * 1e3:.3f}"
    ds.DistanceSourceToPatient = f"{g.sod_m * 1e3:.3f}"
    ds.PositionerPrimaryAngle = f"{g.primary_deg:.4f}"
    ds.PositionerSecondaryAngle = f"{g.secondary_deg:.4f}"
    if run.has_ecg:
        ds.RWavePointer = list(cc.ed_frames(k))
        sig, start_ms = ecg_waveform(cc, k, cfg.ecg_fs_hz, cfg.ecg_noise_mv)
        src = Dataset()
        src.CodeValue, src.CodingSchemeDesignator, src.CodeMeaning = "2:2", "MDC", "Lead II"
        ch = Dataset()
        ch.ChannelSourceSequence = [src]
        ch.ChannelSensitivity = f"{ECG_GAIN_MV:g}"
        ch.WaveformBitsStored = 16
        item = Dataset()
        item.WaveformOriginality = "ORIGINAL"
        item.NumberOfWaveformChannels = 1
        item.NumberOfWaveformSamples = int(sig.size)
        item.SamplingFrequency = f"{cfg.ecg_fs_hz:g}"
        item.MultiplexGroupTimeOffset = f"{run.frame_delay_ms + start_ms:.3f}"
        item.ChannelDefinitionSequence = [ch]
        item.WaveformBitsAllocated = 16
        item.WaveformSampleInterpretation = "SS"
        item.WaveformData = np.round(sig / ECG_GAIN_MV).astype("<i2").tobytes()
        ds.WaveformSequence = [item]
    ds.PixelData = np.ascontiguousarray(frames, np.uint8).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()
