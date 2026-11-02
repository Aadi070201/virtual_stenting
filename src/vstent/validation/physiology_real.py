"""The model's physiology assumptions against real invasive measurements (P4e). Torch-free.

DEFINE-FLOW (:mod:`vstent.data.physiology`) has, for each core-lab-accepted measurement, the
aortic and distal pressures, the Doppler instantaneous peak velocity (IPV) and the ECG, and the
core lab's three-beat windows at rest, at the lowest Pd / Pa (FFR) and at the highest flow. The
checks, pre-registered in docs/PLAN.md (P4e):

1. **Phasic flow profile:** in each resting beat (R wave to R wave, ECG R peaks), the two-level
   step profile of P7's flow time (``vstent.imaging.flow.flow_time``) is fitted to the velocity:
   the systole duration that best splits the beat into a lower and a higher mean velocity, and
   the ratio of the diastolic to the systolic mean. The ratio is free of the APV-to-mean factor.
2. **Hyperemic velocity:** the stenosis-free hyperemic mean velocity per lesion,
   k × APV(hyperemia) / FFR (FFR is the ratio of maximal flows at the same distal site),
   against the fixed velocity and P1's range; k = mean / peak velocity, bracketed 0.5–0.8.
3. **FAVOR Pilot conversion:** FAVOR(k × APV(rest)) against k × APV(hyperemia).
4. **Pressures:** Pa at hyperemia against P1's range.

Doppler velocities are measured at the wire tip, distal to the lesion, with a peak (not a mean)
velocity; the population is DEFINE-FLOW's (stable patients with intermediate lesions). Units:
SI internally (m/s, s); the files are cm/s and mmHg.
"""

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.imaging.cardiac import r_peaks
from vstent.imaging.flow import DS_RATIO_PRIOR, SYSTOLE_S, hyperemic_velocity

RAW_DIR = Path("data/raw/define_flow")
REPORT = "physiology_real"
K_VALUES = (0.5, 0.8)  # mean / peak Doppler velocity: Doucette 1992 (0.5); in vitro 1999 (≈0.8)
FIXED_V_HYP = 0.35  # m/s, configs/physics.yaml
V_HYP_RANGE = (0.25, 0.45)  # m/s, P1's input range
PA_RANGE_MMHG = (70.0, 110.0)  # P1's input range
RATIO_RANGE = (1.5, 2.5)  # P7's generator range; the prior is 2.0
FS_HZ = 200.0  # DEFINE-FLOW tracings: 5 ms samples
MARGIN_S = 1.0  # ECG kept around the core lab's window for R-peak detection
SYSTOLE_GRID_S = np.round(np.arange(0.15, 0.551, 0.01), 3)
MIN_BEAT_S, MAX_BEAT_S = 0.4, 1.6
CONDITIONS = ("ic", "iv")  # pre-PCI hyperemia; post-PCI measurements are left out


def _float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_table(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def accepted_measurements(raw_dir: str | Path = RAW_DIR) -> list[dict[str, Any]]:
    """Core-lab-accepted pre-PCI measurements with a tracing, joined to their lesion."""
    raw_dir = Path(raw_dir)
    lesions = {(r["flow_id"], r["lesion"]): r
               for r in read_table(raw_dir / "DFLOW_data" / "DFLOW_lesions.csv")}  # fmt: skip
    out = []
    for r in read_table(raw_dir / "DFLOW_data" / "DFLOW_measurements.csv"):
        if r["core"] != "1" or not r["filename"] or r["condition"] not in CONDITIONS:
            continue
        les = lesions.get((r["flow_id"], r["lesion"]), {})
        out.append({**r, "syntax_seg": les.get("syntax_seg", ""),
                    "visual_ds": les.get("visual_ds", "")})  # fmt: skip
    return out


def vessel(syntax_seg: str) -> str:
    """Coronary artery of a SYNTAX segment number (RCA 1–4, 16; LM 5; LAD 6–10; LCx 11–15)."""
    s = _float(syntax_seg)
    if not math.isfinite(s):
        return "unknown"
    s = int(s)
    if s in (1, 2, 3, 4, 16):
        return "RCA"
    if s == 5:
        return "LM"
    if 6 <= s <= 10:
        return "LAD"
    if 11 <= s <= 15:
        return "LCx"
    return "other"


def read_tracing(path: str | Path) -> dict[str, np.ndarray]:
    """Columns of one DEFINE-FLOW tracing (time s, Pd / Pa mmHg, IPV cm/s, ECG, RWave)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    return {name: np.asarray(data[name], float) for name in data.dtype.names}


def step_fit(t: np.ndarray, v: np.ndarray, period: float) -> tuple[float, float] | None:
    """The two-level step profile of one beat (``t`` s from its R wave, ``v`` velocity): the
    systole duration (s) on :data:`SYSTOLE_GRID_S` that best separates a systolic and a
    diastolic mean (least squares), and the diastolic / systolic mean ratio."""
    best = None
    for sys in SYSTOLE_GRID_S:
        if sys >= 0.8 * period:
            break
        a, b = v[t < sys], v[t >= sys]
        if a.size < 5 or b.size < 5:
            continue
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(sys), float(b.mean() / a.mean()) if a.mean() > 0 else math.nan)
    return None if best is None else (best[1], best[2])


def beat_profiles(tr: dict[str, np.ndarray], start: int, end: int) -> list[dict[str, float]]:
    """Step-profile fits of the beats inside samples ``start``–``end`` (the core lab's window),
    with R waves from the ECG (:func:`~vstent.imaging.cardiac.r_peaks`), or from the
    recorder's R-wave flags when fewer than two are found there."""
    n = tr["time"].size
    margin = int(MARGIN_S * FS_HZ)
    lo, hi = max(start - margin, 0), min(end + margin, n)
    ecg = tr["ECG"][lo:hi]
    ok = np.isfinite(ecg)
    if ok.sum() < FS_HZ:
        return []
    peaks = r_peaks(np.where(ok, ecg, np.nanmedian(ecg)), FS_HZ) + lo
    if ((peaks >= start) & (peaks <= end)).sum() < 2 and "RWave" in tr:
        peaks = np.nonzero(tr["RWave"][lo:hi] > 0.5)[0] + lo  # the recorder's flags instead
    out = []
    for r0, r1 in zip(peaks[:-1], peaks[1:], strict=True):
        if r0 < start or r1 > end + int(0.2 * FS_HZ):
            continue
        period = (r1 - r0) / FS_HZ
        if not MIN_BEAT_S <= period <= MAX_BEAT_S:
            continue
        v = tr["IPV"][r0:r1]
        t = tr["time"][r0:r1] - tr["time"][r0]
        good = np.isfinite(v)
        fit = step_fit(t[good], v[good], period)
        if fit is not None and math.isfinite(fit[1]):
            out.append({"period_s": period, "systole_s": fit[0], "ratio": fit[1]})
    return out


def _median(values) -> float | None:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], float)
    return float(np.median(v)) if v.size else None


def _quantiles(values, qs=(10, 25, 50, 75, 90)) -> dict[str, float] | None:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], float)
    if not v.size:
        return None
    return {f"p{q}": float(np.percentile(v, q)) for q in qs} | {"n": int(v.size)}


def analyse(raw_dir: str | Path = RAW_DIR) -> dict[str, Any]:
    """Every check on DEFINE-FLOW (per measurement, then per lesion as the median)."""
    raw_dir = Path(raw_dir)
    rows = accepted_measurements(raw_dir)
    per_meas = []
    for r in rows:
        path = raw_dir / "DFLOW_tracings" / r["filename"]
        rec = {"flow_id": r["flow_id"], "lesion": r["lesion"], "condition": r["condition"],
               "vessel": vessel(r["syntax_seg"]), "ffr": _float(r["FFR_c"]),
               "cfr": _float(r["CFR_c"]), "bapv": _float(r["bAPV"]) / 100.0,
               "vapv": _float(r["vAPV"]) / 100.0, "fpa": _float(r["fPa"])}  # fmt: skip
        if path.exists():
            tr = read_tracing(path)
            for key, (a, b) in (("rest", ("bn_start", "bn_end")),
                                ("hyperemia", ("vn_start", "vn_end"))):  # fmt: skip
                s, e = _float(r[a]), _float(r[b])
                if math.isfinite(s) and math.isfinite(e):
                    beats = beat_profiles(tr, int(s), int(e))
                    rec[f"{key}_ratio"] = _median([x["ratio"] for x in beats])
                    rec[f"{key}_systole_s"] = _median([x["systole_s"] for x in beats])
                    rec[f"{key}_beats"] = len(beats)
        per_meas.append(rec)
    lesions: dict[tuple[str, str], list[dict]] = {}
    for m in per_meas:
        lesions.setdefault((m["flow_id"], m["lesion"]), []).append(m)
    per_lesion = []
    for (fid, les), ms in lesions.items():
        keys = ("ffr", "cfr", "bapv", "vapv", "fpa", "rest_ratio", "rest_systole_s",
                "hyperemia_ratio", "hyperemia_systole_s")  # fmt: skip
        per_lesion.append({"flow_id": fid, "lesion": les, "vessel": ms[0]["vessel"],
                           "n_measurements": len(ms),
                           **{k: _median([m.get(k) for m in ms]) for k in keys}})  # fmt: skip
    return {"measurements": per_meas, "lesions": per_lesion, **summarise(per_lesion)}


def summarise(lesions: list[dict[str, Any]]) -> dict[str, Any]:
    """Distributions and the pre-registered rules."""
    out: dict[str, Any] = {"n_lesions": len(lesions),
                           "n_subjects": len({x["flow_id"] for x in lesions})}  # fmt: skip
    out["rest_ratio"] = _quantiles([x["rest_ratio"] for x in lesions])
    out["rest_systole_s"] = _quantiles([x["rest_systole_s"] for x in lesions])
    out["hyperemia_ratio"] = _quantiles([x["hyperemia_ratio"] for x in lesions])
    out["by_vessel"] = {
        v: {"rest_ratio": _quantiles([x["rest_ratio"] for x in lesions if x["vessel"] == v])}
        for v in ("LAD", "LCx", "RCA")
    }
    out["ffr"] = _quantiles([x["ffr"] for x in lesions])
    out["cfr"] = _quantiles([x["cfr"] for x in lesions])
    out["pa_hyperemia_mmhg"] = _quantiles([x["fpa"] for x in lesions])
    pa = [x["fpa"] for x in lesions if x["fpa"] is not None and math.isfinite(x["fpa"])]
    out["pa_inside_range"] = (float(np.mean([PA_RANGE_MMHG[0] <= p <= PA_RANGE_MMHG[1]
                                             for p in pa])) if pa else None)  # fmt: skip
    out["velocity"] = {}
    for k in K_VALUES:
        normal = [k * x["vapv"] / x["ffr"] for x in lesions
                  if x["vapv"] is not None and x["ffr"] and x["ffr"] > 0]  # fmt: skip
        pred = [hyperemic_velocity(k * x["bapv"]) / (k * x["vapv"]) - 1.0 for x in lesions
                if x["bapv"] and x["vapv"]]  # fmt: skip
        q = _quantiles(normal)
        out["velocity"][str(k)] = {
            "normal_hyperemic_m_s": q,
            "fixed_inside_iqr": bool(q and q["p25"] <= FIXED_V_HYP <= q["p75"]),
            "inside_p1_range": (float(np.mean([V_HYP_RANGE[0] <= v <= V_HYP_RANGE[1]
                                               for v in normal])) if normal else None),
            "favor_rel_error": _quantiles(pred, (5, 25, 50, 75, 95)),
        }  # fmt: skip
    med = out["rest_ratio"]["p50"] if out["rest_ratio"] else None
    out["rules"] = {
        "phasic_prior_replaced": bool(
            med is not None and not RATIO_RANGE[0] <= med <= RATIO_RANGE[1]
        ),  # fmt: skip
        "phasic_median_ratio": med,
        "fixed_velocity_revisit": not any(
            out["velocity"][str(k)]["fixed_inside_iqr"] for k in K_VALUES
        ),  # fmt: skip
    }
    return out


def _share(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _q(d, key="p50", fmt="{:.2f}") -> str:
    return "–" if not d else fmt.format(d[key])


def _iqr(d, fmt="{:.2f}") -> str:
    return "–" if not d else f"{fmt.format(d['p25'])}–{fmt.format(d['p75'])}"


def write_report(result: dict[str, Any], git: dict, report_dir: str | Path = "reports",
                 source: dict | None = None) -> None:  # fmt: skip
    """``reports/physiology_real.{md,json}``."""
    report_dir = Path(report_dir)
    r = result
    rules = r["rules"]
    vel = r["velocity"]
    prov = {**git, "vstent_version": __version__,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    lines = [
        "# The model's physiology against real invasive measurements (P4e)",
        "",
        f"> **{DISCLAIMER}** Real patients' invasive measurements, without angiograms: these "
        "check the model's physiology assumptions, not its FFR from images.",
        "",
        "Generated by `vstent physiology-real`. Do not edit by hand.",
        "",
        "**Data:** DEFINE-FLOW (Johnson, Kirkeeide and Gould, JACC Cardiovasc Interv 2021, "
        "doi:10.1016/j.jcin.2021.07.041; data doi:10.5061/dryad.h18931zm6, CC0). "
        f"{r['n_lesions']} lesions of {r['n_subjects']} subjects with a core-lab-accepted "
        "pre-PCI measurement and its raw tracing: Doppler instantaneous peak velocity, aortic "
        "and distal pressure, ECG, with the core lab's three-beat windows at rest and at "
        "peak flow. Values per lesion are the median over its measurements. The checks and "
        "their rules were pre-registered in docs/PLAN.md before the data were downloaded.",
        "",
        "## Phasic coronary flow (P7's prior)",
        "",
        "Each resting beat (ECG R wave to R wave) is fitted with P7's two-level profile: the "
        "systole duration that best splits the beat's velocity into a lower and a higher mean, "
        "and the ratio of the later (diastolic) to the earlier (systolic) mean. The ratio does "
        "not depend on the Doppler peak-to-mean factor.",
        "",
        "| | Median | Interquartile range | 10th–90th percentile | Lesions |",
        "|---|---:|---:|---:|---:|",
        f"| Diastolic / systolic ratio at rest | {_q(r['rest_ratio'])} | {_iqr(r['rest_ratio'])} "
        f"| {_q(r['rest_ratio'], 'p10')}–{_q(r['rest_ratio'], 'p90')} | "
        f"{r['rest_ratio']['n'] if r['rest_ratio'] else 0} |",
        f"| Systole duration at rest (s) | {_q(r['rest_systole_s'])} | "
        f"{_iqr(r['rest_systole_s'])} | {_q(r['rest_systole_s'], 'p10')}–"
        f"{_q(r['rest_systole_s'], 'p90')} | "
        f"{r['rest_systole_s']['n'] if r['rest_systole_s'] else 0} |",
        f"| Diastolic / systolic ratio at hyperemia | {_q(r['hyperemia_ratio'])} | "
        f"{_iqr(r['hyperemia_ratio'])} | {_q(r['hyperemia_ratio'], 'p10')}–"
        f"{_q(r['hyperemia_ratio'], 'p90')} | "
        f"{r['hyperemia_ratio']['n'] if r['hyperemia_ratio'] else 0} |",
        "",
        "Resting ratio by artery: "
        + "; ".join(f"{v} {_q(d['rest_ratio'])} ({_iqr(d['rest_ratio'])}, n "
                    f"{d['rest_ratio']['n'] if d['rest_ratio'] else 0})"
                    for v, d in r["by_vessel"].items()) + ".",
        "",
        f"P7 assumes a ratio of {DS_RATIO_PRIOR:g} (its synthetic hearts draw "
        f"{RATIO_RANGE[0]:g}–{RATIO_RANGE[1]:g}) and a systole of {SYSTOLE_S:g} s. "
        f"**Rule** (pre-registered): the prior is replaced if the median resting ratio lies "
        f"outside {RATIO_RANGE[0]:g}–{RATIO_RANGE[1]:g}. **"
        + ("Fired: the prior is replaced (see the decision log)."
           if rules["phasic_prior_replaced"] else "Did not fire: the prior stays.") + "**",
        "",
        "## Hyperemic velocity (the fixed 0.35 m/s)",
        "",
        "The stenosis-free hyperemic mean velocity per lesion is estimated as k × APV at peak "
        "flow ÷ FFR, k being the Doppler mean-to-peak factor (0.5 for a parabolic profile; "
        "about 0.8 measured in vitro). The model's fixed velocity is "
        f"{FIXED_V_HYP:g} m/s; P1 varies it over {V_HYP_RANGE[0]:g}–{V_HYP_RANGE[1]:g} m/s.",
        "",
        "| k | Median (m/s) | Interquartile range | 10th–90th percentile | Share inside "
        f"{V_HYP_RANGE[0]:g}–{V_HYP_RANGE[1]:g} m/s | 0.35 m/s inside the IQR |",
        "|---:|---:|---:|---:|---:|---|",
        *(f"| {k} | {_q(vel[k]['normal_hyperemic_m_s'])} | "
          f"{_iqr(vel[k]['normal_hyperemic_m_s'])} | "
          f"{_q(vel[k]['normal_hyperemic_m_s'], 'p10')}–"
          f"{_q(vel[k]['normal_hyperemic_m_s'], 'p90')} | "
          f"{_share(vel[k]['inside_p1_range'])} | "
          f"{'yes' if vel[k]['fixed_inside_iqr'] else 'no'} |" for k in vel),
        "",
        "**Rule** (pre-registered): the fixed velocity is revisited if 0.35 m/s lies outside "
        "the interquartile range at both k. **"
        + ("Fired: recorded as a decision." if rules["fixed_velocity_revisit"]
           else "Did not fire.") + "**",
        "",
        "## FAVOR Pilot conversion (P7)",
        "",
        "The hyperemic mean velocity predicted from the resting one, FAVOR(k × APV at rest), "
        "against the measured k × APV at peak flow. FAVOR was fitted on contrast frame counts, "
        "and a contrast injection already raises flow above rest, so this is indicative only.",
        "",
        "| k | Median relative error | Interquartile range | 5th–95th percentile |",
        "|---:|---:|---:|---:|",
        *(f"| {k} | {_q(vel[k]['favor_rel_error'], fmt='{:+.0%}')} | "
          f"{_iqr(vel[k]['favor_rel_error'], fmt='{:+.0%}')} | "
          f"{_q(vel[k]['favor_rel_error'], 'p5', '{:+.0%}')}–"
          f"{_q(vel[k]['favor_rel_error'], 'p95', '{:+.0%}')} |" for k in vel),
        "",
        "## Pressures and the lesions",
        "",
        f"- Aortic pressure at hyperemia: median {_q(r['pa_hyperemia_mmhg'], fmt='{:.0f}')} "
        f"mmHg (IQR {_iqr(r['pa_hyperemia_mmhg'], fmt='{:.0f}')}); "
        f"{_share(r['pa_inside_range'])} "
        f"inside P1's {PA_RANGE_MMHG[0]:g}–{PA_RANGE_MMHG[1]:g} mmHg.",
        f"- FFR (core lab): median {_q(r['ffr'])} (IQR {_iqr(r['ffr'])}); CFR: median "
        f"{_q(r['cfr'])} (IQR {_iqr(r['cfr'])}).",
        "- CONTRAST's tracings carry no markers of when contrast or adenosine was given, so "
        "the pre-registered rest / contrast / adenosine comparison could not be made reliably "
        "and is not reported.",
        "",
        "**Limits:** Doppler measures a peak velocity at the wire tip, distal to the lesion, "
        "and the wire alters the flow it measures; DEFINE-FLOW's patients had stable disease "
        "and intermediate lesions; no angiograms, so nothing here measures FFR from images.",
        "",
        f"Commit `{prov['git_commit']}` (dirty: {prov['git_dirty']}), created "
        f"{prov['created_utc']}.",
        "",
    ]  # fmt: skip
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{REPORT}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    payload = {k: v for k, v in r.items() if k not in ("measurements",)}
    payload |= {"provenance": prov, "source": source}
    (report_dir / f"{REPORT}.json").write_text(json.dumps(payload, indent=1, default=float),
                                               encoding="utf-8", newline="\n")  # fmt: skip


ARTERY_K = 0.8  # P4k-C4: the Doppler mean-to-peak factor of P4e's main estimate
ARTERY_MIN_CHANGE = 0.10  # an artery's own velocity is used if it differs ≥ 10% from pooled


def artery_velocity(lesions: list[dict[str, Any]], k: float = ARTERY_K) -> dict[str, Any]:
    """P4k-C4: the stenosis-free hyperemic mean velocity (k × APV at peak flow ÷ FFR, m/s) per
    artery (left = LAD + LCx; RCA) and pooled, and the pre-registered rule."""
    groups = {"left": ("LAD", "LCx"), "RCA": ("RCA",)}
    vals = {g: [k * x["vapv"] / x["ffr"] for x in lesions if x["vessel"] in arts
                and x["vapv"] is not None and x["ffr"] and x["ffr"] > 0]
            for g, arts in groups.items()}  # fmt: skip
    pooled = float(np.median(vals["left"] + vals["RCA"]))
    out: dict[str, Any] = {"k": k, "pooled": pooled, "n_pooled": len(vals["left"] + vals["RCA"]),
                           "arteries": {}}  # fmt: skip
    for g, v in vals.items():
        med = float(np.median(v)) if v else None
        change = None if med is None else med / pooled - 1.0
        out["arteries"][g] = {"n": len(v), "median": med, "change": change}
    out["use_per_artery"] = any(a["change"] is not None and abs(a["change"]) >= ARTERY_MIN_CHANGE
                                for a in out["arteries"].values())  # fmt: skip
    return out
