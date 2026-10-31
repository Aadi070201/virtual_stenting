"""The reduced-order pressure-loss model against the FDA benchmark nozzle experiments.

The nozzle (:mod:`vstent.data.fda_nozzle`) is an idealised stenosis with an exactly known
geometry and inter-laboratory wall-pressure measurements. The model's net loss between the
upstream and downstream straight pipes is ΔP = a·Q + b·Q² with the M1 coefficients
(:func:`~vstent.physics.solver.viscous_coefficient`,
:func:`~vstent.physics.solver.expansion_coefficient`) on the nozzle's radius profile, with the
experiment's density, viscosity and flow rate. Nothing is fitted to these data; Kt and the
expansion threshold come from ``configs/physics.yaml``.

The measured net loss of a laboratory is the median wall pressure of the three most upstream
taps minus that of the three most downstream taps, all in the 0.012 m pipes. The model has no
reversible (Bernoulli) pressure dip inside the throat, so only the net loss is compared.
Units: SI (m, Pa, m³/s).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER
from vstent.config import git_info
from vstent.data.fda_nozzle import COMMIT, DOI, LICENCE, RAW_DIR, REYNOLDS, NozzleRun, load_runs
from vstent.physics.solver import PhysicsParams, expansion_coefficient, viscous_coefficient
from vstent.plotting import (
    BAND,
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    new_figure,
    save_figure,
    style_axes,
)

D_PIPE = 0.012  # m (Stewart et al. 2012)
D_THROAT = 0.004  # m
L_THROAT = 0.04  # m
CONE_ANGLE_DEG = 20.0  # included angle of the cone
L_CONE = (D_PIPE - D_THROAT) / 2.0 / np.tan(np.deg2rad(CONE_ANGLE_DEG / 2.0))  # 0.022685 m
DZ = 2e-5  # m, profile sampling
N_TAPS = 3  # taps per straight-pipe group
LAMINAR_RE = 500  # the laminar case, the model's regime (coronary throat Re ~ 100–700)


def nozzle_radius(z: np.ndarray, orientation: str) -> np.ndarray:
    """Radius (m) at axial positions ``z`` (m; 0 at the step, flow towards +z).

    ``SE``: pipe → cone (−0.04 − L_cone … −0.04) → throat (−0.04 … 0) → step to the pipe.
    ``CD``: pipe → step down at 0 → throat (0 … 0.04) → cone (0.04 … 0.04 + L_cone) → pipe.
    """
    z = np.asarray(z, dtype=float)
    zz = -z if orientation == "SE" else z  # CD coordinates; the SE nozzle is its mirror image
    r = np.full_like(zz, D_PIPE / 2.0)
    throat = (zz >= 0.0) & (zz <= L_THROAT)
    cone = (zz > L_THROAT) & (zz < L_THROAT + L_CONE)
    r[throat] = D_THROAT / 2.0
    r[cone] = D_THROAT / 2.0 + (zz[cone] - L_THROAT) * np.tan(np.deg2rad(CONE_ANGLE_DEG / 2.0))
    if orientation == "SE":  # the step is at z = 0+: the throat ends exactly there
        r[z > 0.0] = D_PIPE / 2.0
    return r


def model_loss(
    orientation: str, z0: float, z1: float, q: float, rho: float, mu: float, params: PhysicsParams
) -> dict[str, Any]:
    """Model loss (Pa) from ``z0`` to ``z1`` at flow ``q``: total, viscous and expansion parts,
    and the cumulative loss curve along z."""
    z = np.arange(z0, z1 + DZ / 2.0, DZ)
    r = nozzle_radius(z, orientation)
    a = viscous_coefficient(r, DZ, mu)
    b = expansion_coefficient(r[None, :], rho, params.kt, params.expansion_threshold)[0]
    loss = a * q + b * q * q
    return {
        "z": z,
        "loss": loss,
        "total": float(loss[-1]),
        "viscous": float(a[-1] * q),
        "expansion": float(b[-1] * q * q),
    }


def measured_loss(run: NozzleRun) -> dict[str, float]:
    """Net loss (Pa): median of the upstream tap group minus median of the downstream group,
    and the groups' median positions (m)."""
    up, down = slice(0, N_TAPS), slice(len(run.z) - N_TAPS, None)
    return {
        "loss": float(np.median(run.p[up]) - np.median(run.p[down])),
        "z_up": float(np.median(run.z[up])),
        "z_down": float(np.median(run.z[down])),
        "p_down": float(np.median(run.p[down])),
    }


def nozzle_benchmark(params: PhysicsParams, raw_dir: str | Path = RAW_DIR) -> dict[str, Any]:
    """Measured vs model net loss for both orientations and every Reynolds number."""
    cases = []
    for orientation in ("SE", "CD"):
        for re_t in REYNOLDS:
            runs, skipped = load_runs(orientation, re_t, raw_dir)
            if not runs:
                cases.append({"orientation": orientation, "reynolds": re_t, "labs": [],
                              "skipped": skipped})  # fmt: skip
                continue
            meas = [measured_loss(r) for r in runs]
            r0 = runs[0]
            z_up = float(np.median([m["z_up"] for m in meas]))
            z_down = float(np.median([m["z_down"] for m in meas]))
            mod = model_loss(orientation, z_up, z_down, r0.q, r0.rho, r0.mu, params)
            losses = np.array([m["loss"] for m in meas])
            cases.append({
                "orientation": orientation,
                "reynolds": re_t,
                "q": r0.q,
                "v_throat": r0.q / (np.pi * (D_THROAT / 2) ** 2),
                "rho": r0.rho,
                "mu": r0.mu,
                "labs": [{"lab": r.lab, **m} for r, m in zip(runs, meas, strict=True)],
                "skipped": skipped,
                "measured_median": float(np.median(losses)),
                "measured_min": float(losses.min()),
                "measured_max": float(losses.max()),
                "model_total": mod["total"],
                "model_viscous": mod["viscous"],
                "model_expansion": mod["expansion"],
                "ratio": mod["total"] / float(np.median(losses)),
            })  # fmt: skip
    return {"cases": cases}


def _figure(params: PhysicsParams, raw_dir, res: dict[str, Any], path: Path) -> None:
    fig = new_figure(12.0, 4.2, "Reduced-order loss model vs the FDA benchmark nozzle",
                     f"Wall pressure relative to the downstream pipe, throat Re = {LAMINAR_RE} "
                     "(laminar); right: model / measured net loss")  # fmt: skip
    axes = fig.subplots(1, 3, gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.27)
    lab_color = {"243": SERIES_1, "468": SERIES_2, "763": SERIES_3}  # colour follows the lab
    handles: dict[str, Any] = {}
    panels = ((axes[0], "SE", "Collector → throat → sudden expansion"),
              (axes[1], "CD", "Sudden contraction → throat → diffuser"))  # fmt: skip
    for ax, orientation, title in panels:
        style_axes(ax)
        runs, _ = load_runs(orientation, LAMINAR_RE, raw_dir)
        z_lo = min(float(r.z.min()) for r in runs)
        z_hi = max(float(r.z.max()) for r in runs)
        zz = -1.0 if orientation == "SE" else 1.0
        throat = sorted((0.0, zz * L_THROAT))
        ax.axvspan(throat[0] * 1e3, throat[1] * 1e3, color=BAND, lw=0, zorder=0.5)
        for run in runs:
            m = measured_loss(run)
            (line,) = ax.plot(run.z * 1e3, run.p - m["p_down"], "o-", ms=3, lw=1, zorder=3,
                              color=lab_color.get(run.lab, INK_SECONDARY))  # fmt: skip
            handles.setdefault(f"lab {run.lab}", line)
        case = next(c for c in res["cases"]
                    if c["orientation"] == orientation and c["reynolds"] == LAMINAR_RE)  # fmt: skip
        r0 = runs[0]
        mod = model_loss(orientation, z_lo, z_hi, r0.q, r0.rho, r0.mu, params)
        (line,) = ax.plot(mod["z"] * 1e3, mod["loss"][-1] - mod["loss"], color=INK_MUTED, lw=2,
                          ls="--", zorder=4)  # fmt: skip
        handles.setdefault("model: irreversible loss only (no reversible dip)", line)
        ax.set_title(f"{title} (model net {case['model_total']:.0f} Pa)", fontsize=8.5,
                     color=INK_SECONDARY, loc="left")  # fmt: skip
        ax.set_xlabel("Axial position (mm; shaded: throat)", fontsize=8, color=INK_SECONDARY)
        ax.set_ylabel("Pressure − downstream (Pa)", fontsize=8, color=INK_SECONDARY)
    fig.legend(list(handles.values()), list(handles), loc="lower left", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.06, 0.06))  # fmt: skip
    ax = axes[2]
    style_axes(ax)
    for orientation, color, label in (("SE", SERIES_1, "sudden expansion"),
                                      ("CD", SERIES_2, "conical diffuser")):  # fmt: skip
        pts = [(c["reynolds"], c["ratio"]) for c in res["cases"]
               if c["orientation"] == orientation and c["labs"]]  # fmt: skip
        x, y = zip(*pts, strict=True)
        ax.plot(x, y, "o-", color=color, lw=1.8, ms=5, label=label)
    ax.axhline(1.0, color=INK_MUTED, lw=1, ls="--")
    ax.axvspan(0, 1000, color=BAND, lw=0, zorder=0.5)
    ax.set_xscale("log")
    ax.set_xticks(list(REYNOLDS), [str(r) for r in REYNOLDS])
    ax.minorticks_off()
    ax.set_xlabel("Throat Reynolds number (shaded: laminar)", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Model / measured net loss (median of labs)", fontsize=8, color=INK_SECONDARY)
    ax.set_title("Net loss ratio", fontsize=9, color=INK_SECONDARY, loc="left")
    ax.legend(fontsize=7, frameon=False)
    save_figure(fig, path)


def write_nozzle_report(
    params: PhysicsParams,
    out_md: str | Path = Path("reports/physics_nozzle.md"),
    raw_dir: str | Path = RAW_DIR,
) -> dict[str, Any]:
    """Write ``out_md`` (+ ``.json``) and ``reports/figures/physics_nozzle.png``."""
    git = git_info()
    res = nozzle_benchmark(params, raw_dir)
    out_md = Path(out_md)
    fig_path = out_md.parent / "figures" / "physics_nozzle.png"
    _figure(params, raw_dir, res, fig_path)
    lam = {c["orientation"]: c for c in res["cases"] if c["reynolds"] == LAMINAR_RE}
    se = lam["SE"]
    rows = []
    for c in res["cases"]:
        if not c["labs"]:
            rows.append(f"| {c['orientation']} | {c['reynolds']} | – | 0 | – | – | – | – | – |")
            continue
        labs = ", ".join(f"{x['loss']:.0f}" for x in c["labs"])
        rows.append(
            f"| {c['orientation']} | {c['reynolds']} | {c['q'] * 1e6:.2f} | {len(c['labs'])} | "
            f"{labs} | {c['measured_median']:.0f} | {c['model_total']:.0f} "
            f"({c['model_viscous']:.0f} + {c['model_expansion']:.0f}) | {c['ratio']:.2f} |"
        )
    skipped = sorted({(c["orientation"], c["reynolds"], s["file"], s["reason"])
                      for c in res["cases"] for s in c["skipped"]})  # fmt: skip
    lines = [
        "# Physics check: the FDA benchmark nozzle",
        "",
        f"> {DISCLAIMER}",
        "",
        "The reduced-order loss model (M1: Poiseuille viscous loss + an expansion loss "
        f"Kt·ρ/2·(1/A_min − 1/A_max)², Kt = {params.kt}) against an independent bench "
        "experiment: the U.S. FDA's benchmark nozzle, a 0.012 m pipe with a 20° cone, a "
        "0.004 × 0.04 m throat and a sudden step (diameter stenosis 67%, area stenosis 89%), "
        "measured with wall pressure taps in three laboratories (Stewart et al. 2012; data "
        f"doi:{DOI}, commit `{COMMIT[:12]}`; {LICENCE}). Nothing is fitted to these data.",
        "",
        "**Compared quantity:** the net (irreversible) pressure loss between the straight pipes "
        "upstream and downstream: the median of each laboratory's three most upstream taps "
        "minus the median of its three most downstream taps. The model has no reversible "
        "pressure dip inside the throat, so the profiles are compared only qualitatively.",
        "",
        "## Results",
        "",
        "SE: collector → throat → sudden expansion. CD: sudden contraction → throat → conical "
        "diffuser. Flow rates and fluid (1056 kg/m³, 3.5 mPa·s) from the data files.",
        "",
        "| Orientation | Throat Re | Q (mL/s) | Labs | Measured net loss per lab (Pa) | Median "
        "(Pa) | Model (Pa): viscous + expansion | Model / median |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
        *rows,
        "",
        "![Nozzle comparison](figures/physics_nozzle.png)",
        "",
        "## Reading the result",
        "",
        f"- **Laminar case (Re = {LAMINAR_RE}), the regime of coronary stenoses** (throat "
        "Reynolds numbers of a few hundred): the model gives "
        f"{se['model_total']:.0f} Pa against a measured median of "
        f"{se['measured_median']:.0f} Pa (laboratories {se['measured_min']:.0f}–"
        f"{se['measured_max']:.0f} Pa) for the sudden expansion, a ratio of {se['ratio']:.2f}. "
        f"Conical diffuser: {lam['CD']['model_total']:.0f} vs {lam['CD']['measured_median']:.0f} "
        f"Pa ({lam['CD']['ratio']:.2f}).",
        "- **The exit shape matters, and the model cannot see it.** The model's loss depends "
        "only on the areas, so it is the same in both orientations; the measured losses differ "
        f"by a factor of {se['measured_median'] / lam['CD']['measured_median']:.1f}. An abrupt "
        "exit (sudden expansion) loses more than a gradual one (diffuser): the model is low "
        "for the first and agrees with the second.",
        "- **Likely causes** (not separated by these data):",
        "  - The viscous term assumes fully developed (Poiseuille) flow. The paper notes that "
        "at Re = 500 the throat flow is still developing: the entrance length, about "
        "0.06·Re·d ≈ 0.12 m, is three times the throat. Developing flow has a higher wall "
        "shear and so a larger loss.",
        "  - The expansion term does not depend on the expansion angle. A gradual diffuser "
        "recovers more pressure than a sudden step, so for the diffuser this term is likely "
        "too large, which may offset the viscous shortfall. That would be agreement for the "
        "wrong reason.",
        "  - The model has no contraction loss.",
        "- **What it means for FFR.** Coronary stenoses have throat Reynolds numbers of a few "
        "hundred, like the laminar case. For lesions with an abrupt exit the model may "
        "underestimate the pressure loss by tens of percent, and so overestimate FFR (call "
        "the lesion less severe); for gradual lesions it is closer. This is the first "
        "experimental measurement of the model's error. It is a target for the 3D CFD "
        "milestone (P6), which can quantify it on coronary geometries. Kt and the viscous "
        "term are not tuned to this experiment, and a developing-flow correction is not "
        "applied because its exact published form was not verified here.",
        "- **Higher Reynolds numbers** (2000–6500: transitional and turbulent) are outside the "
        "model's laminar assumption and are shown for completeness.",
        "- **Measurement spread:** the laboratories disagree by up to a factor of two at "
        "Re = 500. The paper reports a 100–250 Pa offset at Re = 500 from transducers that were "
        "not differential, and some taps are quantised in steps of about 27 Pa. Every "
        "available data set is shown.",
        "",
        "## Data sets without usable wall pressures",
        "",
        *(f"- {o}, Re = {r}: `{f}`: {why}" for o, r, f, why in skipped),
        "",
        f"Generated by `vstent physics-nozzle` from commit `{git['git_commit']}` "
        f"(dirty: {git['git_dirty']}).",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    out_md.with_suffix(".json").write_text(
        json.dumps({**res, "kt": params.kt, **git}, indent=2) + "\n", encoding="utf-8",
        newline="\n",
    )  # fmt: skip
    return res
