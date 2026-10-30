"""Dataset card for a generated synthetic dataset: ``dataset_card.{md,json}`` + histograms.

All numbers are computed from the saved shards and the config / git metadata stored next to
them; nothing is hand-typed.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from matplotlib.figure import Figure

from vstent import DISCLAIMER
from vstent.config import CONFIG_FILENAME, META_FILENAME
from vstent.data.synth import load_manifest, load_split

FFR_CUTOFF = 0.80
GREY_ZONE = (0.75, 0.85)
TARGET_BELOW_CUTOFF = 0.30
TARGET_GREY_ZONE = 0.15
OOD_SPLITS = frozenset({"ood"})
M3_PER_S_TO_ML_PER_MIN = 6e7

_KEYS = (
    "r",
    "r_ref",
    "ffr",
    "length",
    "q",
    "lesion_count",
    "lesion_shape",
    "diffuse_ds_pct",
    "r_micro_clamped",
)

# Chart tokens (reference palette, light mode).
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND = "#f0efec"


def _quantiles(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), q.tolist(), strict=True))


def derived_quantities(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-sample display quantities: D0 (mm), L (mm), max DS (%), distal FFR, Q (mL/min)."""
    return {
        "d0_mm": 2e3 * arrays["r_ref"][:, 0].astype(float),
        "length_mm": 1e3 * arrays["length"],
        "max_ds_pct": 100.0 * (1.0 - (arrays["r"] / arrays["r_ref"]).min(axis=1).astype(float)),
        "ffr_distal": arrays["ffr"][:, -1].astype(float),
        "q_ml_min": M3_PER_S_TO_ML_PER_MIN * arrays["q"],
    }


def split_stats(arrays: dict[str, np.ndarray], shapes: list[str]) -> dict[str, Any]:
    """Summary statistics of one split."""
    d = derived_quantities(arrays)
    ffr = d["ffr_distal"]
    shape_codes = arrays["lesion_shape"][arrays["lesion_shape"] >= 0]
    counts = np.bincount(arrays["lesion_count"].astype(int), minlength=1)
    return {
        "n": int(ffr.size),
        "frac_ffr_below_cutoff": float(np.mean(ffr < FFR_CUTOFF)),
        "frac_grey_zone": float(np.mean((ffr >= GREY_ZONE[0]) & (ffr <= GREY_ZONE[1]))),
        "ffr_distal": _quantiles(ffr),
        "d0_mm": _quantiles(d["d0_mm"]),
        "length_mm": _quantiles(d["length_mm"]),
        "max_ds_pct": _quantiles(d["max_ds_pct"]),
        "q_ml_min": _quantiles(d["q_ml_min"]),
        "lesion_count": {str(k): int(c) for k, c in enumerate(counts)},
        "lesion_shapes": {
            name: int(c)
            for name, c in zip(shapes, np.bincount(shape_codes, minlength=len(shapes)), strict=True)
        },
        "frac_diffuse": float(np.mean(~np.isnan(arrays["diffuse_ds_pct"]))),
        "r_micro_clamped": int(arrays["r_micro_clamped"].sum()),
    }


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRIDLINE, lw=0.6, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=7, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)


def plot_histograms(arrays: dict[str, np.ndarray], title: str, path: Path) -> None:
    """Small multiples: D0, L, max DS, distal FFR (with cut-off and grey zone), Q, lesion count."""
    d = derived_quantities(arrays)
    panels = [
        ("d0_mm", "Proximal reference diameter D0 (mm)"),
        ("length_mm", "Vessel length L (mm)"),
        ("max_ds_pct", "Max diameter stenosis per vessel (%)"),
        ("ffr_distal", "Distal FFR"),
        ("q_ml_min", "Hyperemic flow Q (mL/min)"),
    ]
    fig = Figure(figsize=(10.0, 6.0), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(2, 3, gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.10)
    for ax, (key, label) in zip(axes.flat, panels, strict=False):
        _style(ax)
        if key == "ffr_distal":
            ax.axvspan(*GREY_ZONE, color=BAND, zorder=0.5)
            ax.axvline(FFR_CUTOFF, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3)
            ax.text(
                FFR_CUTOFF,
                1.02,
                " 0.80",
                transform=ax.get_xaxis_transform(),
                color=INK_SECONDARY,
                fontsize=7,
                va="bottom",
            )
        ax.hist(d[key], bins=60, color=SERIES, edgecolor=SURFACE, linewidth=0.5, zorder=2)
        ax.set_title(label, loc="left", fontsize=8.5, color=INK_SECONDARY)
    ax = axes.flat[5]
    _style(ax)
    counts = np.bincount(arrays["lesion_count"].astype(int))
    ax.bar(np.arange(counts.size), counts, width=0.7, color=SERIES, zorder=2)
    ax.set_xticks(np.arange(counts.size))
    ax.set_title("Focal lesions per vessel", loc="left", fontsize=8.5, color=INK_SECONDARY)

    fig.text(0.06, 0.95, title, fontsize=11, color=INK_PRIMARY)
    fig.text(
        0.06,
        0.915,
        "Counts of samples; grey band on distal FFR = grey zone [0.75, 0.85]",
        fontsize=8,
        color=INK_SECONDARY,
    )
    fig.text(0.06, 0.02, DISCLAIMER, fontsize=7, color=INK_MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def _fmt_q(q: dict[str, float], digits: int = 2) -> str:
    return f"{q['median']:.{digits}f} [{q['p05']:.{digits}f}, {q['p95']:.{digits}f}]"


def _markdown(card: dict[str, Any]) -> str:
    prov, splits = card["provenance"], card["splits"]
    lines = [
        f"# Dataset card: synthetic vessels {card['name']}",
        "",
        f"> **{DISCLAIMER}** Synthetic data only; no patient data.",
        "",
        "Generated by `vstent synth`. Do not edit by hand.",
        "",
        "## What it is",
        "",
        "Single coronary-like vessels (no side branches) as radius profiles on "
        f"{card['n_points']} points: a linearly tapered reference, 0-4 focal lesions "
        f"({', '.join(card['shapes'])}), optional diffuse disease and smoothed measurement noise. "
        "Labels (FFR(s), ΔP(s), Q, R_micro) come from the reduced-order solver "
        "(docs/PHYSICS.md), computed on exactly the stored float32 noisy profile. "
        "Sampling ranges: `config.yaml` next to the data (copy of `configs/synth_v1.yaml`).",
        "",
        "## Acceptance targets (in-distribution splits)",
        "",
        "| Split | FFR < 0.80 | Target | Grey zone [0.75, 0.85] | Target | Pass |",
        "|---|---|---|---|---|---|",
    ]
    for t in card["targets"]:
        lines.append(
            f"| {t['split']} | {t['frac_ffr_below_cutoff']:.1%} | ≥ {TARGET_BELOW_CUTOFF:.0%} "
            f"| {t['frac_grey_zone']:.1%} | ≥ {TARGET_GREY_ZONE:.0%} "
            f"| {'yes' if t['passed'] else 'NO'} |"
        )
    lines += [
        "",
        "## Splits",
        "",
        "Values are median [5th, 95th percentile].",
        "",
        "| Split | n | Seed | Distal FFR | FFR < 0.80 | Grey zone | D0 (mm) | L (mm) "
        "| Max DS (%) | Q (mL/min) | Lesions 0/1/2/3/4 | Diffuse | R_micro clamped |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in splits.items():
        counts = "/".join(str(s["lesion_count"].get(str(k), 0)) for k in range(5))
        lines.append(
            f"| {name} | {s['n']:,} | {s['seed']} | {_fmt_q(s['ffr_distal'], 3)} "
            f"| {s['frac_ffr_below_cutoff']:.1%} | {s['frac_grey_zone']:.1%} "
            f"| {_fmt_q(s['d0_mm'])} | {_fmt_q(s['length_mm'], 1)} | {_fmt_q(s['max_ds_pct'], 1)} "
            f"| {_fmt_q(s['q_ml_min'], 0)} | {counts} | {s['frac_diffuse']:.1%} "
            f"| {s['r_micro_clamped']} |"
        )
    shapes = splits["train"]["lesion_shapes"] if "train" in splits else {}
    lines += [
        "",
        "Lesion shapes (train): " + ", ".join(f"{k} {v:,}" for k, v in shapes.items()) + ".",
        "",
        "## Distributions (train)",
        "",
        "![Dataset histograms](figures/dataset_hist.png)",
        "",
        "## Generation",
        "",
        f"- {card['n_total']:,} samples in {card['generation_seconds']:.1f} s "
        f"with {card['workers']} worker processes on {card['cpu']}",
        f"- git commit: `{prov.get('git_commit')}` (dirty: {prov.get('git_dirty')}), "
        f"vstent {prov.get('vstent_version')}, generated {prov.get('created_utc')}",
        "- Reproducible: shard seeds are (split seed, shard index), independent of the number "
        "of workers; per-shard content SHA-256 hashes are in `manifest.json`.",
        "",
        "## Limitations",
        "",
        "- Geometry is synthetic and idealised (straight single vessel, circular lumen, no side "
        "branches, no bifurcation lesions).",
        "- Labels are reduced-order model outputs, not 3D CFD or invasive FFR; the surrogate "
        "trained on them can be no more accurate than this model.",
        "- `ood` (80-120 mm, 3-4 lesions) is out of distribution by design; it has no targets.",
        "",
    ]
    return "\n".join(lines)


def write_dataset_card(data_dir: str | Path, report_dir: str | Path = "reports") -> dict[str, Any]:
    """Compute statistics for every split and write the card, JSON and histogram figure."""
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    manifest = load_manifest(data_dir)
    config = yaml.safe_load((data_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
    provenance = yaml.safe_load((data_dir / META_FILENAME).read_text(encoding="utf-8"))

    splits: dict[str, Any] = {}
    targets = []
    for name, info in manifest["splits"].items():
        arrays = load_split(data_dir, name, _KEYS)
        splits[name] = {"seed": info["seed"], **split_stats(arrays, manifest["shapes"])}
        if name == "train":
            plot_histograms(
                arrays,
                f"Synthetic dataset {manifest['name']}: training split (n = {info['n']:,})",
                report_dir / "figures" / "dataset_hist.png",
            )
        if name not in OOD_SPLITS:
            s = splits[name]
            targets.append(
                {
                    "split": name,
                    "frac_ffr_below_cutoff": s["frac_ffr_below_cutoff"],
                    "frac_grey_zone": s["frac_grey_zone"],
                    "passed": s["frac_ffr_below_cutoff"] >= TARGET_BELOW_CUTOFF
                    and s["frac_grey_zone"] >= TARGET_GREY_ZONE,
                }
            )

    card = {
        "name": manifest["name"],
        "n_points": manifest["n_points"],
        "shapes": manifest["shapes"],
        "n_total": sum(s["n"] for s in splits.values()),
        "generation_seconds": manifest["generation_seconds"],
        "workers": manifest["workers"],
        "cpu": manifest["cpu"],
        "targets": targets,
        "targets_met": all(t["passed"] for t in targets),
        "splits": splits,
        "sampling": config["sampling"],
        "provenance": provenance,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (report_dir / "dataset_card.md").write_text(_markdown(card), encoding="utf-8", newline="\n")
    return card
