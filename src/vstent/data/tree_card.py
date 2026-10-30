"""Dataset card for synthetic vessel trees: ``tree_dataset_card.{md,json}`` + histograms.

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
from vstent.data.card import (
    BAND,
    FFR_CUTOFF,
    GREY_ZONE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    OOD_SPLITS,
    SURFACE,
    TARGET_BELOW_CUTOFF,
    TARGET_GREY_ZONE,
    _fmt_q,
    _quantiles,
    _style,
)
from vstent.data.synth import load_manifest, load_split
from vstent.plotting import SERIES_1, SERIES_2

_KEYS = (
    "r",
    "r_ref",
    "ffr",
    "ffr_main_single",
    "length",
    "branch_count",
    "q_inlet",
    "q_outlet",
    "r_outlet_clamped",
    "lesion_kind",
)
CARD = "tree_dataset_card"
FIGURE = "tree_dataset_hist.png"


def tree_quantities(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-tree display quantities (FFR dimensionless, lengths and diameters in mm)."""
    nb = arrays["branch_count"].astype(int)
    present = np.arange(arrays["r"].shape[1])[None, :] < nb[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(present[..., None], arrays["r"] / arrays["r_ref"], 1.0)
    side_ffr = arrays["ffr"][:, 1:, -1][present[:, 1:]]
    tree = arrays["ffr"][:, 0, -1].astype(float)
    single = arrays["ffr_main_single"][:, -1].astype(float)
    return {
        "ffr_main_tree": tree,
        "ffr_main_single": single,
        "side_effect": tree - single,
        "side_outlet_ffr": side_ffr.astype(float),
        "any_outlet_below": np.array(
            [np.any(arrays["ffr"][i, : nb[i], -1] < FFR_CUTOFF) for i in range(nb.size)]
        ),
        "main_length_mm": 1e3 * arrays["length"][:, 0],
        "d0_mm": 2e3 * arrays["r_ref"][:, 0, 0].astype(float),
        "max_ds_pct": 100.0 * (1.0 - ratio.min(axis=(1, 2)).astype(float)),
        "main_outlet_share": arrays["q_outlet"][:, 0] / arrays["q_inlet"][:, 0],
    }


def tree_split_stats(arrays: dict[str, np.ndarray], kinds: list[str]) -> dict[str, Any]:
    """Summary statistics of one split of trees."""
    d = tree_quantities(arrays)
    tree = d["ffr_main_tree"]
    side_branches = np.bincount(arrays["branch_count"].astype(int) - 1, minlength=5)
    lesion_kind = arrays["lesion_kind"][arrays["lesion_kind"] >= 0]
    return {
        "n": int(tree.size),
        "side_branches": {str(k): int(c) for k, c in enumerate(side_branches)},
        "frac_ffr_below_cutoff": float(np.mean(tree < FFR_CUTOFF)),
        "frac_grey_zone": float(np.mean((tree >= GREY_ZONE[0]) & (tree <= GREY_ZONE[1]))),
        "frac_single_below_cutoff": float(np.mean(d["ffr_main_single"] < FFR_CUTOFF)),
        "frac_any_outlet_below_cutoff": float(np.mean(d["any_outlet_below"])),
        "ffr_main_tree": _quantiles(tree),
        "ffr_main_single": _quantiles(d["ffr_main_single"]),
        "side_effect": _quantiles(d["side_effect"]),
        "frac_side_effect_negative": float(np.mean(d["side_effect"] < 0)),
        "side_outlet_ffr": _quantiles(d["side_outlet_ffr"]),
        "main_length_mm": _quantiles(d["main_length_mm"]),
        "d0_mm": _quantiles(d["d0_mm"]),
        "max_ds_pct": _quantiles(d["max_ds_pct"]),
        "main_outlet_share": _quantiles(d["main_outlet_share"]),
        "lesion_kinds": {
            name: int(c)
            for name, c in zip(kinds, np.bincount(lesion_kind, minlength=len(kinds)), strict=True)
        },
        "r_outlet_clamped": int(arrays["r_outlet_clamped"].sum()),
    }


def plot_tree_histograms(arrays: dict[str, np.ndarray], title: str, path: Path) -> None:
    """Small multiples: main distal FFR (tree and single path), side-branch effect, side
    outlet FFR, max DS, main outlet flow share, side branches per tree."""
    d = tree_quantities(arrays)
    fig = Figure(figsize=(10.0, 6.0), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(2, 3, gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.10)

    ax = axes[0, 0]
    _style(ax)
    ax.axvspan(*GREY_ZONE, color=BAND, zorder=0.5)
    ax.axvline(FFR_CUTOFF, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3)
    bins = np.linspace(0.0, 1.0, 61)
    for key, color, label in (
        ("ffr_main_single", SERIES_2, "single path (M1)"),
        ("ffr_main_tree", SERIES_1, "tree"),
    ):
        ax.hist(d[key], bins=bins, histtype="step", color=color, lw=1.6, zorder=2, label=label)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_SECONDARY, loc="upper left")
    ax.set_title("Main-vessel distal FFR", loc="left", fontsize=8.5, color=INK_SECONDARY)

    panels = [
        (axes[0, 1], "side_effect", "Tree − single-path distal FFR (main)"),
        (axes[0, 2], "side_outlet_ffr", "Side-branch outlet FFR"),
        (axes[1, 0], "max_ds_pct", "Max diameter stenosis per tree (%)"),
        (axes[1, 1], "main_outlet_share", "Share of inflow reaching the main outlet"),
    ]
    for ax, key, label in panels:
        _style(ax)
        ax.hist(d[key], bins=60, color=SERIES_1, edgecolor=SURFACE, linewidth=0.5, zorder=2)
        ax.set_title(label, loc="left", fontsize=8.5, color=INK_SECONDARY)
    ax = axes[1, 2]
    _style(ax)
    counts = np.bincount(arrays["branch_count"].astype(int) - 1, minlength=5)
    ax.bar(np.arange(counts.size), counts, width=0.7, color=SERIES_1, zorder=2)
    ax.set_xticks(np.arange(counts.size))
    ax.set_title("Side branches per tree", loc="left", fontsize=8.5, color=INK_SECONDARY)

    fig.text(0.06, 0.95, title, fontsize=11, color=INK_PRIMARY)
    fig.text(
        0.06,
        0.915,
        "Counts of trees; grey band = grey zone [0.75, 0.85], dashed = 0.80",
        fontsize=8,
        color=INK_SECONDARY,
    )
    fig.text(0.06, 0.02, DISCLAIMER, fontsize=7, color=INK_MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def _markdown(card: dict[str, Any]) -> str:
    prov, splits = card["provenance"], card["splits"]
    lines = [
        f"# Dataset card: synthetic vessel trees {card['name']}",
        "",
        f"> **{DISCLAIMER}** Synthetic data only; no patient data.",
        "",
        "Generated by `vstent synth-tree`. Do not edit by hand.",
        "",
        "## What it is",
        "",
        f"A main coronary-like vessel with 1–4 side branches, each a radius profile on "
        f"{card['n_points']} points. The main vessel steps down after every take-off "
        "(Murray-like). Lesions use the single-vessel shapes "
        f"({', '.join(card['shapes'])}), including *bifurcation* lesions centred on a take-off "
        "and *ostial* side-branch lesions. Labels come from the tree solver (docs/PHYSICS.md, "
        "*Vessel trees*) on exactly the stored float32 radii: FFR at every sample of every "
        "branch, outlet and inlet flows, and outlet resistances. The single-path FFR of the main "
        "vessel alone (M1 solver, same Pa, Pv and v_hyp) is stored for comparison. Sampling "
        "ranges: `config.yaml` next to the data (copy of `configs/synth_tree_v1.yaml`).",
        "",
        "## Acceptance targets (in-distribution splits; main-vessel distal FFR, tree solver)",
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
        "Values are median [5th, 95th percentile]. *Tree − single* is the main-vessel distal "
        "FFR of the tree minus that of the main vessel alone.",
        "",
        "| Split | n | Seed | Side branches 1/2/3/4 | Main FFR (tree) | FFR < 0.80 (tree) "
        "| Main FFR (single path) | FFR < 0.80 (single) | Tree − single | Any outlet < 0.80 "
        "| Side outlet FFR | Max DS (%) | Clamped outlets |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in splits.items():
        counts = "/".join(str(s["side_branches"].get(str(k), 0)) for k in range(1, 5))
        lines.append(
            f"| {name} | {s['n']:,} | {s['seed']} | {counts} | {_fmt_q(s['ffr_main_tree'], 3)} "
            f"| {s['frac_ffr_below_cutoff']:.1%} | {_fmt_q(s['ffr_main_single'], 3)} "
            f"| {s['frac_single_below_cutoff']:.1%} | {_fmt_q(s['side_effect'], 3)} "
            f"| {s['frac_any_outlet_below_cutoff']:.1%} | {_fmt_q(s['side_outlet_ffr'], 3)} "
            f"| {_fmt_q(s['max_ds_pct'], 1)} | {s['r_outlet_clamped']} |"
        )
    train = splits.get("train", {})
    kinds = train.get("lesion_kinds", {})
    lines += [
        "",
        "Lesions (train): " + ", ".join(f"{k} {v:,}" for k, v in kinds.items()) + ". "
        f"Trees where the tree's main distal FFR is below the single path's: "
        f"{train.get('frac_side_effect_negative', float('nan')):.1%} (train). This can happen "
        "when a proximal lesion is followed by distal disease: the side branches keep drawing "
        "flow through the proximal lesion, which the single path cannot represent.",
        "",
        "## Distributions (train)",
        "",
        f"![Tree dataset histograms](figures/{FIGURE})",
        "",
        "## Generation",
        "",
        f"- {card['n_total']:,} trees in {card['generation_seconds']:.1f} s "
        f"with {card['workers']} worker processes on {card['cpu']}",
        f"- git commit: `{prov.get('git_commit')}` (dirty: {prov.get('git_dirty')}), "
        f"vstent {prov.get('vstent_version')}, generated {prov.get('created_utc')}",
        "- Reproducible: shard seeds are (split seed, shard index), independent of the number "
        "of workers; per-shard content SHA-256 hashes are in `manifest.json`.",
        "",
        "## Limitations",
        "",
        "- Geometry is synthetic and idealised: straight branches, circular lumen, one level of "
        "side branches, no overlap.",
        "- Labels are reduced-order outputs (Murray split, no junction losses), not 3D CFD or "
        "invasive FFR.",
        "- `ood` (100–140 mm main vessel, 4 side branches, 2–3 main lesions) is out of "
        "distribution by design; it has no targets.",
        "",
    ]
    return "\n".join(lines)


def write_tree_dataset_card(
    data_dir: str | Path, report_dir: str | Path = "reports"
) -> dict[str, Any]:
    """Compute statistics for every split and write the card, JSON and histogram figure."""
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    manifest = load_manifest(data_dir)
    config = yaml.safe_load((data_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
    provenance = yaml.safe_load((data_dir / META_FILENAME).read_text(encoding="utf-8"))
    splits: dict[str, Any] = {}
    targets = []
    for name, info in manifest["splits"].items():
        arrays = load_split(data_dir, name, _KEYS)
        splits[name] = {"seed": info["seed"], **tree_split_stats(arrays, manifest["lesion_kinds"])}
        if name == "train":
            plot_tree_histograms(
                arrays,
                f"Synthetic trees {manifest['name']}: training split (n = {info['n']:,})",
                report_dir / "figures" / FIGURE,
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
    (report_dir / f"{CARD}.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (report_dir / f"{CARD}.md").write_text(_markdown(card), encoding="utf-8", newline="\n")
    return card
