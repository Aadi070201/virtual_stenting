"""Stenosis grading on real angiograms against expert lesion grades (CADICA, public data).

For every expert lesion box (:mod:`vstent.data.cadica`) the pipeline measures the diameter
stenosis automatically, without user clicks:

1. segment the frame (the product's ONNX segmenter);
2. on the vessel skeleton, take the path that crosses the box, from the boundary of a region
   that extends the box by its own size on every side (healthy vessel on both sides);
3. run the product's single-path QCA on that path (:func:`qca_from_centerline`, reference by the
   two-sided robust line fit) and take the largest %DS of the points inside the box.

A lesion is one expert track: the boxes of one category in one video, linked across keyframes
by position. Its grade is the median %DS of its measured frames. The experts' categories are
visual estimates (ranges such as 70–90%), so the comparison is ordinal: agreement of
categories, weighted kappa, rank correlation, and the AUC for ≥ 70% and ≥ 50% lesions.
Patients are split once (seed 0) into a development and a held-out test half; any change to the
measurement is chosen on the development half only. Units: px and %.
"""

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import spearmanr

from vstent.data.cadica import CATEGORIES, LesionBox
from vstent.imaging.centerline import smooth_and_resample, vessel_graph
from vstent.imaging.image_io import read_image_file
from vstent.imaging.qca import LEVEL, qca_from_centerline
from vstent.validation.stats import (
    auc_delong,
    auc_interval,
    cluster_bootstrap,
    weighted_kappa,
)

ORDINAL = ("p0_20", "p20_50", "p50_70", "p70_90", "p90_98")  # graded by %DS
BIN_EDGES = (20.0, 50.0, 70.0, 90.0)  # %DS → ordinal index 0 … 4 (≥ 90 → p90_98)
MIDPOINT = {c: sum(CATEGORIES[c]) / 2.0 for c in CATEGORIES}
TRACK_LINK_PX = 40.0  # boxes of one category in one video closer than this are one lesion
SPLIT_SEED = 0


@dataclass(frozen=True, eq=False)
class Track:
    """One expert lesion: its boxes on the keyframes of one video."""

    patient: str
    video: str
    category: str
    boxes: tuple[LesionBox, ...]

    @property
    def key(self) -> str:
        return f"{self.patient}_{self.video}_{self.category}_{self.boxes[0].frame}"


def lesion_tracks(boxes: list[LesionBox]) -> list[Track]:
    """Group boxes into lesions: same patient, video and category, box centres linked across
    frames (single linkage, :data:`TRACK_LINK_PX`)."""
    groups: dict[tuple[str, str, str], list[LesionBox]] = {}
    for b in boxes:
        groups.setdefault((b.patient, b.video, b.category), []).append(b)
    tracks = []
    for (patient, video, category), items in groups.items():
        if len(items) == 1:
            labels = np.array([1])
        else:
            centres = np.array([[b.x + b.w / 2, b.y + b.h / 2] for b in items])
            labels = fcluster(linkage(centres, "single"), TRACK_LINK_PX, criterion="distance")
        for lab in np.unique(labels):
            members = tuple(b for b, k in zip(items, labels, strict=True) if k == lab)
            tracks.append(Track(patient, video, category, tuple(sorted(members,
                                key=lambda b: b.frame))))  # fmt: skip
    return sorted(tracks, key=lambda t: (int(t.patient[1:]), int(t.video[1:]), t.category,
                                        t.boxes[0].frame))  # fmt: skip


def patient_split(patients: list[str], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Patient → ``dev`` or ``test`` (half each, fixed permutation)."""
    order = sorted(set(patients), key=lambda p: int(p[1:]))
    perm = np.random.default_rng(seed).permutation(len(order))
    n_dev = len(order) // 2
    return {order[i]: ("dev" if k < n_dev else "test") for k, i in enumerate(perm)}


def path_through_box(graph: nx.Graph, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Skeleton path (n, 2) as (row, col) that crosses ``box`` (x, y, w, h): between two exits of
    the region that extends the box by max(w, h) on every side, the pair whose path has the
    most nodes inside the box (then the longest). ``None`` if no skeleton crosses the box."""
    x, y, w, h = box
    m = max(w, h)
    r0, r1, c0, c1 = y - m, y + h + m, x - m, x + w + m

    def in_region(n) -> bool:
        return r0 <= n[0] <= r1 and c0 <= n[1] <= c1

    def in_box(n) -> bool:
        return y <= n[0] <= y + h and x <= n[1] <= x + w

    sub = graph.subgraph([n for n in graph.nodes if in_region(n)])
    best, best_score = None, (0, 0.0)
    for comp in nx.connected_components(sub):
        if not any(in_box(n) for n in comp):
            continue
        cg = sub.subgraph(comp)
        exits = [n for n in comp if graph.degree(n) == 1
                 or any(not in_region(v) for v in graph.neighbors(n))]  # fmt: skip
        for i, a in enumerate(exits):
            dist, paths = nx.single_source_dijkstra(cg, a, weight="weight")
            for b in exits[i + 1 :]:
                if b not in paths:
                    continue
                path = paths[b]
                score = (sum(in_box(n) for n in path), dist[b])
                if score > best_score:
                    best, best_score = path, score
    return None if best is None else np.asarray(best, dtype=float)


def stenosis_in_box(
    prob: np.ndarray,
    graph: nx.Graph,
    box: tuple[float, float, float, float],
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    """%DS of the vessel crossing ``box`` (see the module docstring). Lumen edges from the
    probability map ``prob``, or from the ``image`` intensity when it is given
    (:func:`~vstent.imaging.qca.measure_diameters_intensity`). Returns
    ``{"ds", "mld_px", "ref_px", "path_px"}`` or ``{"error"}``."""
    path = path_through_box(graph, box)
    if path is None or len(path) < 5:
        return {"error": "no segmented vessel crosses the box"}
    try:
        cl = smooth_and_resample(path)
        q = qca_from_centerline(prob, cl, image=image)
    except ValueError as exc:
        return {"error": f"QCA: {exc}"}
    x, y, w, h = box
    pts = q.centerline.points
    inside = (pts[:, 0] >= y) & (pts[:, 0] <= y + h) & (pts[:, 1] >= x) & (pts[:, 1] <= x + w)
    if not inside.any():
        return {"error": "the measured path (after end trimming) misses the box"}
    ds = 100.0 * (1.0 - q.profile.r / q.profile.reference_radius())
    k = int(np.argmax(np.where(inside, ds, -np.inf)))
    return {
        "ds": float(max(ds[k], 0.0)),
        "mld_px": float(q.diameters.diameter_px[k]),
        "ref_px": float(q.reference_diameter_mm[k] / q.mm_per_px),
        "path_px": float(q.centerline.s[-1]),
    }


FRAME_COLUMNS = ("track", "patient", "video", "category", "split", "frame", "status", "error",
                 "ds", "mld_px", "ref_px", "path_px")  # fmt: skip


def measure_tracks(
    tracks: list[Track],
    segmenter,
    *,
    split: dict[str, str],
    methods: tuple[str, ...] = ("probability",),
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Per edge method (``probability``: the product's QCA; ``intensity``), one row per expert
    box (frame) with its %DS or the reason it failed. Each frame is segmented once."""
    by_frame: dict[Path, list[tuple[Track, LesionBox]]] = {}
    for t in tracks:
        for b in t.boxes:
            by_frame.setdefault(b.image, []).append((t, b))
    rows: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    for k, (image, items) in enumerate(sorted(by_frame.items(), key=lambda kv: str(kv[0]))):
        img = read_image_file(image).image
        prob = segmenter.predict(img)
        graph = vessel_graph(prob >= LEVEL)
        for (t, b), m in ((item, m) for item in items for m in methods):
            box = (b.x, b.y, b.w, b.h)
            res = stenosis_in_box(prob, graph, box, img if m == "intensity" else None)
            rows[m].append({
                "track": t.key, "patient": t.patient, "video": t.video, "category": t.category,
                "split": split[t.patient], "frame": b.frame,
                "status": "failed" if "error" in res else "ok", "error": res.get("error", ""),
                **{c: round(res[c], 3) if c in res else "" for c in ("ds", "mld_px", "ref_px",
                                                                     "path_px")},
            })  # fmt: skip
        if progress is not None:
            progress(k + 1, len(by_frame))
    return rows


def write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FRAME_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def track_grades(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per lesion: the median %DS of its measured frames, frames measured / boxed."""
    tracks: dict[str, dict[str, Any]] = {}
    for r in rows:
        t = tracks.setdefault(r["track"], {"track": r["track"], "patient": r["patient"],
                                           "category": r["category"], "split": r["split"],
                                           "ds_frames": [], "n_frames": 0})  # fmt: skip
        t["n_frames"] += 1
        if r["status"] == "ok":
            t["ds_frames"].append(float(r["ds"]))
    out = []
    for t in tracks.values():
        ds = t.pop("ds_frames")
        out.append({**t, "n_measured": len(ds), "ds": float(np.median(ds)) if ds else None})
    return out


def ordinal(ds: float) -> int:
    """%DS → ordinal index of :data:`ORDINAL` (0–20, 20–50, 50–70, 70–90, ≥ 90)."""
    return int(np.searchsorted(BIN_EDGES, ds, side="right"))


def grading_stats(
    grades: list[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> dict[str, Any]:
    """Agreement of measured %DS with the expert categories for lesions graded p0_20 … p90_98
    (occlusions and sub-occlusions are summarised separately)."""
    graded = [g for g in grades if g["category"] in ORDINAL]
    ok = [g for g in graded if g["ds"] is not None]
    out: dict[str, Any] = {
        "n_lesions": len(graded),
        "n_patients": len({g["patient"] for g in graded}),
        "n_measured": len(ok),
        "feasibility": len(ok) / len(graded) if graded else float("nan"),
    }
    if len(ok) < 3:
        return out
    ds = np.array([g["ds"] for g in ok])
    expert = np.array([ORDINAL.index(g["category"]) for g in ok])
    measured = np.array([ordinal(v) for v in ds])
    patients = np.array([g["patient"] for g in ok])
    conf = np.zeros((len(ORDINAL), len(ORDINAL)), int)
    np.add.at(conf, (expert, measured), 1)
    out.update(
        confusion=conf.tolist(),
        exact=float(np.mean(measured == expert)),
        within_one=float(np.mean(np.abs(measured - expert) <= 1)),
        kappa=weighted_kappa(expert, measured, len(ORDINAL)),
        spearman=float(spearmanr(ds, expert).statistic),
        mean_ds={
            c: float(ds[expert == i].mean()) if (expert == i).any() else None
            for i, c in enumerate(ORDINAL)
        },  # fmt: skip
        median_ds={
            c: float(np.median(ds[expert == i])) if (expert == i).any() else None
            for i, c in enumerate(ORDINAL)
        },  # fmt: skip
        n_by_category={c: int((expert == i).sum()) for i, c in enumerate(ORDINAL)},
    )
    for name, cut in (("ge70", 3), ("ge50", 2)):
        pos = expert >= cut
        auc, se = auc_delong(ds, pos)
        out[f"auc_{name}"] = auc
        out[f"auc_{name}_ci"] = auc_interval(auc, se, level)
        out[f"n_pos_{name}"] = int(pos.sum())

    def stat(idx):
        e, d, m = expert[idx], ds[idx], measured[idx]
        return {
            "kappa": weighted_kappa(e, m, len(ORDINAL)),
            "spearman": float(spearmanr(d, e).statistic) if np.unique(e).size > 1 else np.nan,
            "auc_ge70": auc_delong(d, e >= 3)[0],
            "auc_ge50": auc_delong(d, e >= 2)[0],
            "exact": float(np.mean(m == e)),
        }

    out["bootstrap"] = {k: [lo, hi, n] for k, (lo, hi, n) in
                        cluster_bootstrap(patients, stat, n_boot, seed, level).items()}  # fmt: skip
    return out


def occlusion_summary(grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Sub-total (p99) and total (p100) occlusions: how often a lumen was measured through
    the box, and its %DS (a measured lumen through a total occlusion means the segmentation
    bridged it)."""
    out = {}
    for c in ("p99", "p100"):
        gs = [g for g in grades if g["category"] == c]
        ds = [g["ds"] for g in gs if g["ds"] is not None]
        out[c] = {"n": len(gs), "n_measured": len(ds),
                  "median_ds": float(np.median(ds)) if ds else None}  # fmt: skip
    return out


EDGE_METHODS = {
    "probability": "probability-map edges (the product's QCA)",
    "intensity": "intensity edges guided by the segmentation (candidate)",
}


def _row(name: str, s: dict[str, Any]) -> str:
    b = s.get("bootstrap", {})

    def ci(k):
        v = b.get(k)
        return f" ({v[0]:.2f}–{v[1]:.2f})" if v else ""

    if "kappa" not in s:
        return f"| {name} | {s['n_measured']} / {s['n_lesions']} | – | – | – | – | – | – |"
    return (
        f"| {name} | {s['n_measured']} / {s['n_lesions']} ({s['feasibility']:.0%}) | "
        f"{s['kappa']:.2f}{ci('kappa')} | {s['spearman']:.2f}{ci('spearman')} | "
        f"{s['exact']:.0%} | {s['within_one']:.0%} | {s['auc_ge70']:.2f}{ci('auc_ge70')} | "
        f"{s['auc_ge50']:.2f}{ci('auc_ge50')} |"
    )


def _confusion(s: dict[str, Any]) -> list[str]:
    conf = s.get("confusion")
    if conf is None:
        return []
    lines = ["| Expert (rows) / measured (columns) | " + " | ".join(ORDINAL) + " |",
             "|---|" + "---:|" * len(ORDINAL)]  # fmt: skip
    for c, row in zip(ORDINAL, conf, strict=True):
        lines.append(f"| {c} | " + " | ".join(str(v) for v in row) + " |")
    return lines


def grading_figure(grades: dict[str, list[dict[str, Any]]], path: Path, split: str) -> None:
    """Measured %DS by expert category (each edge method, one split)."""
    from vstent.plotting import (
        BAND,
        INK_MUTED,
        INK_SECONDARY,
        SERIES_1,
        SERIES_2,
        new_figure,
        save_figure,
        style_axes,
    )

    fig = new_figure(11.0, 4.2, "Automatic %DS at expert-graded lesions (CADICA)",
                     f"{split} patients; dots = lesions (median over keyframes), bands = the "
                     "experts' visual range, line = median")  # fmt: skip
    axes = np.atleast_1d(fig.subplots(1, len(grades), sharey=True, gridspec_kw={"wspace": 0.08}))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.17)
    rng = np.random.default_rng(0)
    colors = {"probability": SERIES_1, "intensity": SERIES_2}
    for ax, (method, gs) in zip(axes, grades.items(), strict=True):
        style_axes(ax)
        for i, c in enumerate(ORDINAL):
            lo, hi = CATEGORIES[c]
            ax.fill_between([i - 0.4, i + 0.4], lo, hi, color=BAND, lw=0, zorder=0.5)
            ds = np.array([g["ds"] for g in gs if g["split"] == split and g["category"] == c
                           and g["ds"] is not None])  # fmt: skip
            if ds.size:
                ax.scatter(i + rng.uniform(-0.25, 0.25, ds.size), ds, s=10, alpha=0.7, lw=0,
                           color=colors.get(method, INK_MUTED), zorder=3)  # fmt: skip
                ax.plot([i - 0.3, i + 0.3], [np.median(ds)] * 2, color=INK_SECONDARY, lw=2,
                        zorder=4)  # fmt: skip
        labels = [c.replace("p", "").replace("_", "–") + "%" for c in ORDINAL]
        ax.set_xticks(range(len(ORDINAL)), labels)
        ax.set_title(EDGE_METHODS[method], fontsize=8.5, color=INK_SECONDARY, loc="left")
        ax.set_xlabel("Expert grade (visual %DS)", fontsize=8, color=INK_SECONDARY)
        ax.set_ylim(-2, 100)
    axes[0].set_ylabel("Measured %DS (QCA)", fontsize=8, color=INK_SECONDARY)
    save_figure(fig, path)


def write_grading_report(
    rows: dict[str, list[dict[str, Any]]],
    out_md: str | Path,
    *,
    source: dict[str, Any],
    n_boot: int = 2000,
    git: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the CADICA grading report (``out_md`` + ``.json``, figure, per-lesion CSVs)."""
    import json

    from vstent import DISCLAIMER
    from vstent.config import git_info
    from vstent.data.cadica import CITATION

    git = git or git_info()
    out_md = Path(out_md)
    grades = {m: track_grades(r) for m, r in rows.items()}
    stats = {m: {sp: grading_stats([g for g in gs if g["split"] == sp], n_boot=n_boot)
                 for sp in ("dev", "test")} for m, gs in grades.items()}  # fmt: skip
    occl = {m: occlusion_summary(gs) for m, gs in grades.items()}
    grading_figure(grades, out_md.parent / "figures" / "stenosis_grading.png", "test")
    any_g = next(iter(grades.values()))
    n_patients = len({g["patient"] for g in any_g})
    split_n = {sp: len({g["patient"] for g in any_g if g["split"] == sp}) for sp in ("dev", "test")}
    table_head = [
        "| Edge method | Measured / lesions | Weighted kappa | Spearman ρ | Exact | ±1 category "
        "| AUC ≥ 70% | AUC ≥ 50% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def med(s, c):
        v = s.get("median_ds", {}).get(c)
        return "–" if v is None else f"{v:.0f}"

    def occ(o, c):
        m = o[c]["median_ds"]
        return f"{o[c]['n_measured']} / {o[c]['n']}, {'–' if m is None else round(m)}"

    lines = [
        "# Stenosis grading on real angiograms: CADICA expert grades",
        "",
        f"> {DISCLAIMER}",
        "",
        "Public data, independent ground truth: the CADICA dataset (University of Málaga; "
        f"{CITATION} Mendeley Data doi:{source['doi']}, {source['licence']}). Cine angiograms "
        f"of {n_patients} patients from a different hospital and X-ray systems than the "
        "segmentation model's training data (ARCADE), with lesion boxes and lesion grades "
        "drawn by the clinicians. The grades are **visual estimates** of diameter stenosis in "
        "ranges (0–20, 20–50, 50–70, 70–90, 90–98%, sub-total, total occlusion). Visual "
        "grading is subjective and varies between observers, so perfect agreement is not "
        "expected from any method.",
        "",
        "**What was measured, without user input:** for every expert box on every keyframe, "
        "the frame is segmented, the vessel path crossing the box is taken from the skeleton "
        "(extended by the box size on every side for a healthy reference), and QCA gives the "
        "largest %DS inside the box. A lesion's value is the median over its keyframes. %DS "
        f"needs no pixel size. Patients were split once (seed 0) into {split_n['dev']} "
        f"development and {split_n['test']} test patients; the test half is the result.",
        "",
        "Lumen-edge methods compared on the same paths:",
        "",
        *(f"- **{m}**: {EDGE_METHODS[m]}" for m in rows),
        "",
        "## Results on the test patients",
        "",
        "Lesions graded 0–98% (ordinal categories). Intervals: 95% patient-level bootstrap. "
        "Exact / ±1: the measured %DS falls in the expert's category / within one category.",
        "",
        *table_head,
        *(_row(m, s["test"]) for m, s in stats.items()),
        "",
        "![Measured %DS by expert grade](figures/stenosis_grading.png)",
        "",
        "Median measured %DS by expert grade (test):",
        "",
        "| Edge method | " + " | ".join(ORDINAL) + " |",
        "|---|" + "---:|" * len(ORDINAL),
        *(
            f"| {m} | " + " | ".join(med(s["test"], c) for c in ORDINAL) + " |"
            for m, s in stats.items()
        ),  # fmt: skip
        "",
    ]
    for m, s in stats.items():
        lines += [f"Confusion matrix, {m} edges (test lesions):", "", *_confusion(s["test"]), ""]
    lines += [
        "## Development patients",
        "",
        *table_head,
        *(_row(m, s["dev"]) for m, s in stats.items()),
        "",
        "## Sub-total and total occlusions (all patients)",
        "",
        "| Edge method | p99: measured / lesions, median %DS | p100: measured / lesions, "
        "median %DS |",
        "|---|---:|---:|",
        *(f"| {m} | {occ(o, 'p99')} | {occ(o, 'p100')} |" for m, o in occl.items()),
        "",
        "A lumen measured through a total occlusion means the segmentation bridged the "
        "occluded segment (or the path in the box ran through another vessel).",
        "",
        f"Generated by `vstent stenosis-grading` from commit `{git['git_commit']}` "
        f"(dirty: {git['git_dirty']}). Per-lesion values: "
        f"`{out_md.with_suffix('').name}/lesions_<method>.csv`.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    lesion_dir = out_md.with_suffix("")
    lesion_dir.mkdir(parents=True, exist_ok=True)
    fields = ["track", "patient", "category", "split", "n_frames", "n_measured", "ds"]
    for m, gs in grades.items():
        with (lesion_dir / f"lesions_{m}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows([{**g, "ds": "" if g["ds"] is None else round(g["ds"], 2)} for g in gs])
    payload = {"stats": stats, "occlusions": occl, "source": source, **git}
    out_md.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8", newline="\n"
    )
    return stats
