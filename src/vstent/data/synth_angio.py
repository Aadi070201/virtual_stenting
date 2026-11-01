"""Rendered angiograms for segmentation training (P8, sim-to-real). Not clinical data.

Each image shows one or two synthetic coronary trees from one ostium (P2 lumen profiles laid
on a heart-sized sphere, :mod:`vstent.imaging.synthetic3d`), seen in one random clinical view
and rendered with the ARCADE-fitted X-ray imaging chain (D39). Added distractors, drawn but
not labelled, make the scene closer to a clinical frame:
- small branches (below the label threshold) along the trees;
- a contrast-filled guiding catheter from the ostium out of the image;
- smooth anatomy-like shadows (heart, diaphragm), rib-like bands and vignetting.

**Labels** follow ARCADE's annotation of the major (SYNTAX) segments: the lumen of every
vessel whose reference diameter at its start is at least ``label_min_diameter_mm``; the
catheter and smaller branches are background. Masks are the lumen coverage ≥ 0.5 (uint8).

Splits are written to ``<out_dir>/<split>.npz`` (``images``, ``masks``: (N, H, W) uint8) with a
manifest (config, seeds, git commit, hashes). Units: SI unless named.
"""

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field
from scipy.ndimage import gaussian_filter

from vstent.config import StrictModel, git_info, load_config
from vstent.data.synth import Range
from vstent.data.synth_tree import TreeSynthConfig, sample_tree
from vstent.imaging.synthetic import render_projection, render_tubes
from vstent.imaging.synthetic3d import (
    MIN_SINE,
    TwoViewSynthConfig,
    layout_tree,
    sample_views,
    surface_curve,
)
from vstent.units import mm_to_m

MANIFEST = "manifest.json"
SPLITS = ("train", "val", "test")


class SynthAngioConfig(StrictModel):
    """Rendered-angiogram dataset (uniform over [low, high] unless noted). May be tuned."""

    name: str = "synth_angio_v1"
    out_dir: str = "data/synth_angio/v1"
    seed: int = 8001
    n: dict[str, int] = Field(default_factory=lambda: {"train": 2000, "val": 200, "test": 200})
    workers: int | None = None
    scene: TwoViewSynthConfig = Field(default_factory=TwoViewSynthConfig)  # trees, views, chain
    two_trees_prob: float = Field(0.6, ge=0, le=1)  # left system: two trees from one ostium
    label_min_diameter_mm: float = Field(1.5, gt=0)
    small_branches: tuple[int, int] = (2, 8)
    small_diameter_mm: Range = (0.7, 1.4)
    small_length_mm: Range = (8.0, 30.0)
    catheter_prob: float = Field(0.9, ge=0, le=1)
    catheter_diameter_mm: Range = (1.7, 2.3)  # 5F-7F
    catheter_length_mm: float = Field(200.0, gt=0)
    shadows: tuple[int, int] = (1, 3)
    shadow_amplitude: Range = (0.03, 0.12)
    shadow_softness_px: Range = (20.0, 80.0)
    ribs: tuple[int, int] = (0, 5)
    rib_width_px: Range = (12.0, 30.0)
    rib_amplitude: Range = (0.02, 0.06)
    vignetting: Range = (0.0, 0.15)
    min_inside: float = Field(0.6, gt=0, le=1)  # share of labelled vessel points in the image
    tries: int = Field(200, ge=1)


def _small_branches(rng, curves, centre, radius, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    """Unlabelled small branches leaving the given curves: (points (k, 3) m, radius (k,) m)."""
    out = []
    for _ in range(int(rng.integers(cfg.small_branches[0], cfg.small_branches[1] + 1))):
        parent = curves[int(rng.integers(len(curves)))]
        j = int(rng.integers(len(parent) // 10, max(len(parent) // 10 + 1, 9 * len(parent) // 10)))
        p0 = parent[j]
        nrm = (p0 - centre) / np.linalg.norm(p0 - centre)
        a, b = parent[max(j - 5, 0)], parent[min(j + 5, len(parent) - 1)]
        tp = (b - a) / np.linalg.norm(b - a)
        angle = math.radians(rng.uniform(*cfg.scene.side_angle_deg)) * rng.choice((-1.0, 1.0))
        t0 = tp * math.cos(angle) + np.cross(nrm, tp) * math.sin(angle)
        length = mm_to_m(rng.uniform(*cfg.small_length_mm))
        pts = surface_curve(rng, centre, float(np.linalg.norm(p0 - centre)), p0, t0, length,
                            cfg.scene)  # fmt: skip
        r0 = mm_to_m(rng.uniform(*cfg.small_diameter_mm)) / 2.0
        out.append((pts, r0 * np.linspace(1.0, 0.6, len(pts))))
    return out


def _catheter(rng, start: np.ndarray, centre: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """A guiding catheter from the ostium away from the heart and towards the head: a gently
    curved tube (points (k, 3) m, radius (k,) m)."""
    out = start - centre
    out /= np.linalg.norm(out)
    up = np.array([0.0, 0.0, 1.0])
    s = np.linspace(0.0, mm_to_m(cfg.catheter_length_mm), 400)[:, None]
    bend = np.clip(s / 0.04, 0.0, 1.0)  # turns towards the head over its first 40 mm
    direction = (1.0 - bend) * out + bend * (up + 0.3 * rng.normal(size=3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    step = s[1, 0]
    pts = start + np.cumsum(direction * step, axis=0)
    r = mm_to_m(rng.uniform(*cfg.catheter_diameter_mm)) / 2.0
    return pts, np.full(len(pts), r)


def _background(rng, n: int, base: float, cfg) -> np.ndarray:
    """Textured background with anatomy-like shadows, rib-like bands and vignetting."""
    sc = cfg.scene
    bg = np.full((n, n), base)
    tex = gaussian_filter(rng.standard_normal((n, n)), sc.texture_scale_px)
    bg += sc.texture_sd * tex / max(float(tex.std()), 1e-12)
    rr, cc = np.mgrid[0:n, 0:n].astype(float)
    for _ in range(int(rng.integers(cfg.shadows[0], cfg.shadows[1] + 1))):
        theta = rng.uniform(0, 2 * np.pi)
        offset = rng.uniform(-0.3, 0.3) * n
        d = (rr - n / 2) * np.sin(theta) + (cc - n / 2) * np.cos(theta) - offset
        soft = rng.uniform(*cfg.shadow_softness_px)
        bg -= rng.uniform(*cfg.shadow_amplitude) / (1.0 + np.exp(-d / soft))
    for _ in range(int(rng.integers(cfg.ribs[0], cfg.ribs[1] + 1))):
        theta = rng.uniform(-0.6, 0.6) + np.pi / 2
        curve = rng.uniform(-1.5e-3, 1.5e-3)
        d = (rr - rng.uniform(0, n)) * np.sin(theta) + (cc - n / 2) * np.cos(theta)
        d += curve * (cc - n / 2) ** 2
        width = rng.uniform(*cfg.rib_width_px)
        bg -= rng.uniform(*cfg.rib_amplitude) * np.exp(-0.5 * (d / (width / 2)) ** 2)
    radial = ((rr - n / 2) ** 2 + (cc - n / 2) ** 2) / (n / 2) ** 2
    bg *= 1.0 - rng.uniform(*cfg.vignetting) * np.clip(radial, 0.0, 2.0) / 2.0
    return bg


def _tube(points: np.ndarray, radius: np.ndarray, g) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D tube of a 3D vessel in view ``g``: points (px), half-width (px), half-depth (px)."""
    p = g.project(points)
    size = g.pixel_size_at(points)
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    ray = points - g.source
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    sine = np.maximum(np.linalg.norm(np.cross(tangent, ray), axis=1), MIN_SINE)
    return p, radius / size, radius / size / sine


def render_sample(seed: int, cfg: SynthAngioConfig, tree_cfg: TreeSynthConfig
                  ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:  # fmt: skip
    """One rendered angiogram: image (H, W) uint8, label mask (H, W) uint8, and facts."""
    rng = np.random.default_rng(seed)
    sc = cfg.scene
    sampling = type(tree_cfg.sampling).model_validate(
        {**tree_cfg.sampling.model_dump(), **sc.sampling}
    )
    n = sc.image_size
    label_r = mm_to_m(cfg.label_min_diameter_mm) / 2.0
    for _ in range(cfg.tries):
        centre = rng.normal(0.0, mm_to_m(sc.heart_offset_sd_mm), 3)
        radius = mm_to_m(rng.uniform(*sc.heart_radius_mm))
        n_trees = 2 if rng.random() < cfg.two_trees_prob else 1
        vessels, start = [], None  # (points, lumen r, reference r, labelled)
        for _t in range(n_trees):
            tree = sample_tree(rng, sampling, sc.n_points)
            curves, arcs = layout_tree(rng, tree, centre, radius, sc, start=start)
            start = curves[0][0]
            for b, curve, arc in zip(tree.tree.branches, curves, arcs, strict=True):
                r, r_ref = np.interp(arc, b.s, b.r), np.interp(arc, b.s, b.reference_radius())
                vessels.append((curve, r, r_ref, bool(r_ref[0] >= label_r)))
        mains = [v[0] for v in vessels]
        smalls = _small_branches(rng, mains, centre, radius, cfg)
        g = sample_views(rng, sc)[0]
        labelled = [v for v in vessels if v[3]]
        if not labelled:
            continue
        pts = np.concatenate([v[0] for v in labelled])
        p = g.project(pts)
        inside = np.all((p >= 0) & (p <= n - 1), axis=1).mean()
        p_start = g.project(start[None])[0]
        if inside < cfg.min_inside or not np.all((p_start >= 16) & (p_start <= n - 17)):
            continue
        break
    else:
        raise ValueError(f"seed {seed}: no view shows the vessels")
    chain = ("mu_per_mm", "background", "blur_px", "edge_amount", "edge_sigma_px", "noise_sd")
    im = {k: float(rng.uniform(*getattr(sc, k))) for k in chain}
    tubes = [_tube(v[0], v[1], g) for v in vessels]
    tubes += [_tube(pts_, r_, g) for pts_, r_ in smalls]
    has_catheter = bool(rng.random() < cfg.catheter_prob)
    if has_catheter:
        tubes.append(_tube(*_catheter(rng, start, centre, cfg), g))
    img = render_projection(
        (n, n), tubes, mm_per_px=g.iso_pixel_m * 1e3, mu_per_mm=im["mu_per_mm"],
        background=_background(rng, n, im["background"], cfg), blur_sigma=im["blur_px"],
        edge_amount=im["edge_amount"], edge_sigma=im["edge_sigma_px"], noise_std=im["noise_sd"],
        seed=int(rng.integers(2**31)),
    )  # fmt: skip
    lumen = [(_tube(v[0], v[1], g)[0], 2.0 * _tube(v[0], v[1], g)[1]) for v in labelled]
    mask = (render_tubes((n, n), lumen, blur_sigma=0.0, noise_std=0.0) >= 0.5).astype(np.uint8)
    facts = {"seed": seed, "trees": n_trees, "vessels": len(vessels), "labelled": len(labelled),
             "small_branches": len(smalls), "catheter": has_catheter, "view": g.label(),
             "iso_pixel_mm": g.iso_pixel_m * 1e3, "mask_fraction": float(mask.mean())}  # fmt: skip
    return np.round(img * 255).astype(np.uint8), mask, facts


def _chunk(args) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    seeds, cfg_dict = args
    cfg = SynthAngioConfig.model_validate(cfg_dict)
    tree_cfg = load_config(cfg.scene.tree_config, TreeSynthConfig)
    out = [render_sample(s, cfg, tree_cfg) for s in seeds]
    return (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]), [o[2] for o in out])


def split_seeds(cfg: SynthAngioConfig, split: str) -> list[int]:
    """Seeds of a split: disjoint blocks of 1,000,000 per split."""
    base = cfg.seed + 1_000_000 * SPLITS.index(split)
    return [base + i for i in range(cfg.n[split])]


def generate(cfg: SynthAngioConfig, log=print) -> Path:
    """Render every split into ``cfg.out_dir`` (``<split>.npz`` + manifest)."""
    import hashlib
    import os

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    workers = cfg.workers or max(1, (os.cpu_count() or 2) - 1)
    manifest: dict[str, Any] = {"config": cfg.model_dump(mode="json"), **git_info(), "splits": {}}
    for split in SPLITS:
        seeds = split_seeds(cfg, split)
        chunks = [seeds[i : i + 25] for i in range(0, len(seeds), 25)]
        cfg_dict = cfg.model_dump(mode="json")
        images, masks, facts = [], [], []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, (im, mk, fa) in enumerate(pool.map(_chunk, [(c, cfg_dict) for c in chunks])):
                images.append(im)
                masks.append(mk)
                facts += fa
                log(f"{split}: {min((k + 1) * 25, len(seeds))} / {len(seeds)}")
        images, masks = np.concatenate(images), np.concatenate(masks)
        path = out / f"{split}.npz"
        np.savez_compressed(path, images=images, masks=masks)
        digest = hashlib.sha256(images.tobytes() + masks.tobytes()).hexdigest()
        manifest["splits"][split] = {"n": len(seeds), "first_seed": seeds[0], "sha256": digest,
                                     "facts": facts}  # fmt: skip
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                                newline="\n")  # fmt: skip
    return out


def load_synth_split(out_dir: str | Path, split: str, max_images: int | None = None):
    """(images, masks) uint8 arrays of a rendered split."""
    with np.load(Path(out_dir) / f"{split}.npz") as data:
        images, masks = data["images"], data["masks"]
    return images[:max_images], masks[:max_images]


def write_card(out_dir: str | Path, report_dir: str | Path = "reports") -> dict[str, Any]:
    """``reports/synth_angio_card.md`` and an example figure from a generated dataset."""
    from vstent import DISCLAIMER
    from vstent.plotting import INK_SECONDARY, new_figure, save_figure

    out_dir, report_dir = Path(out_dir), Path(report_dir)
    manifest = json.loads((out_dir / MANIFEST).read_text(encoding="utf-8"))
    cfg = SynthAngioConfig.model_validate(manifest["config"])
    stats = {}
    for split, info in manifest["splits"].items():
        f = info["facts"]
        stats[split] = {
            "n": info["n"],
            "two_trees": float(np.mean([x["trees"] == 2 for x in f])),
            "labelled_vessels": float(np.mean([x["labelled"] for x in f])),
            "unlabelled_vessels": float(np.mean([x["vessels"] - x["labelled"] for x in f])),
            "small_branches": float(np.mean([x["small_branches"] for x in f])),
            "catheter": float(np.mean([x["catheter"] for x in f])),
            "mask_fraction": float(np.mean([x["mask_fraction"] for x in f])),
            "sha256": info["sha256"],
        }
    images, masks = load_synth_split(out_dir, "val", 4)
    fig = new_figure(10.0, 5.4, "Rendered angiograms (validation split, first four)",
                     "Top: image. Bottom: label (vessels ≥ "
                     f"{cfg.label_min_diameter_mm:g} mm) in green over the image")  # fmt: skip
    for k in range(len(images)):
        for row, show_mask in ((0, False), (1, True)):
            ax = fig.add_axes((0.02 + 0.245 * k, 0.48 - 0.4 * row, 0.23, 0.38))
            rgb = np.repeat(images[k][..., None], 3, axis=2).astype(float) / 255.0
            if show_mask:
                m = masks[k] > 0
                rgb[m] = 0.5 * rgb[m] + 0.5 * np.array([0.11, 0.69, 0.48])
            ax.imshow(rgb)
            ax.set_xticks([])
            ax.set_yticks([])
    fig_path = report_dir / "figures" / "synth_angio_examples.png"
    save_figure(fig, fig_path)
    del INK_SECONDARY
    lines = [
        "# Rendered angiograms: dataset card (P8 sim-to-real)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent synth-angio`. Do not edit by hand. Synthetic images, not clinical "
        "data.",
        "",
        "One or two synthetic coronary trees from one ostium (P2 lumen profiles on a heart-sized "
        "sphere, P5), one random clinical view, the ARCADE-fitted X-ray chain (D39). Drawn but "
        "not labelled: small branches, a contrast-filled catheter, anatomy-like shadows, "
        f"rib-like bands, vignetting. **Label:** the lumen of every vessel whose reference "
        f"diameter is ≥ {cfg.label_min_diameter_mm:g} mm (pre-registered; ARCADE annotates its "
        "SYNTAX segments, which is not the same set).",
        "",
        "| Split | Images | Two trees | Labelled vessels | Unlabelled tree branches | Small "
        "branches | Catheter | Labelled pixels | SHA-256 (images + masks) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        *(f"| {s} | {v['n']} | {v['two_trees']:.0%} | {v['labelled_vessels']:.1f} | "
          f"{v['unlabelled_vessels']:.1f} | {v['small_branches']:.1f} | {v['catheter']:.0%} | "
          f"{v['mask_fraction']:.1%} | `{v['sha256'][:16]}…` |" for s, v in stats.items()),
        "",
        "![Examples](figures/synth_angio_examples.png)",
        "",
        f"Config `{cfg.name}` (seed {cfg.seed}; seeds per split in the manifest). Commit "
        f"`{manifest['git_commit']}` (dirty: {manifest['git_dirty']}).",
        "",
    ]  # fmt: skip
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "synth_angio_card.md").write_text("\n".join(lines), encoding="utf-8",
                                                    newline="\n")  # fmt: skip
    return stats
