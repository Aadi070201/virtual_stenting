"""Datasets of rendered frames in the exported format (docs/DATA_FORMAT.md).

Each image is one random scene (:func:`vcathlab.scene.sample_scene`) from its own seed; the
splits use disjoint seed ranges, so no patient appears in two splits. Arrays go to compressed
shards of ``shard`` images, the scene, truth and lesions of every image to one JSON line, and
a manifest records the config, the seeds, the git commit and each file's SHA-256. A dataset
card summarises what was drawn.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from vcathlab import DISCLAIMER
from vcathlab.config import StrictModel, git_info
from vcathlab.scene import SEGMENT_CODES, SceneConfig, sample_scene, simulate

FORMAT, VERSION = "vcathlab-frames", 1
SPLIT_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}
ARRAYS = ("images", "vessel", "segments", "overlap", "catheter")


class DatasetConfig(StrictModel):
    """A rendered-frame dataset."""

    name: str = "lesion_v1"
    out_dir: str = "data/lesion_v1"
    seed: int = 20261104
    n: dict[str, int] = Field(default_factory=lambda: {"train": 3000, "val": 300, "test": 300})
    shard: int = Field(200, ge=1)
    workers: int = Field(4, ge=1)
    scene: SceneConfig = Field(default_factory=SceneConfig)


def seeds(cfg: DatasetConfig, split: str) -> list[int]:
    """The scene seeds of ``split`` (disjoint between splits)."""
    base = cfg.seed + SPLIT_OFFSET[split]
    return [base + i for i in range(cfg.n.get(split, 0))]


def _render(args: tuple[int, dict]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seed, scene_cfg = args
    cfg = SceneConfig.model_validate(scene_cfg)
    scene = sample_scene(seed, cfg)
    f = simulate(scene, cfg)
    arrays = {"images": f.image, "vessel": f.vessel, "segments": f.segments,
              "overlap": f.overlap, "catheter": f.catheter}  # fmt: skip
    record = {"seed": seed, "scene": scene.model_dump(), "truth": f.truth, "lesions": f.lesions}
    return arrays, record


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_dataset(cfg: DatasetConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Render every split into ``cfg.out_dir``; returns the manifest."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    git = git_info()
    splits: dict[str, Any] = {}
    scene_cfg = cfg.scene.model_dump()
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        for split in cfg.n:
            todo = seeds(cfg, split)
            shards, records = [], []
            for first in range(0, len(todo), cfg.shard):
                chunk = todo[first : first + cfg.shard]
                results = list(pool.map(_render, [(s, scene_cfg) for s in chunk]))
                name = f"{split}-{first // cfg.shard:03d}.npz"
                np.savez_compressed(out / name, **{k: np.stack([a[k] for a, _ in results])
                                                   for k in ARRAYS})  # fmt: skip
                for k, (_, rec) in enumerate(results):
                    records.append({"index": first + k, **rec})
                shards.append({"file": name, "first": first, "count": len(chunk),
                               "sha256": _sha256(out / name)})  # fmt: skip
                log(f"  {split}: {first + len(chunk)} / {len(todo)}")
            rec_file = out / f"{split}.jsonl"
            rec_file.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            splits[split] = {"n": len(todo), "shards": shards, "records": rec_file.name,
                             "records_sha256": _sha256(rec_file)}  # fmt: skip
    manifest = {
        "format": FORMAT, "version": VERSION, "disclaimer": DISCLAIMER, "name": cfg.name,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), **git,
        "config": cfg.model_dump(), "segment_codes": SEGMENT_CODES, "splits": splits,
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "card.md").write_text(card(out, manifest), encoding="utf-8")
    return manifest


def _pct(n: int, d: int) -> str:
    return f"{n} ({100 * n / d:.0f}%)" if d else "0"


def card(root: Path, manifest: dict[str, Any]) -> str:
    """The dataset card: what was drawn, per split."""
    lines = [f"# Dataset card: {manifest['name']}", "", f"> **{DISCLAIMER}**", "",
             f"Rendered by `vcathlab dataset` (format {manifest['format']} v{manifest['version']},"
             f" commit `{manifest.get('git_commit')}`, dirty: {manifest.get('git_dirty')}). Every "
             "image is a simulated patient; no patient data.", ""]  # fmt: skip
    for split, info in manifest["splits"].items():
        recs = [json.loads(x) for x in (root / info["records"]).read_text("utf-8").splitlines()]
        les = [le for r in recs for le in r["lesions"]]
        sig = [le for le in les if le["ds_pct"] >= 50]
        n_les = Counter(len(r["lesions"]) for r in recs)
        px = np.array([r["truth"]["iso_pixel_mm"] for r in recs])
        bins = Counter("< 50%" if le["ds_pct"] < 50 else "50–69%" if le["ds_pct"] < 70
                       else "70–89%" if le["ds_pct"] < 90 else "≥ 90%" for le in les)  # fmt: skip
        lines += [
            f"## {split}: {len(recs)} images", "",
            f"- Arteries: {_pct(sum(r['scene']['artery'] == 'left' for r in recs), len(recs))} "
            "left, the rest right; views: "
            + ", ".join(f"{k} {v}" for k, v in Counter(r["truth"]["view"].split()[0]
                                                       for r in recs).most_common()) + ".",
            f"- Pixel size at the isocentre (mm): median {np.median(px):.3f}, range "
            f"{px.min():.3f}–{px.max():.3f}.",
            "- Lesions per image: " + ", ".join(f"{k}: {v}" for k, v in sorted(n_les.items()))
            + f" ({len(les)} lesions).",
            "- Severity: " + ", ".join(f"{k} {v}" for k, v in sorted(bins.items())) + ".",
            f"- Of the ≥ 50% lesions ({len(sig)}): inside the image (≥ half) "
            f"{_pct(sum(le['inside_share'] >= 0.5 for le in sig), len(sig))}; on an overlap "
            f"(≥ 30% of it) {_pct(sum(le['overlap_share'] >= 0.3 for le in sig), len(sig))}; "
            f"foreshortened {_pct(sum(le['foreshortened'] for le in sig), len(sig))}; eccentric "
            f"(< 0.8) {_pct(sum(le['eccentricity'] < 0.8 for le in sig), len(sig))}.", "",
        ]  # fmt: skip
    return "\n".join(lines) + "\n"
