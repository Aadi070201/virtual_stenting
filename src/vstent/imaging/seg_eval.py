"""Segmentation evaluation (``vstent eval-seg``) → ``reports/seg_eval.{md,json}`` + figures.

Per-image Dice of the thresholded probability map against the ARCADE ``syntax`` vessel masks
(union of the annotated segment polygons), at the original image size, in float32 (the
precision of the exported ONNX model). Every number in the report is computed here.
"""

import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vstent import DISCLAIMER, __version__
from vstent.config import CONFIG_FILENAME, META_FILENAME, git_info, load_config, load_yaml
from vstent.data.arcade import load_split
from vstent.imaging.seg_train import (
    METRICS,
    SUMMARY,
    SegConfig,
    load_checkpoint,
    predict_probs,
    resolve_device,
    validation_tensors,
)
from vstent.imaging.segment import THRESHOLD, dice_score, probability_to_image
from vstent.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    new_figure,
    save_figure,
    style_axes,
)

TARGET_DICE = 0.75
STRETCH_DICE = 0.80
EVAL_SPLITS = ("val", "test")


def pixel_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Pooled pixel precision, recall and Dice over all images (boolean (N, H, W))."""
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    tp = float((pred & target).sum())
    return {
        "precision": tp / max(float(pred.sum()), 1.0),
        "recall": tp / max(float(target.sum()), 1.0),
        "pooled_dice": 2.0 * tp / max(float(pred.sum() + target.sum()), 1.0),
    }


def _plot_hist(dice: dict[str, np.ndarray], path: Path) -> None:
    fig = new_figure(
        7.2,
        3.8,
        "Per-image Dice of the vessel mask",
        f"ARCADE syntax splits; dashed = target {TARGET_DICE} (mean val Dice)",
    )
    ax = fig.add_axes((0.08, 0.15, 0.88, 0.65))
    style_axes(ax)
    bins = np.linspace(0.0, 1.0, 41)
    for (split, d), color in zip(dice.items(), (SERIES_1, SERIES_2), strict=False):
        ax.hist(
            d,
            bins=bins,
            histtype="step",
            lw=1.8,
            color=color,
            label=f"{split} (n = {d.size})",
            zorder=2,
        )
    ax.axvline(TARGET_DICE, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.set_xlabel("Dice", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Images", fontsize=8, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    save_figure(fig, path)


def _plot_examples(images, masks, prob, dice, path: Path) -> None:
    order = np.argsort(dice)
    picks = [
        ("Highest Dice", order[-1]),
        ("Median Dice", order[len(order) // 2]),
        ("Lowest Dice", order[0]),
    ]
    fig = new_figure(
        9.0,
        3.9,
        "Validation examples: annotation vs prediction",
        "Green = ARCADE syntax annotation (union of segments); blue = predicted mask "
        f"(probability ≥ {THRESHOLD})",
    )
    axes = fig.subplots(1, 3, gridspec_kw={"wspace": 0.04})
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.08)
    for ax, (label, i) in zip(axes, picks, strict=True):
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        ax.contour(masks[i], levels=[0.5], colors=[SERIES_3], linewidths=0.8)
        ax.contour(prob[i], levels=[THRESHOLD], colors=[SERIES_1], linewidths=0.8)
        ax.set_title(f"{label}: {dice[i]:.3f}", loc="left", fontsize=8, color=INK_SECONDARY)
        ax.set_axis_off()
    save_figure(fig, path)


def _plot_training(run_dir: Path, path: Path) -> None:
    with (run_dir / METRICS).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig = new_figure(
        7.2, 3.8, "Validation Dice per epoch", f"Dashed = target {TARGET_DICE} (mean val Dice)"
    )
    ax = fig.add_axes((0.08, 0.15, 0.88, 0.65))
    style_axes(ax)
    ax.plot(
        [int(r["epoch"]) for r in rows],
        [float(r["val_dice"]) for r in rows],
        color=SERIES_1,
        lw=2.0,
        marker="o",
        ms=3,
        zorder=2,
    )
    ax.axhline(TARGET_DICE, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("Epoch", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Mean val Dice", fontsize=8, color=INK_SECONDARY)
    save_figure(fig, path)


def evaluate_seg_run(
    run_dir: str | Path,
    report_dir: str | Path = "reports",
    *,
    device: str = "auto",
    max_images: int | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate the best model of a run on ``syntax`` val and test; write the report."""
    log = log or (lambda _msg: None)
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    run_dir, report_dir = Path(run_dir), Path(report_dir)
    cfg = load_config(run_dir / CONFIG_FILENAME, SegConfig)
    dev = resolve_device(device)
    model = load_checkpoint(run_dir, dev)
    fig_dir = report_dir / "figures"
    splits: dict[str, Any] = {}
    dice_by_split: dict[str, np.ndarray] = {}
    for split in EVAL_SPLITS:
        data = load_split(cfg.raw_dir, "syntax", split, max_images)
        x, _ = validation_tensors(data.images, data.masks, cfg.model.image_size)
        prob_model = predict_probs(model, x, cfg.train.eval_batch_size, dev, amp=False)
        pairs = zip(prob_model, data.images, strict=True)
        prob = np.stack([probability_to_image(p, im.shape) for p, im in pairs])
        pred = prob >= THRESHOLD
        dice = dice_score(pred, data.masks)
        dice_by_split[split] = dice
        splits[split] = {
            "n": int(dice.size),
            "dice_mean": float(dice.mean()),
            "dice_median": float(np.median(dice)),
            "dice_p10": float(np.percentile(dice, 10)),
            "dice_min": float(dice.min()),
            **pixel_metrics(pred, data.masks),
        }
        log(f"{split}: mean Dice {dice.mean():.4f} (median {np.median(dice):.4f}, n = {dice.size})")
        if split == "val":
            _plot_examples(data.images, data.masks, prob, dice, fig_dir / "seg_examples.png")
    _plot_hist(dice_by_split, fig_dir / "seg_dice_hist.png")
    _plot_training(run_dir, fig_dir / "seg_training.png")

    # The plan's target is on val; val also chose the best epoch, so the held-out test split is
    # reported against the same thresholds as the unbiased estimate.
    targets = [
        {
            "metric": metric,
            "value": f"{dice:.4f}",
            "target": f"≥ {TARGET_DICE} (stretch {STRETCH_DICE})",
            "passed": dice >= TARGET_DICE,
            "stretch_passed": dice >= STRETCH_DICE,
        }
        for metric, dice in (
            ("Mean val Dice (plan target; val also chose the model)", splits["val"]["dice_mean"]),
            ("Mean test Dice (held out)", splits["test"]["dice_mean"]),
        )
    ]
    report = {
        "targets": targets,
        "targets_met": all(t["passed"] for t in targets),
        "splits": splits,
        "training": json.loads((run_dir / SUMMARY).read_text(encoding="utf-8")),
        "config": cfg.model_dump(mode="json"),
        "provenance": {
            "run": run_dir.as_posix(),
            "run_meta": load_yaml(run_dir / META_FILENAME),
            "eval": provenance,
            "device": str(dev),
            "torch": torch.__version__,
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "seg_eval.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    (report_dir / "seg_eval.md").write_text(_markdown(report), encoding="utf-8", newline="\n")
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    return report


def _markdown(report: dict[str, Any]) -> str:
    tr, cfg, prov = report["training"], report["config"], report["provenance"]
    lines = [
        "# Vessel segmentation evaluation (ARCADE)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent eval-seg`. Do not edit by hand.",
        "",
        "Reference masks are the union of the annotated SYNTAX segment polygons of each ARCADE "
        "`syntax` image (docs/DATA.md). Smaller side branches are not annotated, so predictions "
        "on them count as false positives. Dice per image at probability ≥ "
        f"{THRESHOLD}, float32, original image size.",
        "",
        "## Target",
        "",
        "| Metric | Value | Target | Pass |",
        "|---|---|---|---|",
        *(
            f"| {t['metric']} | {t['value']} | {t['target']} | "
            f"{'yes' if t['passed'] else 'NO'}"
            f"{' (stretch met)' if t['stretch_passed'] else ' (stretch not met)'} |"
            for t in report["targets"]
        ),
        "",
        "## Results",
        "",
        "| Split | Images | Mean Dice | Median | 10th percentile | Min | Pixel precision "
        "| Pixel recall | Pooled Dice |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split, m in report["splits"].items():
        lines.append(
            f"| {split} | {m['n']} | {m['dice_mean']:.4f} | {m['dice_median']:.4f} "
            f"| {m['dice_p10']:.4f} | {m['dice_min']:.4f} | {m['precision']:.4f} "
            f"| {m['recall']:.4f} | {m['pooled_dice']:.4f} |"
        )
    aug = cfg["augment"]
    lines += [
        "",
        "The test split was evaluated once, after training. Model selection (best epoch, "
        "early stopping) used val, so the val Dice is optimistic; the test Dice is the unbiased "
        "estimate.",
        "",
        "## Figures",
        "",
        "![Dice histogram](figures/seg_dice_hist.png)",
        "",
        "![Validation examples](figures/seg_examples.png)",
        "",
        "![Training curve](figures/seg_training.png)",
        "",
        "## Model and training",
        "",
        f"- U-Net (segmentation-models-pytorch {tr['smp']}), `{cfg['model']['encoder']}` encoder "
        f"with `{cfg['model']['encoder_weights']}` weights, 1 input channel, "
        f"{cfg['model']['image_size']} × {cfg['model']['image_size']} input, "
        f"{tr['n_parameters']:,} parameters.",
        f"- Loss BCE + {cfg['train']['dice_weight']} × soft Dice; AdamW lr {cfg['train']['lr']:g} "
        f"with cosine decay, batch {cfg['train']['batch_size']}, up to {cfg['train']['epochs']} "
        f"epochs, early stopping patience {cfg['train']['patience']} on mean val Dice.",
        f"- Augmentation: rotation ±{aug['rotation_deg']:g}°, scale {aug['scale'][0]:g}–"
        f"{aug['scale'][1]:g}, brightness ±{aug['brightness']:g}, contrast "
        f"{aug['contrast'][0]:g}–{aug['contrast'][1]:g}, gamma {aug['gamma'][0]:g}–"
        f"{aug['gamma'][1]:g}, Gaussian noise σ ≤ {aug['noise_std'][1]:g} "
        f"(p = {aug['p_noise']:g}), CLAHE (p = {aug['p_clahe']:g}); no flips.",
        f"- Best epoch {tr['best_epoch']} of {tr['epochs_run']}, "
        f"{tr['train_seconds'] / 60:.1f} min on {tr['gpu'] or tr['device']} (AMP: {tr['amp']}); "
        f"{tr['n_train']} train / {tr['n_val']} val images.",
        "",
        "## Provenance",
        "",
        f"- Training run `{prov['run']}`, commit `{prov['run_meta'].get('git_commit')}` "
        f"(dirty: {prov['run_meta'].get('git_dirty')})",
        f"- Evaluation commit `{prov['eval']['git_commit']}` (dirty: {prov['eval']['git_dirty']}), "
        f"vstent {prov['eval']['vstent_version']}, {prov['eval']['created_utc']}, "
        f"device {prov['device']}",
        "",
        "## Limitations",
        "",
        "- One public dataset (ARCADE), 512 × 512 frames; no external validation.",
        "- Masks cover only the annotated SYNTAX segments, not every visible vessel.",
        "- Segmentation quality on other scanners, frame rates or contrast protocols is unknown.",
        "",
    ]
    return "\n".join(lines)
