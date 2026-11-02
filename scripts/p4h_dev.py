"""P4h development step (docs/PLAN.md): the lumen methods on CADICA's development patients only.

Measures intensity edges, dense_log and dense_linear on the development patients' expert boxes
(test patients are removed before anything is measured), applies the pre-registered rules for
the domain and the adoption, and saves the rows and the summary next to the git state.

    uv run python scripts/p4h_dev.py --workers 14
"""

import argparse
import json
from pathlib import Path

from vstent.config import git_info
from vstent.data.cadica import load_lesion_boxes
from vstent.inference.models import latest_model_dir
from vstent.inference.onnx_runtime import SEG_ONNX_FILE, SegmentationRunner
from vstent.validation import stenosis_grading as sg

METHODS = ("intensity", "dense_log", "dense_linear")
OUT = Path("data/derived/p4h")


def summary(stats: dict) -> dict:
    keep = ("n_lesions", "n_measured", "kappa", "auc_ge70", "auc_ge50", "spearman", "exact",
            "median_ds", "n_by_category")  # fmt: skip
    return {k: stats.get(k) for k in keep}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    git = git_info()
    tracks = sg.lesion_tracks(load_lesion_boxes())
    split = sg.patient_split([t.patient for t in tracks])
    dev = [t for t in tracks if split[t.patient] == "dev"]
    seg_dir = latest_model_dir(Path("models/seg"), SEG_ONNX_FILE)
    seg = SegmentationRunner.from_dir(seg_dir)

    def progress(k, n):
        if k % 250 == 0 or k == n:
            print(f"  {k} / {n} frames", flush=True)

    rows = sg.measure_tracks(dev, seg, split=split, methods=METHODS, workers=args.workers,
                             progress=progress)  # fmt: skip
    stats = {m: summary(sg.grading_stats(sg.track_grades(rows[m]))) for m in METHODS}
    # the pre-registered domain rule: higher development kappa; a tie → log
    domain = "linear" if stats["dense_linear"]["kappa"] > stats["dense_log"]["kappa"] else "log"
    chosen, edges = stats[f"dense_{domain}"], stats["intensity"]
    cadica_ok = {
        "kappa_higher": chosen["kappa"] > edges["kappa"],
        "auc_ge70_higher": chosen["auc_ge70"] > edges["auc_ge70"],
        "median_0_20_le_30": (chosen["median_ds"].get("p0_20") or 0) <= 30.0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_rows.json").write_text(json.dumps({"rows": rows, "git": git,
                                                   "seg_model": seg_dir.as_posix()}),
                                       encoding="utf-8")  # fmt: skip
    result = {"git": git, "seg_model": seg_dir.as_posix(), "stats": stats, "domain": domain,
              "cadica_conditions": cadica_ok}  # fmt: skip
    (OUT / "dev_summary.json").write_text(json.dumps(result, indent=1, default=float),
                                          encoding="utf-8")  # fmt: skip
    for m in METHODS:
        s = stats[m]
        print(
            f"{m}: kappa {s['kappa']:.3f}, AUC>=70 {s['auc_ge70']:.3f}, median DS "
            f"{ {k: round(v, 1) for k, v in s['median_ds'].items() if v is not None} }"
        )
    print(f"domain by the rule: {domain}; CADICA conditions: {cadica_ok}")


if __name__ == "__main__":
    main()
