# vcathlab: a virtual cath lab

> **Research use only.** Simulated data for developing and testing methods. Not a medical
> device, not clinical data, and never a substitute for validation on real patients.

A virtual patient's coronary arteries, seen through a virtual C-arm: an anatomical heart with
named coronary segments, lesions you place and grade, standard projections, and an X-ray image
with **every label known**: vessel and segment masks, lesion boxes with their true severity,
where vessels overlap, the catheter's size and the true millimetres per pixel.

Planned next: the beating heart with an ECG (cine runs), an editor app, and pressure and flow
through the heartbeat (FFR, iFR) from its own physics.

## Why a separate app

The cath lab produces data with ground truth; the vstent planner (in the parent folder) is
trained and tested on it. The two never import each other. Their only link is the dataset
format in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). Keeping the simulator independent means a
model is never checked by its own code.

## Setup and commands

```bash
cd cathlab
uv sync --all-extras
uv run ruff check . && uv run ruff format --check . && uv run pytest -q   # checks
uv run vcathlab frame --seed 0 --out frame/       # one random patient, one view, all labels
```

Plan and decisions: [docs/PLAN.md](docs/PLAN.md), [docs/DECISIONS.md](docs/DECISIONS.md).
