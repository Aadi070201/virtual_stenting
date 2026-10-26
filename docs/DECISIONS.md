# Design decisions

One short paragraph per decision: decision, reason, alternatives.

---

### D1 — uv with Python 3.14 (2026-10-26)
**Decision:** manage the environment with `uv`; pin Python 3.14 in `.python-version`; keep `requires-python = ">=3.11"`.
**Reason:** uv gives a lock file (`uv.lock`) and reproducible installs on a fresh clone. Python 3.14.5 is already installed, and wheels for all heavy dependencies (torch, onnxruntime, opencv, pycocotools) exist for it on Windows.
**Alternatives:** plain `venv` + `pip` (still documented in the README as a fallback); Python 3.12 managed by uv (switch if a dependency breaks on 3.14).

### D2 — PyTorch from the CUDA 12.6 wheel index (2026-10-26)
**Decision:** on Windows and Linux, `torch` and `torchvision` resolve from `https://download.pytorch.org/whl/cu126` (`[tool.uv.sources]`). `torchvision` is listed directly in the `imaging` extra because uv sources only apply to direct dependencies; as a transitive dependency it came from PyPI as a CPU build whose ops failed on CUDA tensors.
**Reason:** the PyPI torch wheel for Windows is CPU-only. The dev machine has an RTX 3060 (driver 566.07, CUDA ≤ 12.7), so cu126 is the newest build it can run; cu130 needs a newer driver.
**Alternatives:** CPU-only torch (smaller download, much slower training); cu130 after a driver update.

### D3 — Dependency layout and build backend (2026-10-26)
**Decision:** core dependencies cover physics, configs, CLI and the ML/ONNX stack (torch, onnx, onnxruntime); extras are `imaging` (segmentation, OpenCV, COCO, DICOM), `app` (Streamlit, Plotly) and `dev` (pytest, ruff). Build with hatchling, src layout.
**Reason:** the surrogate and ONNX runtime are central to the MVP, while imaging and the app are separable. hatchling is widely supported by both uv and pip.
**Alternatives:** a separate `ml` extra (more friction for little benefit); `uv_build` backend.

### D4 — Configs and run provenance (2026-10-26)
**Decision:** every config is a Pydantic model deriving from `StrictModel` (unknown keys rejected, frozen). `save_run_metadata(run_dir, config)` writes `config.yaml` (fully resolved, defaults included) and `run_meta.yaml` (git commit, dirty flag, vstent and Python versions, platform, UTC time).
**Reason:** catches typos in YAML early, and makes every run reproducible from the files saved next to it.
**Alternatives:** Hydra/OmegaConf (heavier than needed for an MVP).

### D5 — One vectorised solver implementation (2026-10-26)
**Decision:** `solve_ffr_batch` (NumPy over shape (B, N)) is the only implementation of the physics; `solve_ffr` calls it with B = 1. Diverging runs are labelled with a flattened cumulative count of run starts and aggregated with `np.bincount`, so there is no per-profile Python loop.
**Reason:** a single source of truth means the dataset labels (M2), the app and the tests cannot drift apart; the loop-free version reaches ~50k profiles/s, 10× the target.
**Alternatives:** a readable per-profile loop plus a separate vectorised version (two implementations to keep consistent); Numba (an extra dependency for no need).

### D6 — Physics config: pressures in mmHg, everything else SI (2026-10-26)
**Decision:** `PhysicsParams` stores `pa_mmhg` / `pv_mmhg` (the config file is an input boundary that humans edit) and exposes `.pa` / `.pv` in Pa; solver code only uses the SI properties. Other constants are already SI.
**Reason:** keeps the YAML readable and clinically familiar without letting mmHg leak into the solver.
**Alternatives:** Pa in the YAML (error-prone to edit); a units library such as pint (overkill).

### D7 — R_micro floor (2026-10-26)
**Decision:** the spec's "clamp R_micro > 0" is implemented as a floor at 1% of (Pa − Pv)/Q_ref, with the sample flagged `r_micro_clamped`.
**Reason:** a strictly positive but meaningful floor keeps the quadratic well-posed; the flag makes any clamped sample visible so it can be filtered or counted in M2. It is a numerical guard, not a tuned physics constant, and is never reached for realistic coronary geometry.
**Alternatives:** clamp to a tiny epsilon (numerically fine but physically meaningless flows); reject the sample (loses information about why).

### D8 — `vstent physics-report` command (2026-10-26)
**Decision:** a CLI command computes the M1 sanity values, the batch throughput and the FFR-vs-%DS sweep, and writes `reports/figures/ffr_vs_ds.png` plus `reports/physics_checks.{json,md}` with config and git hash.
**Reason:** the M8 MVP report must be assembled from generated files only; this is the physics part of it.
**Alternatives:** a standalone script outside the package (not covered by tests, easy to forget).
