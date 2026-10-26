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
