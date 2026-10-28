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

### D9 — Synthetic data storage (2026-10-28)
**Decision:** shards of 10,000 samples as uncompressed `.npz` (`{split}_{index:04d}.npz`); profile arrays (r, r_ref, FFR, ΔP) in float32, scalars in float64. The noisy profile is cast to float32 **before** labelling, so the stored labels are exactly the solver output for the stored profile. Reproducibility is checked with a SHA-256 over array contents (`manifest.json`), because `.npz` files embed timestamps. Shard seeds are `SeedSequence([split_seed, shard_index])`, so output does not depend on the number of workers.
**Reason:** ~0.5 GB instead of ~1 GB, fast loading for training, and labels that match the model input bit for bit.
**Alternatives:** compressed npz (noisy floats compress poorly, slower loads); HDF5/zarr (extra dependency); float64 (2× size, no benefit for labels at 1e-7 precision).

### D10 — Lesion DS sampling tuned to Beta(3, 3.5) (2026-10-28)
**Decision:** lesion throat DS ~ 20% + 70% × Beta(3.0, 3.5) instead of the first choice Beta(2, 2).
**Reason:** Beta(2, 2) gave 58.5% of training samples below FFR 0.80 but only 14.3% in the grey zone [0.75, 0.85] (target ≥ 15%). A 10k-sample sweep of seven Beta settings (PROGRESS.md, 2026-10-28) picked Beta(3, 3.5): ~18% grey zone, ~53% below 0.80, median FFR ~0.78, and 80% of lesions in 40–80% DS (uniform: 57%), so it still oversamples 40–80%. Only the sampling distribution changed; no physics constant was touched.
**Alternatives:** Beta(3, 4) (~19% grey zone but fewer severe cases); Beta(2.5, 3) (~17.5%, smaller margin).

### D11 — Synthetic geometry details not fixed by the plan (2026-10-28)
**Decision:** (a) lesions are placed by sampling lengths, shrinking them together only if they cannot fit (short vessels with several lesions), then spreading the free space uniformly at random, which guarantees ≥ 5 mm margins and ≥ 2 mm gaps without rejection sampling; (b) shapes: cosine; skewed cosine (throat at 20–80% of the length); plateau (flat 30–70% with half-cosine shoulders); irregular (cosine × 1 ± 0.3 smooth random sinusoids); each bump is normalised so the sampled throat DS is attained exactly; (c) diffuse disease is a plateau with smooth modulation over 30–80% of the vessel; (d) noise is white noise smoothed with a 1 mm Gaussian, rescaled to σ × r_ref, and the noisy radius is floored at 50% of the noise-free radius to keep r > 0 in severe lesions.
**Reason:** simple, deterministic given the seed, and every constraint in the plan holds by construction (tested).
**Alternatives:** rejection sampling for placement (unbounded loops for crowded vessels); multiplicative noise (would shrink noise in severe lesions, unlike QCA edge noise).

### D12 — Surrogate architecture details (2026-10-28)
**Decision:** the 1D U-Net (levels with 32/64/128/256 channels, GroupNorm, GELU, FiLM from a 2-layer MLP over the 5 global features) uses residual conv blocks and a **global-context step at the bottleneck**: mean- and max-pooled features plus the conditioning vector pass through an MLP that updates the FiLM conditioning used by the decoder. The monotone head is `FFR = 1 − [0, cumsum(softplus(z[1:]) / (N−1))]`, computed in float32, with the output conv zero-initialised and bias −1.5 so an untrained model starts at distal FFR ≈ 0.8.
**Reason:** flow Q, and therefore every local pressure drop, depends on the whole vessel (total viscous and expansion resistance); a mean-pool is a learned integral along the vessel that convolutions with a limited receptive field cannot provide. The head initialisation avoids a huge initial drop (softplus(0)·255 ≈ 177).
**Alternatives:** deeper U-Net or dilated convs for a global receptive field; self-attention at the bottleneck (heavier, less ONNX-friendly).

### D13 — Feature standardisation and training mechanics (2026-10-28)
**Decision:** per-point channels are standardised too (per channel, over samples and positions), not only the global features; statistics are saved in `norm_stats.json` with a feature version. The whole training split is kept on the GPU and batched by random permutation (`SurrogateDataset.batches`) instead of a per-sample `DataLoader`. Mixed precision is float16 autocast with a GradScaler; head and loss stay float32. Runs are named `runs/<name>-<UTC timestamp>/` so reruns never overwrite earlier runs.
**Reason:** the derivative channel has a very different scale from the ratio channels; GPU-resident batching removes Python collation overhead (≈ 20 s per 100k-sample epoch on the RTX 3060); float32 cumsum keeps the curve accurate.
**Alternatives:** DataLoader with workers (slower on Windows for small samples); bf16 autocast (fewer mantissa bits).

### D14 — Baseline features measured from the profile (2026-10-28)
**Decision:** the gradient-boosting baseline uses max DS and "lesion length" (total length with DS > 30%) measured from the noisy profile against the reference, plus D0, L, Pa and R_micro; it is fitted on the training split inside `vstent eval-surrogate` with the run's seed.
**Reason:** those are the quantities the app can measure; using the sampler's metadata (nominal DS, true lesion lengths) would give the baseline information the surrogate does not have.
**Alternatives:** metadata features (unfair comparison); per-lesion features (variable length, needs padding).

### D15 — Resumable training via a shared checkpoints/ folder (2026-10-28)
**Decision:** every trainer saves its full state (model, optimiser, scheduler, AMP scaler, torch CPU/CUDA RNG and its shuffle generator, metric history, early-stopping counters, accumulated train time) to `checkpoints/<run-name>/last.pt` after every epoch, written atomically (temp file + `os.replace`), plus a readable `state.json` (kind, status running/stopped/finished, epoch, best metric, resume command, resume history). A `STOP` file (`vstent stop <run-name>`) makes the trainer save and exit cleanly after the current epoch; `--resume <run-name>` reloads the config from the run directory and continues. The deliverables (config, metrics, `best.pt`) stay in `runs/<run-name>/`; `checkpoints/` is git-ignored. Implemented once in `vstent/checkpoints.py` for reuse by the segmentation trainer (M5).
**Reason:** the user asked to be able to pause and restart long trainings for any model. Restoring every piece of state makes a resumed run identical to an uninterrupted one (tested: same metrics and bit-identical weights on CPU); at most one epoch is lost if a process is killed instead of stopped.
**Alternatives:** keep the state inside `runs/<run>/` (mixes deliverables with large resumable state); restart from `best.pt` weights only (loses optimiser and schedule state, so not the same training); signal handlers for Ctrl+C (unreliable on Windows and for background processes).

### D16 — ONNX export with the TorchScript exporter at opset 17 (2026-10-28)
**Decision:** `surrogate/export.py` calls `torch.onnx.export(..., dynamo=False, opset_version=17)` with a dynamic batch axis (N fixed at 256) and embeds metadata (feature version, N, channel names, run, disclaimer) that `SurrogateRunner` checks against the normalisation statistics. Tracer warnings raised by GroupNorm's training-time batch-size check, and the exporter's own deprecation warning, are silenced during export; tests run the exported model at batch sizes 1, 7 and 33.
**Reason:** the plan asks for opset 17. The default exporter in torch 2.14 (`dynamo=True`) needs the extra `onnxscript` package and, for this model, cannot convert its opset-18 graph down to 17 (the ONNX version converter fails), so it falls back to opset 18. The TorchScript export matches torch to < 1e-6.
**Alternatives:** dynamo exporter at opset 18 (adds `onnxscript`, deviates from the plan; the route to take once torch removes the TorchScript exporter, since the pinned torch in `uv.lock` keeps it working now).

### D17 — Chunked, parallel ONNX inference in `SurrogateRunner` (2026-10-28)
**Decision:** `SurrogateRunner.predict` splits batches into chunks of 16 vessels and runs them in parallel on a thread pool (default: half the logical CPUs, i.e. 8 on the Ryzen 7 6800H), each ONNX Runtime call single-threaded. `chunk_size=None, workers=1` gives the plain single-call behaviour. Results do not depend on the chunking (tested to 1e-6).
**Reason:** with one call on a batch of 1,024 the latency target (< 300 ms) was missed (≈ 0.56–0.58 s; PyTorch on the same CPU takes about the same). Profiling showed ≈ 43% of the time in normalisation and element-wise ops and almost no gain beyond 4 threads, i.e. memory-bound: each first-level activation is ≈ 34 MB at batch 1,024, larger than the 16 MB L3 cache. Small chunks keep activations in cache; an exploration over chunk sizes 8–128 and 4–16 workers gave ≈ 0.2 s for chunks of 8–16 on 8–16 workers. The report `reports/onnx_benchmark.md` measures both configurations every time it is generated. Cost: single-vessel latency rises from ≈ 0.9 ms to ≈ 1.4 ms (one thread per call), still far below the 10 ms target.
**Alternatives:** a smaller model (retraining; deviates from the planned architecture and has nothing to do with the runtime); Conv1d → Conv2d rewrite to reach ONNX Runtime's NCHWc kernels (model change for a runtime concern); INT8 (D18).

### D18 — Dynamic INT8 quantisation evaluated, not kept (2026-10-28)
**Decision:** every `vstent export-surrogate` also builds a dynamically quantised INT8 model (unsigned 8-bit weights, `onnxruntime.quantization.quantize_dynamic`) in a temporary directory, measures its distal-FFR MAE on the whole test split and its batch latency, reports both next to FP32, and keeps it only if the MAE rises by < 0.002 (plan) **and** a batch is ≥ 1.1× faster. The INT8 file is not saved; the app uses FP32.
**Reason:** quantisation is only worth an accuracy loss if it buys speed. Measured on this CPU: the accuracy criterion is met but there is no speed-up (see the report), because the time goes to memory-bound element-wise ops, not to the convolutions that INT8 accelerates. Signed 8-bit weights were also tried once during exploration and were about 10× slower on this CPU, so the check uses unsigned weights.
**Alternatives:** static quantisation with calibration data (more work, same bottleneck); keep INT8 for the smaller file (4.3 MB is not a constraint).
