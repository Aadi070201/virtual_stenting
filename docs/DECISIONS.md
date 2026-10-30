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
**Decision:** `SurrogateRunner.predict` splits batches into chunks of 8 vessels and runs them in parallel on a thread pool (default: all logical CPUs, i.e. 16 on the Ryzen 7 6800H), each ONNX Runtime call single-threaded; batches of ≥ 512 vessels are also pre-processed in parallel slices. (Updated in M6: the first version used chunks of 16 on 8 threads. Re-measured for the planner, 8 on 16 was ≈ 7% faster on 5,000 vessels and, with parallel pre-processing, brings the planner's 5,000-candidate evaluation under 1 s; the M4 report was regenerated with these defaults.) `chunk_size=None, workers=1` gives the plain single-call behaviour. Results do not depend on the chunking (tested to 1e-6).
**Reason:** with one call on a batch of 1,024 the latency target (< 300 ms) was missed (≈ 0.56–0.58 s; PyTorch on the same CPU takes about the same). Profiling showed ≈ 43% of the time in normalisation and element-wise ops and almost no gain beyond 4 threads, i.e. memory-bound: each first-level activation is ≈ 34 MB at batch 1,024, larger than the 16 MB L3 cache. Small chunks keep activations in cache; an exploration over chunk sizes 8–128 and 4–16 workers gave ≈ 0.2 s for chunks of 8–16 on 8–16 workers. The report `reports/onnx_benchmark.md` measures both configurations every time it is generated. Cost: single-vessel latency rises from ≈ 0.9 ms to ≈ 1.4 ms (one thread per call), still far below the 10 ms target.
**Alternatives:** a smaller model (retraining; deviates from the planned architecture and has nothing to do with the runtime); Conv1d → Conv2d rewrite to reach ONNX Runtime's NCHWc kernels (model change for a runtime concern); INT8 (D18).

### D18 — Dynamic INT8 quantisation evaluated, not kept (2026-10-28)
**Decision:** every `vstent export-surrogate` also builds a dynamically quantised INT8 model (unsigned 8-bit weights, `onnxruntime.quantization.quantize_dynamic`) in a temporary directory, measures its distal-FFR MAE on the whole test split and its batch latency, reports both next to FP32, and keeps it only if the MAE rises by < 0.002 (plan) **and** a batch is ≥ 1.1× faster. The INT8 file is not saved; the app uses FP32.
**Reason:** quantisation is only worth an accuracy loss if it buys speed. Measured on this CPU: the accuracy criterion is met but there is no speed-up (see the report), because the time goes to memory-bound element-wise ops, not to the convolutions that INT8 accelerates. Signed 8-bit weights were also tried once during exploration and were about 10× slower on this CPU, so the check uses unsigned weights.
**Alternatives:** static quantisation with calibration data (more work, same bottleneck); keep INT8 for the smaller file (4.3 MB is not a constraint).

### D19 — ARCADE masks and splits (2026-10-28)
**Decision:** the vessel mask of an image is the union of all its `syntax`-task segment polygons, rasterised with pycocotools. Segmentation trains on `syntax/train`, selects the model on `syntax/val` (mean per-image Dice) and reports `syntax/val` and `syntax/test`. The `stenosis` task is used only for overlays (its boxes on `stenosis/val` images). Duplicates are left as provided. Inspection details are in docs/DATA.md.
**Reason:** the plan specifies the union of segment polygons. Pixel-level hashing showed the `syntax` splits are disjoint, the 5 duplicates sit inside `syntax/train` (no leakage), and the only cross-task overlap is 123 images between the two test splits, which never meet in training.
**Alternatives:** per-segment multi-class segmentation (not needed for QCA); merging both tasks (`stenosis` images have no vessel polygons).

### D20 — Segmentation training set-up (2026-10-28)
**Decision:**
- Model: smp U-Net with a `resnet34` encoder, ImageNet weights from Hugging Face `smp-hub/resnet34.imagenet` (pinned revision in smp 0.5.0), `in_channels=1`, 512 px input.
- Input scaling: the inputs are standardised with the channel-mean ImageNet statistics (0.449 / 0.226) in `imaging/segment.py`, shared with ONNX and the app.
- Loss: BCE + soft Dice.
- Optimiser: AdamW 3e-4 with cosine decay, batch 8, AMP, early stopping on mean val Dice.
- Augmentation: done on the CPU in 2 DataLoader workers. Each sample's random generator is seeded from (a per-epoch seed drawn from the checkpointed generator, sample index), so a resumed run repeats the same augmentations regardless of worker count; stop and resume were tested to be bit-identical on CPU.
- Memory format: the default NCHW layout. `channels_last` was measured to be 4× slower per step on the RTX 3060 Laptop (694 vs 182 ms per batch of 8); cuDNN benchmark mode is on.
**Reason:** the plan's model and loss. Two workers keep augmentation (≈ 7 ms per image) off the critical path without copying the image arrays (≈ 0.5 GB) into more processes on Windows.
**Alternatives:** `efficientnet-b0` encoder (allowed by the plan; resnet34 reached the target first); GPU augmentation (CLAHE is awkward on the GPU); `channels_last` (slower here).

### D21 — QCA method and the pixel-size assumption (2026-10-28)
**Decision:** centreline = pruned skeleton (spurs < 10 px removed), shortest path between the snapped user points, Savitzky–Golay smoothing (15 px window, order 3), resampled every 1 px. The diameter is measured along the normal at the 0.5 crossings of the probability map: bilinear sampling every 0.1 px, linear sub-pixel interpolation of each crossing, with 2 × distance transform as the fallback. One median diameter is trimmed from each end of the centreline: on the first ARCADE run the "MLD" (0.19 mm) was the tapering tip of the mask, an end effect rather than a narrowing. Real images added three rules:
- **One-sided branch:** if one edge is much farther than the other (> 1.5 × + 1 px, or beyond the search range), the point is next to a side branch and 2 × the nearer edge distance is used.
- **Centreline outside the vessel:** where smoothing cut a sharp bend and the point left the vessel, the diameter is interpolated from its neighbours.
- **Reference and capping:** the reference is a two-sided robust line, a Theil–Sen start refined on points within 0.85–1.15 × the line. Diameters above 1.15 × the reference are capped before the physics, because a spike from an overlapping vessel would otherwise act as a narrowing plus an expansion loss (ARCADE val 107: reference 16.6 mm before, 2.3 mm after).

The M1 estimator is left unchanged for synthetic profiles, which have no branches. Pixels are converted to mm with 0.30 mm/px unless a calibration is given; `QCAResult.calibration_assumed` and the pipeline warnings say so. Reference diameter, %DS, lesions and MLD reuse the M1 functions on the SI `VesselProfile`.
**Reason:** the plan's method. `tests/test_qca.py` checks the plan's targets on five synthetic curved tubes with blur, noise and a 60% stenosis (median diameter error ≤ 0.7 px, MLD within 3 px, %DS within ±7 points) and all pass; an exploratory run during development (not a saved report) showed median errors around 0.1 px.
**Alternatives:** distance transform only (quantised to ±0.5 px); edge detection on the raw image (the probability map is smoother and is what the model outputs).

### D22 — End-to-end pipeline conventions (2026-10-28)
**Decision:** `inference/pipeline.py` (torch-free) runs segmentation → QCA → the solver on the QCA profile; R_micro is calibrated by the solver on the QCA reference, and the same Pa, Pv and R_micro go to the surrogate. Warnings are returned for: an assumed pixel size, > 10% fallback diameters, R_micro at its floor, and vessel length, proximal reference diameter, Pa or Pv outside the surrogate's synthetic training ranges (`SamplingConfig`). Without user points, `longest_path_endpoints` picks the longest path of the largest skeleton component (double sweep) and orders it wider end first, so flow enters proximally (the first ARCADE run had picked it distal → proximal). Auto-selected paths often run the whole artery into thin distal branches; with constant flow and no side branches the model then overstates the pressure loss, so their FFR is illustrative only.
**Reason:** solver and surrogate must see identical inputs to be comparable. Real angiogram paths can be longer than the 30–80 mm training vessels, especially at an assumed pixel size, and the user must know when the surrogate extrapolates.
**Alternatives:** clipping inputs to the training range (hides the problem); refusing out-of-range vessels (the solver result is still valid).

### D23 — Virtual stent model (2026-10-28)
**Decision:** `planning/stent.py` sets r_new = r + w(s)·max(0, r_stent − r), with r_stent = expansion · nominal diameter / 2. The weight w is 1 inside the stent and ramps from 0 to 1 with a half-cosine over the first and last `edge` (1 mm) **inside** the nominal stent ends, so a stent never acts outside its own length. Several stents combine through the maximum. The reference radius is kept. Positions and sizes are in metres (the plan's `edge_mm` became `edge` in m; the YAML configs stay in mm, as for mmHg in D6). Post-PCI evaluation keeps Pa, Pv and **R_micro at their pre-PCI values**: a stent changes the epicardial geometry, not the microvasculature. For this, `solve_ffr_batch` gained an optional per-sample `r_micro`; no physics constant changed.
**Reason:** the lumen is never reduced (plan); blending inside the ends is conservative and keeps landing zones meaningful. Re-calibrating R_micro after stenting happens to give the same value (the reference is unchanged), but fixing it states the physiology explicitly.
**Alternatives:** blending outside the stent ends (makes every stent effectively 2 mm longer); a step change at the ends (an artificial expansion loss at the distal edge).

### D24 — Plan optimiser (2026-10-28)
**Decision:** see the module docstring of `planning/optimizer.py`.
- **Planning grid:** every profile is resampled to 256 points (the surrogate's N) with its reference made explicit.
- **Candidates:** catalogue diameters × lengths × start positions on a 0.5 mm grid, each covering at least one lesion MLD (DS > 30%).
- **Constraints, vectorised:** the stent lies inside the vessel; nominal diameter ≤ 1.1 × the reference at the distal stent end; maximum DS < 30% over the first and last **2 mm** of the stent (the landing zones; range maxima via a sparse table plus the interpolated window ends).
- **Ranking:** plans reaching the target FFR (0.90) first, by total length, ties broken by higher FFR (so at equal length the largest allowed diameter wins); otherwise by FFR.
- **Two stents (with ≥ 2 lesions):** for each lesion a pool is kept with the best candidates *per stent length* (top_k / number of lengths). Without overlap, each pair is evaluated once.
- **Pareto front:** total length vs FFR over all evaluated plans.
- **Surrogate backend:** the returned plans are re-evaluated with the solver (`Plan.ffr_solver`).
**Reason:** the plan's objective and constraints. Keeping only the top candidates by FFR made every pool consist of the longest stents, and the first tandem test then returned 38 + 12 mm instead of 12 + 12 mm. The solver re-check shows how far the surrogate's choice can be trusted.
**Alternatives:** exhaustive pairs (≈ 10⁶ for two lesions; too slow with the surrogate); a tie-break on diameter closest to the reference (clinically appealing, but not what the plan specifies); overlapping stents for lesions longer than 38 mm (not supported in the MVP).

### D25 — Streamlit app design (2026-10-28)
**Decision:**
- **Code split:** `app/core.py` holds all logic without Streamlit: inputs, presets, pre/post-PCI evaluation, Plotly figures, the image overlay and the case report. `app/streamlit_app.py` only wires widgets.
- **Caching:** segmentation (per image SHA-256) and QCA + pipeline (per image, points, calibration and pressures) are cached with `st.cache_resource`, so a stent-control change only re-runs the stent, both FFR models and the plots.
- **Post-PCI physics:** uses the pre-PCI R_micro, Pa and Pv (D23).
- **Points:** chosen by clicking (`streamlit-image-coordinates`), with "Auto path" and a manual row/col form as alternatives that tests can drive.
- **State reset:** a new vessel (preset/seed, or image + points) resets the stent controls and the plan selection. The browser check showed the previous case's stent being applied while the toggle looked off.
- **Calibration:** "assumed 0.30 mm/px" unless entered, or taken from DICOM `ImagerPixelSpacing` with a note that magnification is not corrected. DICOM uses the first frame.
- **Case report:** prepared on demand (JSON + matplotlib PNG of the plots) with the disclaimer, every assumption and the model provenance, and **no image pixels** (only the SHA-256 and size).
- **Launch:** `vstent app` binds to **localhost** by default (`--address` to change).
- **Testing hooks:** environment variables `VSTENT_SURROGATE_MODEL`, `VSTENT_SEG_MODEL` and `VSTENT_ARCADE_DIR` override the model and example folders for tests. The app shows its own server-side rerun time, which `vstent app-latency` reads.
**Reason:**
- The < 300 ms target applies to updates after segmentation, so the slow steps must be cached and everything else kept to a few tens of ms (measured ≈ 50–80 ms).
- Testable logic outside Streamlit keeps the app thin.
- Binding to localhost and keeping pixels out of reports limit accidental exposure of uploaded images.
**Alternatives:** Plotly → PNG export via kaleido (an extra Chrome-dependent package; matplotlib is already used for all report figures); launching on all interfaces (Streamlit's default).

### D26 — Demo and the assembled MVP report (2026-10-28)
**Decision:**
- **`vstent demo`** runs the tandem-lesion preset through the whole chain: profile → FFR → plan optimisation → best plan → post-PCI FFR → `reports/demo/` (Markdown, JSON case report, figure). It uses the newest exported surrogate if there is one, else the solver, and says which.
- **`vstent qca-synthetic`** turns the QCA test cases into a report; `tests/test_qca.py` imports the same cases and metric code.
- **`vstent mvp-report`** assembles `reports/mvp_report.md` only from the other reports' JSON files:
  - a target table normalised from each report's own target list;
  - facts without targets;
  - a provenance table (the commit and dirty flag of every source);
  - fixed prose limitations without numbers;
  - the Phase 2 list parsed from `docs/PLAN.md`.

  Missing reports are listed rather than skipped. Each report is generated from a clean commit and committed before the next one is generated, so every provenance line reads `dirty: False`.
**Reason:** the plan requires no hand-typed numbers; the demo must work on a fresh clone, where no trained model exists (models are never committed).
**Alternatives:** a notebook (not a source of truth here); templated numbers in Markdown (drift from the data).

### D27 — Phase 2 ordering (2026-10-28)
**Decision:** Phase 2 runs as milestones P1–P8 (`docs/PLAN.md`), in this order:
- **P1, uncertainty:** the quickest step and the one that makes every number safer to read.
- **P2, P3, vessel trees with side branches** (physics, image path, graph surrogate): they fix the largest known modelling error seen on ARCADE.
- **P4, validation readiness:** tools and protocol only, until a hospital partner exists.
- **P5, two-view geometry:** built on synthetic data first.
- **P6, 3D CFD labels** and multi-fidelity training.
- **P7, cine** angiography.
- **P8, sim-to-real** angiograms and regulatory drafts.

Items that depend on the human (data registration, compute budget, a hospital partner, a regulatory reviewer) are marked **[needs human]**.
**Reason:** the MVP showed three main gaps: no confidence information, single-tube physics without side branches, and no comparison with measured FFR. P1–P3 can be done locally and remove the first two. The third needs a partner, so P4 prepares everything that can be prepared without patient data.
**Alternatives:** starting with 3D CFD (the original Phase 2 headline). It needs dataset registrations, meshing and simulation tooling, and probably more compute than this laptop for a training-sized dataset, so it would block on decisions only the human can make.

### D28 — Surrogate ensemble: training, export and planner screening (2026-10-30)
**Decision:**
- **Training.** The ensemble is the M3 configuration trained with seeds 0–4 (`vstent train-ensemble`, `configs/ensemble.yaml`). Each member is an ordinary resumable run under `runs/<ensemble>/members/`. The ensemble keeps its own checkpoint status, so `vstent stop <ensemble>` stops the member that is training and `--resume` continues where it stopped.
- **Export.** It writes one ONNX graph with two outputs: `ffr` (the members' mean) and `ffr_std` (their sample SD). It also writes `screen.onnx`, which holds member 0 alone.
- **Planner.** The planner screens its thousands of candidates with `screen.onnx`, then re-scores and re-ranks the returned `n_plans` with the full ensemble. Re-scoring gives each plan its mean, SD and conformal interval. The Pareto front stays on the screening values.
**Reason:** The ensemble costs 5× the single model per vessel. That is a few milliseconds per vessel for the app and `analyse`, but it would push the 5,000-candidate planner far past its 1 s target (M6). Member 0 and the ensemble mean differ by about the surrogate's own error (≈ 0.001 FFR), so screening with one member changes the candidate ranking very little. The plans shown carry the ensemble's values.
**Alternatives:**
- Five separate ONNX files, averaged in Python. This means five sessions and five thread pools, and the plan asks for mean and spread in the ONNX export.
- Screening with the full ensemble. Too slow.
- A distilled single model that predicts its own variance. This needs new training and is not yet validated.

### D29 — Split-conformal interval with an ensemble-normalised score (2026-10-30)
**Decision:**
- **Score.** The model interval for distal FFR is split-conformal on the whole `cal` split, using the score |y − ŷ| / (σ + f) with the floor f = 0.0002. Here ŷ is the ensemble mean, σ the ensemble SD of distal FFR, and y the solver.
- **Interval.** The interval is ŷ ± q̂·(σ + f), capped at 1, where q̂ is the ⌈(n + 1)·0.9⌉-th smallest score.
- **When it is calibrated.** Calibration happens at export with the ONNX model (`conformal.json`), so it matches what the app runs.
- **Comparison.** The report also shows an absolute score (one width for every vessel).
- **Calibration split.** `cal` (10,000 vessels, seed 1005) is drawn exactly like `test` and is used for nothing else. Splits are seeded independently, so adding it left the other splits byte-identical (same shard hashes). Calibration first used `val`, but `val` had chosen each member's best epoch and the σ floor, so its scores were not exchangeable with new vessels and the guarantee did not formally hold. That version gave 90.01% on `val` and 90.15% on `test`; the change was made after the code review of P1.
- **The floor.** The floor was first set to 0.001 by assumption. With the trained ensemble, the median σ on `val` is 0.00073 and 65% of `val` vessels have σ < 0.001. A 0.001 floor would therefore make the score half absolute: on `val`, coverage by σ quintile ranged from 98% (lowest σ) to 77% (highest σ). The floor is now the 10th percentile of σ on `val` (0.0002), so it only acts where the members agree almost exactly. Coverage by σ quintile is then 85–94% on `val`. Of three floors compared (0.001, 0.0005, 0.0002), this one gives the most even coverage. An absolute score ranges from 100% to 63%. The choice used `val` only, before any `test` evaluation.

**Reason:** A normalised score widens the interval where the members disagree, which happens out of distribution and at severe stenoses. The guarantee is the standard marginal one: at least 90% for new vessels drawn like `cal`, because nothing else was fitted or chosen on `cal`. Coverage on `test` is the empirical check.
**Alternatives:**
- Conformalised quantile regression, which needs quantile heads and retraining.
- Calibrating on `val` (first version): no extra data, but `val` was used for model selection.
- Raw ±1.645σ ensemble intervals, which are not calibrated.

### D30 — Input-uncertainty Monte Carlo: what is varied and how (2026-10-30)
**Decision:** Each case gets 500 fixed-seed Monte Carlo draws through the solver. Each assumption varies over half the width of its synthetic sampling range, centred on the value used:
- hyperemic velocity ±0.10 m/s;
- Pa ±20 mmHg;
- Pv ±3 mmHg, kept ≥ 0.

Two assumptions apply to image cases only:
- **Pixel size:** ±10%, and only when it was assumed. All lengths and diameters scale together.
- **QCA diameter error:** a Gaussian field with the SD and correlation length measured on rendered tubes (`reports/qca_synthetic.json`, now part of `vstent qca-synthetic`). The reference moves by the path mean of the error.

For post-PCI, the same draws are reused with each draw's own pre-PCI R_micro. A stent keeps its physical length and nominal diameter, and its start scales with the pixel size, because it is placed where it was drawn on the image.

The model interval and the input interval are reported side by side, not combined. The grey-zone flag is raised when either interval contains the threshold: 0.80 pre-PCI, the planner target post-PCI.
**Reason:**
- **Ranges.** The sampling ranges are the only ranges in the repo with a documented basis, and using them needs no invented clinical numbers. The plan asks for ±10% pixel size explicitly.
- **Fixed seed.** Fixed draws make the interval reproducible and stable while a slider moves. They also let pre/post differences cancel noise.
- **Separate intervals.** They answer different questions: surrogate vs solver, and inputs vs the solver. Adding them would suggest a total uncertainty that excludes the largest unknown, the reduced-order model's own error.
**Alternatives:**
- Gaussian ranges with clinical SDs: no source in the repo.
- Combining the two intervals by summing variances.
- Scaling the stent length with the pixel size, which would hide the "stent too short" risk when the pixel size is underestimated.

### D31 — Early stopping only after the learning rate has decayed (2026-10-30)
**Decision:** Early stopping on validation distal MAE (patience 10) counts only from epoch 40 of 50 (`train.min_epochs: 40`), when the cosine learning rate is below 10% of its peak. The first P1 ensemble (`runs/unet1d_ens5-20261030-100555`) was discarded and retrained with this setting.
**Reason:** In the first ensemble, seed 3 hit a low validation value at epoch 11 (0.0030, a high-learning-rate fluctuation) and was stopped at epoch 21 with the learning rate still at 62% of its peak. Its best validation distal MAE stayed at 0.0030. The four other seeds trained to their best at epoch 49–50 (0.0013–0.0014). During the high-learning-rate phase the validation error swings between 0.003 and 0.007 for every seed, so a 10-epoch patience can fire on noise. The decision was made on `val` only, before any test-split evaluation. The M3 single model trained all 50 epochs, so its result is unaffected.
**Alternatives:**
- Keeping the weak member: an avoidable training artefact.
- Retraining only seed 3: mixed member configs make provenance harder.
- No early stopping at all: equivalent in practice for this schedule, but this keeps the guard for longer schedules.

### D32 — Vessel trees: continuous branches, outlet-flow Newton (2026-10-30)
**Decision:** A tree is a set of branches.
- **Branches.** Each branch keeps one continuous radius profile and its own M1 coefficients a(s) and b(s). Side branches start at a sample inside their parent.
- **Flow.** The flow is piecewise constant along a branch, dropping at each take-off. A segment with flow Q loses Δa·Q + Δb·Q².
- **Unknowns.** The unknowns are the outlet flows (one per branch). The outlet equations are solved for a padded batch of trees by Newton's method, starting from the linear solution.
- **Output.** Pressures are then integrated down every branch.

**Reason:**
- **M1 stays exact.** Keeping a branch continuous means the M1 expansion-loss runs are found on the whole branch, so a lesion that spans a take-off keeps its full expansion loss. Cutting the branch there would have split the run and underestimated the loss, because a sum of squares is less than the square of the sum. A test checks this: a side branch carrying almost no flow inside a lesion's expansion run changes main-vessel FFR by < 2e-4.
- **The solve is well-posed.** The outlet equations are the gradient of a strictly convex function, so the solution is unique and Newton is well-behaved (about 5 iterations). The systems are tiny (at most 5 outlets), so batching gives ~4,500 trees/s on one CPU thread against the 1,000 trees/s target.
- **One branch is M1.** A one-branch tree is exactly the M1 solver (test, to 1e-12).

**Alternatives:**
- Cutting branches at take-offs into independent tubes: simpler, but it loses expansion losses at bifurcation lesions.
- Solving for junction pressures: equivalent, but harder to batch across trees with different topologies.
- A general sparse network solver: unnecessary for trees.

### D33 — Tree calibration: Murray split at junction diameters, viscous-only reference (2026-10-30)
**Decision:**
- **Healthy flow split.** In the stenosis-free tree, the root carries Q_ref = v_hyp·π·r_ref,root(0)². The flow arriving at each junction splits between the side branch(es) and the continuing parent in proportion to d^k, with k = 2.33 (`murray_exponent` in `configs/physics.yaml`). d is the side branch's reference diameter at its start and the parent's one sample after the take-off.
- **Outlet resistances.** Each outlet resistance is R_j = (P_j − Pv)/Q_j, from the healthy pressures with viscous loss only, floored like M1.
- **After a virtual stent.** The pre-PCI outlet resistances are kept, as M1 keeps R_micro.

**Reason:**
- **Murray split.** The plan asks for a Murray split ("flow ∝ diameter^k"). Applying it at each junction makes the healthy split directly testable and independent of how much each branch tapers downstream. k = 7/3 is the coronary scaling fit of Huo & Kassab (2012).
- **Viscous-only calibration.** It matches M1 exactly (M1 calibrates on a_ref only), so a one-branch tree reproduces M1.
- **No new constants.** Pa, Pv and v_hyp keep their M1 meaning.

**Alternatives:**
- Splitting by outlet diameters (common in CT-FFR outlet boundary conditions): it depends on each branch's taper and on where the path was cut, which is arbitrary in 2D QCA.
- Splitting by territory or myocardial mass: there is no such information in a single angiogram.
- k = 3 (original Murray's law): available through the config.

### D34 — One solver geometry per case; the surrogate keeps its 256-point input (2026-10-30)
**Decision:**
- **Planner.** The planner's solver (solver backend, solver check of every plan, pre-PCI value and R_micro) runs on the profile's own points. That is the geometry the app, `vstent analyse` and the image pipeline solve on. Candidates, lesion detection and the surrogate stay on the 256-point planning grid. Stent positions are in metres, so both grids share them.
- **Surrogate input.** The surrogate keeps resampling its input to 256 points with linear interpolation.

**Reason:**
- **The bug.** The MVP code review found that image paths (360–890 points) were solved on the native grid in the app but on the 256-point resample in the planner. A plan's "Solver check" could then differ from the app's own post-PCI solver value for the same stents (0.765 vs 0.752 on one ARCADE image), and `meets_target` could flip near 0.90. Synthetic vessels already have 256 points, so the planner benchmark, demo and uncertainty report are unchanged.
- **Why not resample everything.** Resampling the whole case to 256 points would coarsen the physics for long paths.
- **Why the surrogate input stays.** Downsampling (which can step over a narrow throat) happens only for paths longer than 256 px, about 77 mm at 0.30 mm/px. That is at or beyond the surrogate's 30–80 mm training range, where the app and `vstent analyse` already warn of extrapolation. The surrogate's reported error on image cases therefore includes this resampling, which is what it really does.

**Alternatives:**
- Resampling every case once to 256 points with block minima: consistent, but it loses resolution and biases throats.
- An anti-aliased (resistance-preserving) surrogate resample: possible later, if image cases within the training length show aliasing.

### D35 — Vessel trees from the image: shortest-path tree, one level of side branches (2026-10-30)
**Decision:** `imaging/qca_tree.py` builds the tree from the pruned skeleton.
- **Main path.** The shortest paths from the start point reach every endpoint. The main path ends at the chosen end, or at the farthest point.
- **Side branches.** The other endpoints are taken longest first. Each joins at the last pixel it shares with an accepted branch, so a branch's trunk is accepted before its sub-branches.
- **What is kept.** Only side branches of the main path are kept (depth 1), and only if they are at least 5 mm long and their median diameter is at least 0.3 × the parent's diameter at the take-off. Everything else is listed as "not modelled".
- **Measuring a side branch.** QCA starts where the branch's centreline leaves the parent's lumen (the parent's half-width at the junction plus 1 px). Only the far end is trimmed.
- **Take-off.** It is the parent's measured sample nearest to the skeleton junction.
- **Physics.** FFR comes from the tree solver (no surrogate until P3).
- **Virtual stents.** They are placed in path coordinates (distance from the inlet along a root-to-outlet path). Each branch piece of the path is widened only up to its take-off.

**Reason:**
- **Ostial lesions.** Starting a side branch at the parent's wall and not trimming its start keeps ostial lesions. The usual one-diameter trim would remove them, and inside the parent's lumen the measured width is the parent's.
- **Depth and size limits.** One level and the size limits keep noisy fragments out of the flow split.
- **Accepted error.** A skeleton junction sits up to about one vessel width downstream of where the centrelines cross (test: 8 px for a 12 px vessel). That shifts the take-off by about 2 mm, which matters little next to the 2D-projection errors.
- **Finding on ARCADE.** The segmentation model was trained on SYNTAX-segment masks and rarely segments small side branches. Lowering the threshold did not add junctions on the first 10 val images. Most image trees therefore have few side branches. This is reported in `reports/qca_tree_gallery.md`; better segmentation or synthetic angiograms come in P8.

**Alternatives:**
- Junction detection on the skeleton graph alone: it cannot tell the main path from a side branch.
- Using every endpoint: too many unmeasurable fragments.
- A separate low-threshold mask for topology: no effect on these images.

### D36 — Tree surrogate: branch-wise U-Net, message passing, monotone tree head (2026-10-30)
**Decision:**
- **Input.** Trees are padded to 5 branches of 256 samples. Every branch goes through the M3 FiLM 1D U-Net (shared weights), with point channels relative to the branch's own inlet reference. Branch features are length, inlet diameter, ln R_out, take-off position, a root flag, Pa and Pv.
- **Message passing.** At the bottleneck, 3 rounds of message passing update each branch's summary from its children (sum), its parent and the whole tree (masked mean). This is plain PyTorch with parent and take-off one-hot matrices, so it exports to ONNX with fixed shapes.
- **Head.** The decoder gives non-negative pressure-drop increments per branch (softplus, as M3). FFR is assembled down the tree: a branch starts at its parent's FFR at the take-off.
- **Training.** Loss is MSE over every sample of real branches + 2 × MSE at every outlet. Early stopping uses the validation outlet MAE (from epoch 40, D31).
- **Outlet resistances.** They are an input, from the solver's cheap calibration, as M3 takes R_micro.

**Reason:**
- **Physics by construction.** The head generalises M3's guarantee: FFR = 1 at the inlet, continuity at every junction and monotone decrease along every path all hold by construction (tested on an untrained model).
- **Why message passing.** A branch's flow depends on every other branch (its lesions and outlets), which a branch-wise network alone cannot see. Plain PyTorch keeps the ONNX export simple and avoids a graph library.
- **Padding.** It fits the synthetic trees (≤ 4 side branches). Image trees with more branches fall back to the solver.

**Alternatives:**
- A graph network over segment nodes (one node per segment between take-offs): finer, but variable segment lengths complicate the 1D convolutions.
- A graph library (PyG / DGL): extra dependency, harder ONNX export.
- Predicting outlet flows and integrating physics: closer to the solver, but loses the learned-pressure pipeline that the CFD labels need later.

### D37 — Tree planner: one path at a time, tree solver by default, 1 s budget (2026-10-30)
**Decision:**
- **Planning on a path.** The tree planner works on one root-to-outlet path. Lesions, candidates and the M6 sizing and landing-zone rules apply on a 256-point grid of path coordinates. A candidate widens only the path's branch pieces.
- **Scoring.** Every outlet's FFR is computed with the pre-PCI outlet resistances. Plans are ranked by the chosen outlet's FFR, as in M6, and all outlets are reported.
- **Default backend.** The tree solver evaluates candidates by default, through `solve_tree_variants` (one shared topology, batched Newton). The tree surrogate is optional, with a solver re-check of the shown plans.
- **Speed target** (the plan left it "to be set from P2 measurements"): **5,000 candidates in < 1 s** with the default backend, the M6 budget.

**Reason:**
- **Measurements.** P2 measured 3,474 generic trees/s. Variants of one tree are faster: 49 ms for 5,000 (measured while training ran).
- **Honest speed claims.** The tree surrogate (a U-Net per branch) is slower than the tree solver on a CPU. Claiming a surrogate speed-up would be false (CLAUDE.md), so the solver screens and the surrogate's time is reported without a target.
- **Why one path.** Planning one path at a time matches how stents are placed: along the vessel being treated, possibly across a take-off into a side branch.

**Alternatives:**
- Joint search over several paths: combinatorial and rarely needed.
- Surrogate screening as in M6: slower here, and the M6 reason (validating the pipeline) is covered by the surrogate backend and its benchmark.

### D38 — Validation tool: two files, locked predictions, local only, intention to diagnose (2026-10-30)
**Decision:**
- **Two files.** The analysts get `cases.csv` (images, points, calibration, Pa; no wire values).
  The data manager keeps `reference.csv` (FFR or iFR, exclusions). `vstent validate --manifest`
  writes `predictions.csv` and locks it with its SHA-256 (`predictions_meta.json`, together with
  the commit, model and manifest hashes). Scoring (`--predictions --reference`) refuses changed
  predictions.
- **Local only.** Case lists, images, masks, reference files and outputs must be outside the
  repository or git-ignored; the tool checks with `git check-ignore` and refuses otherwise.
- **Index tests.** Every vessel gets three index tests:
  - single path with the solver;
  - single path with the surrogate (with the P1 intervals and grey-zone flag);
  - the tree model with the solver.

  The primary one is `tree_solver`, set in `configs/validation.yaml` and fixed before
  unblinding.
- **Failures are kept.** A vessel the pipeline cannot analyse keeps its reason. Feasibility and
  intention-to-diagnose accuracy (failures counted as misclassified) are reported next to the
  accuracy on analysable vessels.
- **Statistics:**
  - Wilson intervals for proportions;
  - DeLong AUC standard error on the logit scale;
  - Bland–Altman limits with the sqrt(3 s² / n) standard error and the proportional-bias slope;
  - every interval also from a patient-level cluster bootstrap (2,000 resamples).
- **iFR references** are analysed as classification only (FFR and iFR are different
  quantities).
- **Optional vessel mask per case.** It stands for an analyst-corrected contour, as offered by
  commercial angiography-FFR workflows.

**Reason:** blinding, locking and complete accounting of failures are what make a diagnostic-
accuracy result credible (STARD 2015); a tool that enforces them avoids mistakes that
cannot be undone after unblinding. The local-only check enforces the project's no-patient-data
rule mechanically.

**Alternatives:**
- One file with predictions and wire values: simpler, but it breaks blinding.
- GEE or Obuchowski's clustered AUC instead of the bootstrap: valid, but more assumptions. A
  statistician can add them, and the protocol leaves room.
- Dropping failed vessels: the common shortcut, and it inflates accuracy.

### D39 — Synthetic stand-in study: rendered trees, true masks for the main run (2026-10-30)
**Decision:**
- **Cases.** The P4 dry run uses synthetic patients with one to three vessels. Each vessel is a P2
  tree laid out in 2D, with the main vessel continuing before the start and beyond the sensor.
- **Rendering.** Each vessel is rendered as an X-ray projection: transmission exp(−μ · chord),
  then blur, unsharp-mask edge enhancement and noise.
- **Reference.** The stand-in "wire FFR" is the tree solver on the true geometry, with sampled
  Pa, Pv and hyperemic velocity. The case list gives only Pa, the clicked points (1.5 px error)
  and, for 80% of patients, a pixel size with a 3% error.
- **Rendering fit to ARCADE.** The contrast and processing were fitted to ARCADE: on `syntax/val`,
  centreline transmission is 0.73–0.84 of the local background and nearly flat with width,
  and high-frequency noise is about 0.017. Attenuation × pixel size ≈ 0.010–0.020 per px with
  edge enhancement (amount 2–3, σ 1.5–3 px) fits with RMSE 0.023; without edge enhancement
  the best RMSE is 0.053. The pixel size (0.22–0.30 mm/px) and inlet diameters (2.5–4.0 mm)
  give the widest vessel 9–18 px, as in ARCADE (13–20 px).
- **The main run uses the true lumen masks; a second run uses the segmenter.**

**Reason:**
- **Contrast is not what defeats the segmenter.** With contrast and noise matched, the
  ARCADE-trained segmenter still misses most stand-in vessels. One rendered at high contrast,
  wide and isolated, was not segmented at all. The model relies on the anatomical context of
  clinical angiograms (dense trees, catheter, ribs), which the renders lack.
- **Separating two questions.** Running the main analysis on true masks measures the rest of the
  pipeline, while the second run shows the domain gap. Closing that gap is P8's task, not
  something to tune away here.

**Alternatives:**
- Tuning the renders until the segmenter works: it hides the gap and makes the stand-in look
  better than it is.
- Pasting tubes into ARCADE backgrounds: closer to clinical images, but it is P8's rendering
  work, and ARCADE's own vessels would confuse the paths.

### D40 — Points off the segmented vessel are refused (2026-10-30)
**Decision:** a start or end point farther from its nearest skeleton pixel than the vessel's
half-width there + 5 px raises an error in the single-path and tree QCA. Previously it was
silently snapped. In tree QCA the end point is snapped on the whole skeleton; if it lands on
another vessel, "no vessel path connects" is raised. Previously it was moved onto the start's
vessel.

**Reason:** the stand-in study found clicks 60–90 px beyond the end of a partly segmented vessel
being snapped to its visible end. That measured a path of half the length with no warning,
and a wrong FFR. The error is visible to the user (the app shows it; `vstent validate` records
the vessel as not analysable).

**Alternatives:**
- A warning only: the result would still be shown, and in a validation it would count as a
  success for the wrong vessel.
- A fixed pixel limit: it does not scale with vessel width.
