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

### D41 — Validation on public data, part by part (2026-10-31)
**Decision:**
- **Why public data.** The human cannot obtain a hospital partner, so validation uses public
  data with independent ground truth, one component at a time (P4b):
  - stenosis grading on real angiograms from another hospital against expert grades
    (CADICA, CC BY 4.0);
  - the pressure-loss physics against an inter-laboratory bench experiment (FDA benchmark
    nozzle, public domain);
  - P5–P7 use public multi-view and ECG data where it exists.
- **The P4 tool stays as built.** It is the path for real FFR data when any becomes available.
- **ICADD.** Its 70 Indian patients with angiography and invasive FFR are the only paired data
  found, and access must be requested by the human.

**Reason:**
- **Search result.** No public dataset pairs angiograms with invasive FFR (searched
  2026-10-31). Component-level evidence is what public data can give.
- **Rigour.** Each validation uses ground truth that the pipeline never saw. Each has a
  pre-defined comparison and reports what it measures, with every lab, lesion or failure
  shown.
- **Not a clinical validation.** Regulators (CDSCO, CE, FDA) expect clinical performance data
  against the reference standard for a device like this. Public data supports the technical
  file but does not replace that.

**Alternatives:**
- **Synthetic data only:** it cannot reveal errors shared by the generator and the model. The
  nozzle and CADICA did reveal such errors.
- **Literature-level comparison with published QFR results:** not reproducible from our code,
  and not the same patients.

### D42 — Physics check: FDA nozzle, net loss, nothing fitted (2026-10-31)
**Decision:**
- **Comparison.** The M1 loss model (Poiseuille + Kt expansion, unchanged constants) on the
  exact nozzle geometry (Stewart et al. 2012: 12 mm pipe, 20° cone, 4 × 40 mm throat, step)
  with the experiment's fluid and flow rates, against each laboratory's measured net loss.
- **Net loss.** Median of the three most upstream wall taps minus median of the three most
  downstream taps.
- **Scope.** Both orientations and all five Reynolds numbers, with every usable data set; files
  without wall pressures and the curator-deleted set are listed.

**Reason:**
- **Like for like.** The model has no reversible (Bernoulli) dip, so only the net loss
  between straight pipes compares like with like.
- **Robust to bad taps.** Medians of three taps absorb single bad taps (one laboratory's first
  tap reads a third of its neighbours).
- **No tuning.** Kt and the viscous term are not tuned to the data (CLAUDE.md). A
  developing-flow correction was considered, but its published form could not be verified
  and was not used.

**Alternatives:**
- Comparing whole pressure profiles: not meaningful without the reversible term.
- Fitting Kt to the nozzle: tuning, and it would hide the finding that the loss depends on the
  exit shape, which the model cannot represent.

### D43 — Edge method for stenosis grading: rule fixed before the results (2026-10-31)
**Decision:** for the CADICA grading (P4b), the intensity-edge QCA replaces the
probability-map edges as the grading method only if, on the **development** patients, both its
weighted kappa and its AUC for ≥ 70% lesions are higher. The test patients are not looked at
before this choice, and the test results of both methods are reported whatever they show.
Committed before any development result was seen.

**Reason:**
- **The suspected flaw.** On CADICA frames the segmenter appears to smooth over severe
  narrowings.
- **The synthetic check.** On rendered tubes, probability-map edges see 1–2% DS when the
  segmentation fills a stenosis, while intensity edges recover it to within about ±9 points,
  but with more noise when the segmentation is good.
- **No looking first.** Which effect dominates on real angiograms must be measured, not
  assumed, and choosing after seeing the test half would overfit the evaluation.

**Alternatives:**
- Choosing on all patients: no unbiased estimate is left.
- Always reporting both without a rule: the product needs one default.

**Outcome** (`reports/stenosis_grading.md`):
- **Development half: intensity edges selected** (both conditions met):
  - weighted kappa 0.36 vs 0.12;
  - AUC for ≥ 70% 0.81 vs 0.73.
- **Test half: confirmed, with a smaller margin** (patient-bootstrap intervals overlap):
  - kappa 0.22 (0.15–0.31) vs 0.12 (0.06–0.19);
  - AUC for ≥ 70% 0.73 (0.61–0.84) vs 0.69 (0.60–0.79);
  - within one grade 71% vs 54%.
- **Next:** make intensity edges the product's QCA default where the segmentation is
  automatic; analyst-corrected masks keep mask edges.

### D44 — Lumen method: imaging-chain model vs intensity edges, rule fixed before the results (2026-10-31)
**Decision:** the model-based lumen (`imaging/lumen_model.py`: blur and the X-ray system's
edge enhancement fitted to the straightened vessel) replaces intensity edges (D43) as the
product's lumen measurement for automatic segmentations only if, on the CADICA **development**
patients, both its weighted kappa and its AUC for ≥ 70% lesions are higher. The test patients
are then run once, and both methods' test results are reported whatever they show. Otherwise
the model stays an option and intensity edges stay the default. Written while the development
run was in progress, before any of its results were seen. During prototyping, six development
lesions were inspected visually; no test lesion was looked at.

**Reason:**
- **Edges have a resolution floor.** On rendered vessels of 8–16 px with ARCADE-like blur and
  edge enhancement, intensity edges never read more than ~35–75% DS, however tight the lesion
  (true 85–95%), the same compression seen on CADICA. The model fit read 50–95% lesions
  within ~2 points and recovered the blur and edge-enhancement settings.
- **But the renderer and the model share their physics**, so the synthetic result is a best
  case; real angiograms (scatter, background texture, processing the model lacks) decide.
- **The test half was used once before** (P4b, D43). This is a second look at the same 21
  patients, for one pre-specified comparison; there is no fresh public test set.

**Alternatives:**
- Simple densitometry (integrated contrast across the vessel, calibrated on healthy
  segments): tried first; on rendered vessels it overestimated severity by 10–25 points
  because edge enhancement also acts along the vessel, so it was dropped.
- Choosing on all patients, or after looking at the test half: no unbiased estimate left.

**Outcome (2026-10-31, development patients, recorded before the test half was run):** the
model lost on both criteria, so **intensity edges stay the product's lumen measurement** and
the model stays an option (`--lumen model`, the app's selector). On the 256 development lesions
(251 measured by each method), weighted kappa was 0.25 for the model vs 0.36 for intensity
edges, and the AUC for ≥ 70% lesions 0.73 vs 0.81. The model did not undo the severity
compression on real angiograms: median %DS of lesions graded 70–90% was 45 (edges 51), and of
lesions graded 90–98% 58 (edges 55). On rendered vessels it read the same lesions within
~2 points, so the imaging-chain model is missing something real angiograms have (scatter,
background structure, overlapping vessels, the vendor's processing), and visual grades are
known to read higher than QCA. Every number here is reproduced by `vstent stenosis-grading`
(development table of `reports/stenosis_grading.md`). The test half is run once for the
report, with all three methods, and nothing is chosen on it.

### D45 — Intensity edges: no side branch from edge asymmetry; minimum-cost edge tracking (2026-10-31)
**Decision:** three corrections to the intensity edges (D43), found when the ARCADE gallery
was regenerated after the D44 run, before any CADICA result with them:
1. **Side branches only where the segmentation's outline shows one.** One intensity edge much
   nearer the centreline than the other was taken for a branch on the far side and the
   diameter set to 2 × the near edge. But the centreline is the mask's skeleton, not the
   image's lumen axis, an eccentric lesion narrows one side only, and a gradient inside the
   lumen (contrast streaming, a superimposed vessel) is a near edge too. The diameter is now
   left + right. Where the mask's outline does show a branch, the probability-map diameter is
   kept, as superimposed vessels put intensity steps inside the lumen there.
2. **Minimum-cost edge tracking** (the classical QCA edge criterion): each wall is one
   continuous contour along the centreline (dynamic programming, at most 1 px of wall
   movement per px of path), instead of the steepest rise per normal, which flickered between
   two candidate gradients.
3. **Bridged gap points never count as a branch** (the mask has no vessel there).

**Evidence (before CADICA):** on the gallery's 20 ARCADE paths, with the original intensity
edges, 8 paths read ≥ 85% DS and 13 a distal FFR ≤ 0.2 (probability edges: 0 and 2); every
dip below 40% of the mask's diameter came from the 2 × near-edge rule (25 of 25, 9 of 9, 3 of
3 points on three images). With the corrections, 0 paths read ≥ 85% DS and 5 an FFR ≤ 0.2; two
of these 5 are as low with probability edges (long paths into thin branches, no side-branch
outflow). Short dips (≤ 3 points below 50% of the reference) fell from 75 to 25
(probability edges: 11). New tests: an eccentric 40% lesion read 78% before (now within the
edges' usual bias), and a flickering competing gradient is ignored by the tracking.

**Pre-registered check (written before the development run):** these are corrections of a
geometric error, adopted whatever CADICA shows. Intensity edges stay the product's default
only if, on the CADICA development patients, their weighted kappa and their AUC for ≥ 70%
lesions are each at least those of probability edges (the D43 comparison); otherwise the
default returns to probability edges. The full report is then run once more, with all three
methods (the third look at the test patients; nothing is chosen on it).

**Alternatives:**
- A median filter on the final diameter, or a minimum lesion length: suppresses short real
  lesions as well, and hides the cause.
- Keeping the 2 × near-edge rule with a stricter ratio: still wrong for eccentric lumens.

**Outcome (2026-10-31, development patients, recorded before the test half was run):** the
corrected intensity edges stay the product's default: weighted kappa 0.19 vs 0.12 for
probability edges, AUC for ≥ 70% lesions 0.82 vs 0.73 (251 of 256 lesions measured by each).
**Against the uncorrected intensity edges, kappa fell from 0.36 to 0.19 while the AUC held
(0.81 → 0.82).** Median %DS by expert grade (0–20, 20–50, 50–70, 70–90, 90–98%): 19, 27, 34,
42, 42, vs 23, 31, 45, 51, 55 before. Part of the earlier agreement with the visual grades
came from the 2 × near-edge rule, which exaggerated exactly the one-sided narrowings lesions
have. The corrected geometry is the right input for the FFR model (left + right is the
projected lumen width), but it reads severity lower than visual estimates, so the severity
compression on real angiograms is larger than P4b's kappa suggested. Reproduced by the
development table of `reports/stenosis_grading.md` (run next).

### D46 — Frame-level imaging-chain calibration: kept as an option; lesion insertion dropped (2026-11-01)
**Decision:** the imaging chain's blur and edge enhancement can be fitted once per frame,
jointly over its longest branch-free vessel segments (`imaging/calibration.py`,
`lumen_model.calibrate_chain`, `fit_lumen(chain=...)`), but this is not used by default, and
physics-consistent lesion insertion into real angiograms (planned in P4d) is dropped for now.

**Reason:**
- **Rendered frames:** the joint fit recovers the chain (5 vessels: blur 1.01 for 1.0, edge
  amount 2.20 for 2.5, width 2.17 for 2.0), and lesions fitted with the chain fixed read
  within 3 points. With only three vessel sizes, amount and width still trade off (3.1 for
  2.5); the test uses four.
- **Real ARCADE val frames (13 frames, 2–6 segments each):** the frame-level settings are not
  physically consistent: blur 0.6–3.0 px, edge amount 0.2–5.0, width 0.5–5.0 px, often at a
  bound. Fixing them raises each vessel's residual only slightly (e.g. 1.39 → 1.44 noise
  units), so the data hardly constrains them. The model is mis-specified for real images
  (background structure, scatter, non-circular lumens), not just ambiguous per vessel; this
  also explains the model's CADICA result (D44).
- **Lesion insertion** would add the model's predicted change in intensity for a narrower
  lumen. With the chain this poorly determined, tight inserted lesions would extrapolate an
  unidentified model, so a benchmark built on them could not be trusted.

**Alternatives:**
- Calibrating on the catheter (a cylinder of known size, as classical QCA does for the pixel
  size): a cleaner target for the chain, and a pixel-size calibration at the same time. Not
  done yet; it needs catheter detection.
- Generative lesion editing (diffusion, optimal transport; published 2025–2026): looks
  realistic, but its lumen is not known exactly, so it cannot serve as measurement truth.

### D47 — Cine runs: analysis frame and opacified window, chosen on development runs (2026-11-01)
**Decision:** the analysis frame of a cine run is the peak of its opacification curve (the
segmented vessel area per frame, median-filtered over 3 frames), and the opacified window is
the run of frames around the peak above **70%** of it. Contrast transit along the analysed
vessel is reported only when its front speed and the lower end of the speed's 95% interval
are positive; it is shown, not used for FFR (runs are recorded at rest).

**Reason (CADICA development patients only, 205 runs; written before the test runs were
scored):**
- the peak frame is one of the clinicians' selected frames in 84% of runs and lies within
  their span in 90%, whatever the window fraction;
- window overlap with the selected frames (median IoU): 0.60, 0.62, 0.56, 0.33 for fractions
  0.6, 0.7, 0.8, 0.9; 0.7 is the best;
- contrast arrival precedes the first selected frame in 95% of runs;
- transit on 12 development runs: 2 gave negative speeds and some had intervals spanning 0
  (overlapping vessels, the catheter in the path), so a speed is shown only for a front that
  advances.

**Pre-registered test:** `vstent cine-frames` scores the test patients once with these
settings. Primary measure: the share of runs whose analysis frame is one of the selected
frames; secondary: within their span, window IoU, contrast arrival, transit feasibility.
Results are reported whatever they show.

**Alternatives:**
- An image-only opacification proxy (black top-hat of each frame): it barely moved over the
  runs (background structure dominates), so the segmentation is used.
- End-diastolic frame selection from the ECG: CADICA has no ECG; it stays in P7.

### D48 — Regulatory drafts as checked, generated documents; proposed class C (2026-11-01)
**Decision:** the regulatory groundwork (P8) lives in `docs/regulatory/`. The intended use,
safety class, development plan, architecture and risk management plan are written by hand.
Requirements, hazards and the SOUP list are kept as YAML sources:
- `vstent regulatory` renders REQUIREMENTS.md, HAZARD_ANALYSIS.md, TRACEABILITY.md, SOUP.md
  and a CycloneDX SBOM;
- it fails on any broken reference (code `path::name`, test, report, control);
- a test checks that the committed documents match their sources;
- the SOUP list is generated from `uv.lock` for the release targets (CPython 3.14 on Windows
  and Linux), and the build fails if the code imports a package that is not declared and
  listed (this found pandas and pyarrow, imported by the app but only installed through
  Streamlit; both are now declared).

Proposed IEC 62304 class: **C**. The physician cannot recognise a wrong FFR estimate by looking
at the angiogram, so external controls cannot be credited. Preliminary market classes:
- India: **C** (CDSCO software guidance CDSCO/MD/GD/MDSW/01/2026, Table 2: critical
  situation × drive clinical management);
- EU: **IIb** (Rule 11);
- US: **class II, 510(k)** (21 CFR 870.1415; angiography-derived FFR precedents with
  product codes QEK and QHA).
All are drafts for a qualified reviewer.

**Reason:**
- hand-written traceability goes stale as soon as a test or function is renamed; a
  generated, checked trace stays true and is cheap to keep;
- a reviewer or partner can re-run it;
- risks are scored conservatively: unmeasured probabilities are set high and lowered only by
  evidence cited from a generated report, so the analysis shows the real state (13 of 19
  residual risks unacceptable for clinical use) and the evidence that would change it.

**Alternatives:**
- A requirements-management tool or spreadsheet: not linked to the code, so it is not checked.
- Class B by crediting the physician or the pressure wire as an external control: possible
  only with a narrower intended use (always confirm with the wire); recorded in SAFETY_CLASS.md
  for the reviewer.

### D49 — Projection geometry and two-view 3D reconstruction (2026-11-01)
**Decision:**
- **Geometry convention.** DICOM PS3.3 C.8.7.5 defines the positioner angles as the image
  detector's position (primary in the transaxial plane, + LAO; secondary in the sagittal
  plane, + cranial) but gives no rotation order. vstent uses the detector direction
  u = (sin α cos β, −cos α cos β, sin β) in the patient system (primary, then secondary about
  the rotated horizontal axis). Image columns run towards the patient's left and rows towards
  the feet at AP.
- **Pixel size.** For DICOM input, ImagerPixelSpacing × SOD / SID (the isocentre), or ÷ the
  EstimatedRadiographicMagnificationFactor. `vstent analyse` uses it automatically, and the
  app offers it first. Detector spacing without magnification information is no longer
  offered as a calibration. Before P5 the loader passed the detector spacing on, which
  overstates sizes by SID / SOD.
- **Two views.** The two centrelines are paired by dynamic programming over the distance
  between their rays (a symmetric epipolar cost in mm), with a weak equal-progress term
  (0.5 mm per unit of fractional position). The pairs are triangulated as ray midpoints and
  fitted with a weighted least-squares cubic spline (one knot per 4 mm). Each pair is
  weighted by the sine of the angle between the vessel and the epipolar line, with a floor
  of 0.05. A shift of view 2 across its epipolar lines (motion between the runs) is
  estimated by a grid and a bounded search over ±10 mm, and is on by default.
- **Diameters and checks.** Diameters come from each view's QCA at the pixel size of the 3D
  point, combined as sqrt(d1 · d2). The mean ray gap (> 0.5 mm), the reprojection residual
  (> 2 px), view separation (< 30°) and depth fixed mostly by interpolation give warnings.
- **Synthetic test.** P2 trees laid on a heart-sized sphere and rendered in two views of one
  C-arm. Views are chosen as QCA practice asks (the analysed vessel overlaps nothing), with
  random views as a separate stress test. Method settings (knot spacing, correction on) were
  chosen on development seeds (0–7, 900000+), then fixed before the report seeds (5001+,
  105001+) were run. Targets: the median over cases of the per-case mean 3D distance
  ≤ 1 mm, and the median |length error| ≤ 5%.

**Reason:**
- The isocentre correction is free wherever DICOM is available, and it removes a 20%
  (median) length and diameter error from geometry alone.
- The DP pairing is simple, deterministic and fast (tens of ms), and its weak-conditioning
  handling and spline give sub-millimetre centrelines on synthetic pairs.
- Motion between two monoplane runs is common (breathing, table), and on synthetic cases it
  made the length 8.5% wrong (median) without correction; the vessel itself calibrates the
  detectable component.
- The ellipse-equivalent diameter is the usual two-view QCA assumption.

**Alternatives:**
- Point-wise epipolar line intersections without global ordering: fail where the vessel runs
  along the epipolar lines.
- A deformable 3D curve fitted to both images (snake / bundle adjustment): more accurate on
  real data with distortion, but much more code; possible later on real data.
- Calibration from a catheter of known size: single view only, user-dependent; possible
  addition.
- Rotational angiography or CT for 3D: other data, other workflow.

The angle convention and the real C-arm's distances must be confirmed on real two-view DICOM
(a phantom or de-identified runs): the renderer shares the projection model, so the
synthetic test cannot reveal a convention error.

### D50 — Scaled lumen as the default; two-view knots 8 mm and a reliability check (2026-11-01)
**Decision:**
- **Scaled lumen** (`lumen="scaled"`, now `AUTO_LUMEN`). Intensity edges give the lumen's
  shape, so %DS and the CADICA grading are unchanged. The absolute size is the median of
  fit / edge diameters, with the imaging chain fitted jointly on up to 4 healthy windows of
  30 points. A factor outside 1.0–1.4, or a failed fit, keeps the edges' size with a
  warning. Side branches take the main vessel's factor.
- **Chosen by the rule pre-registered in docs/PLAN.md** (commit 7437238, before any held-out
  or real-frame measurement). The first variant fitted the whole vessel; it met the rule but
  took 6–16 s per vessel on ARCADE:
  - held-out renders (seeds 920000+): median −0.7%, 100% within ±15%;
  - ARCADE frames 1–20: factor accepted on 85%, median 1.133 (development range 1.100–1.198).

  The windowed fit (0.6–1.3 s) replaced it and was judged again on fresh data:
  - development renders (910000+): median +0.1%, 96% within ±5%; central 80% of the factor
    1.120–1.209;
  - held-out renders (930000+): median 0.0%, 97% within ±15%; 5 of 79 views without a
    healthy window kept the edges' size;
  - ARCADE frames 21–40: factor accepted on 18 of 20, median 1.148.

  Intensity edges alone read the same renders 13–14% narrow.
- **Two views:** the spline knot spacing is now 8 mm, chosen on 199 development cases. Cases
  with length error > 5% dropped from 17 to 8 (4 mm: median 0.74%; 8 mm: 0.78%; bias −0.6%;
  12 mm cuts corners). A **reliability check** refits the curve with twice the knot spacing
  and warns when the length changes by more than 2.5% or the median reprojection residual
  exceeds 0.30 px. On development data it flagged 7 of the 8 large errors, with 13.6% of all
  cases flagged. It was chosen for sensitivity over the 10% alarm target: a missed wrong
  length is a hazard, while an alarm only asks the user to check the views.

**Reason:**
- Absolute size sets the stent diameter and part of the viscous loss. With DICOM geometry
  there is no catheter to cancel the edge bias, as catheter calibration in classical QCA
  does.
- Keeping the edges' shape keeps D44's real-data choice for severity.
- The chain model explains the blur and edge enhancement that cause the bias, and fitting it
  only where the lumen is healthy keeps it fast and away from lesions.

**Alternatives:**
- The full model lumen as the default: slow on real vessels, and its severity was not chosen
  on CADICA (D44).
- An additive correction from the fitted blur: it changes %DS towards milder lesions, the
  direction of the known compression (H-01).
- Catheter calibration: needs a catheter in view and the user's input; a possible addition.
- Frame-level chain calibration: not physically consistent on real frames (D46).

**Risk:** the absolute size on real angiograms is unverified. If real edges were not biased,
the scaled lumen would oversize by about 15%. The factor range, the warning and hazard H-09
cover this until there is real ground truth (a phantom, or IVUS / OCT reference diameters).

### D51 — The two-view smoothing check is reported, not a warning (2026-11-01)
**Decision:** the founder chose to drop the P5b reliability warning and keep the number. The
length change under twice the knot spacing is printed with every two-view result and stored in
the case report; it no longer raises a warning.

**Reason:** on 200 fresh synthetic cases the warning fired on 21 cases, plus 14 from the older
warnings (18% in all), and missed the one large error. Large errors are now rare (1 in 200)
after the 8 mm knots. Frequent alarms without a measured benefit invite alarm fatigue (H-19).

**Alternatives:** keep it as a warning; redesign it on a much larger synthetic set.

### D52 — Two views in the app (2026-11-01)
**Decision:** the app takes a second view in two ways: a toggle with a second uploader, shown
only when the first image has DICOM projection geometry, or the built-in synthetic two-view
example (the first case of the two-view report). The example uses its rendered vessel masks
instead of the segmentation, because the ARCADE-trained segmenter misses many synthetic
vessels (the domain gap in reports/validation_synthetic.md). The second view's points are
clicked on its image, with the epipolar lines of the view-1 points drawn as guides. Stents
are drawn in both views by projecting the 3D centreline back. Tree mode stays single-view.

**Reason:** the analysis matches the command line, and the example makes the feature demonstrable and
testable without patient data.

**Alternatives:** finding the view-2 points automatically on the epipolar lines (ambiguous
where the line crosses several vessels; possible later); tree mode in two views (needs 3D
side branches).

### D53 — Segmenter trained with rendered angiograms adopted; angiogram → FFR deferred (2026-11-01)
**Decision:** the product's segmenter is now `seg_unet_r34_synth`. It uses M5's configuration
(seed, schedule, early stopping on ARCADE val), trained on ARCADE's 1,000 syntax images plus
1,000 rendered angiograms (`vstent synth-angio`). It was adopted by the rule pre-registered in
docs/PLAN.md (commit 9537e4e) before any rendered data existed: a higher mean ARCADE test Dice
(0.7947 vs 0.7925) and a lower bound of the paired 95% interval above −0.01 (−0.0052).

**What the result means** (`reports/sim_to_real.md`):
- On real images the two models are statistically indistinguishable. The paired difference is
  +0.0022 on ARCADE test (95% interval −0.0052 to +0.0095) and −0.0047 on ARCADE val (−0.0105
  to +0.0012). The rule tests non-inferiority with a higher mean; it does not show that rendered
  data helps on real angiograms.
- On the rendered test split the new model scores 0.907 against 0.600 (10th percentile 0.828
  vs 0.303). The split comes from the generator the new model trained on, so this shows that
  the domain gap D52 met is closed for rendered images, not a real-world gain.
- ARCADE annotates SYNTAX segments, not every major vessel in view, so ARCADE Dice cannot
  reward a model that finds more of the major vessels.

**Reason:** a pre-registered rule is only worth something if it is followed. The new model is
no worse on real images and much better on rendered ones, which the validation stand-in and
the synthetic examples use.

**Deferred:** angiogram → FFR models trained on rendered images, the other half of the P8
backlog item. The product keeps the explainable chain (segmentation → QCA → physics). A
learned image → FFR model can only add something with labels better than the reduced-order
model's own output: CFD (P6) or wire FFR (P4).

**Alternatives:**
- Keep M5: the rule said adopt.
- More rendered images, another mix, or rendered pre-training then ARCADE fine-tuning: not
  pre-registered. ARCADE test has now been used for this comparison, so judging another
  variant would need fresh real test data.

**Downstream:** reports produced with M5 record the model they used and stay valid as
records: QCA gallery, stenosis grading, cine frames, validation stand-in, app latency. The
CADICA numbers in the hazard analysis (H-02, H-04) were measured with M5. Regenerating them
with the adopted model is a separate step.

### D54 — The adopted segmenter is kept despite a worse automatic frame choice (2026-11-02)
**Finding:** after D53 the reports that use the segmenter were regenerated. Stenosis grading
on CADICA is unchanged within its intervals: intensity edges, test patients, weighted kappa
0.13 vs 0.14 with M5, AUC for ≥ 70% 0.71 vs 0.72. The tree gallery finds side branches in
16 of 20 images (14 with M5). The validation stand-in's automatic arm analyses 96 of 102
vessels (42 with M5), but those renders are close to the new training data. **The P4d
analysis frame got worse:** it lies within the span of the clinicians' selected frames in
82% of CADICA test runs (88% with M5) and 81% of development runs (90%). Paired on the same
runs, M5 was inside the span where the new model was not in 25 development runs; the
reverse happened in 7 (exact McNemar p = 0.002). The test figures were 17 against 7
(p = 0.06). The opacification curve is the segmented area per frame, and its peak moves
with the segmenter: the analysis frame changed in 81% of runs (median 3 frames).

**Decision:** keep the D53 segmenter and recover the frame choice in P7, whose ED-frame rule
replaces P4d's peak rule. It is chosen on the CADICA development runs with this segmenter,
and its pre-registered test target is set against M5's rate (≥ 83%, i.e. 88% minus 5
points). The user can still move to any frame.

**Reason:** the adoption rule was met; reverting after one unfavourable downstream result,
without a rule for doing so, would be a post-hoc choice. The frame choice is being replaced
anyway, and the H-11 controls (frame shown, user confirmation) are unchanged.

**Alternatives:**
- Revert to M5: loses the rendered-image robustness; not supported by any rule.
- Keep M5 for the opacification curve only: two segmentations per frame, which doubles the
  cine run's CPU time.

### D55 — P7 methods: ED frames, motion, phasic flow time, intervals, run curves (2026-11-02)
Every choice below was made on development data only, before the held-out synthetic seeds
(9200+) and the CADICA test patients were used for P7:
- synthetic seeds 9100–9123 for the methods;
- seeds 9124–9163, then 9164–9199, for the intervals;
- the CADICA development patients for the run curves.

**Decisions:**
- **ED frames:** from the R-wave pointer, else the ECG's R peaks, else the vessel tree's
  spread in the images. The analysis frame is the ED frame in the opacified window with the
  most vessel; with none there, P4d's peak frame.
- **Heart motion:** the vessel-probability maps are registered (affine, ECC) to the analysis
  frame on the opacification plateau only: the frames from the first to the last at ≥ 90% of
  the peak area. Filling and washout frames take the transform of the same heart phase a
  whole number of beats away, or the nearest plateau frame.
  - Registering every frame slid partly filled vessels along themselves: a motionless test
    vessel read 26% fast.
  - On the development cases it gave a velocity bias of +3.8% (mean), against −1.6% for
    plateau tracking.
- **Phasic flow:** arrival times are mapped to flow time, with systole 0.31 s after each R
  wave and a diastolic / systolic ratio of 2.0 (a prior). On the development cases (3D path,
  image-based ED) the median error fell from 19.9% (plain time) to 9.7%. Fitting the ratio
  per run gave 16.1%.
- **Intervals:** split-conformal 90% quantiles of |ln(true / measured)|. The two-run mean is
  used only when the runs agree within ×1.5 (`reports/cine_calibration.md`). One view: upper
  end × 1.37 (P5 foreshortening, 95th percentile).
  - The Theil–Sen interval covered the truth in only 10–29% of runs: phasic flow, not
    scatter, dominates.
  - A first calibration on seeds 9124–9163 used |true − measured| / measured. It gave 0.42
    per run and 0.47 for the two-run mean. That score penalises underestimates more, and its
    tail came from cases whose two runs disagreed fivefold.
  - The score and the agreement guard were therefore chosen on the method seeds (consistent
    pairs agreed within ×1.24; the three outliers were ×1.48–2.22, each with one run off by
    more than 25%). The intervals were then recalibrated on fresh seeds 9164–9199: × / ÷
    1.68 for one run (0.518, 61 runs) and × / ÷ 1.23 for the mean of two agreeing runs
    (0.204, 23 of 36 cases). A single run's velocity is therefore barely more informative than
    the fixed velocity (0.25–0.45 m/s); two agreeing runs narrow it.
- **Run curves at probability ≥ 0.9** (QCA keeps 0.5). On the CADICA development runs the ED
  rule put the analysis frame inside the clinicians' span in 171 of 205 runs (163 at 0.5),
  P4d's peak rule in 175 (166). On the synthetic development cases, 0.9 left the velocity
  unchanged (two-view median error 6.8% vs 6.3%).

**Not chosen:**
- The M5 segmenter for the run curves. On development runs it put the frame inside the span
  in 183 of 205, but it means a second segmentation of every frame, a second model in the
  product, and M5 misses many rendered vessels (D52). It is kept as an option for the
  founder.
- On CADICA the ED rule does not beat the peak rule (171 vs 175 at 0.9). The clinicians'
  frames are lesion keyframes, not ED frames. ED is kept for physiology: the QCA convention,
  and two views at the same phase.

**Risks:**
- The flow-time correction assumes the flow-profile family that the generator draws from;
  its real benefit needs a phantom or a Doppler wire.
- The FAVOR Pilot conversion and frame-count velocity are the subject of QFR patents
  (freedom to operate, INVENTION_NOTES §7).
- The fixed velocity stays the default.

### D56 — Real coronary physiology checks the model's assumptions; nothing changed by rule (2026-11-02)
**Finding** (`reports/physiology_real.md`; DEFINE-FLOW, 455 lesions of 379 patients,
core-lab-accepted pre-PCI measurements; rules pre-registered in docs/PLAN.md, P4e):
- **Phasic flow exists in real patients.** The resting diastolic / systolic ratio of the
  Doppler velocity has a median of 1.73 (IQR 1.50–2.01) against P7's prior of 2.0, and a
  systole of 0.39 s against 0.31 s. It differs by artery: LAD 1.77, LCx 1.81, RCA 1.35. The
  rule (replace the prior if the median lies outside 1.5–2.5) did not fire.
- **Hyperemic velocity.** The stenosis-free hyperemic mean velocity is 0.21 m/s (median, k 0.5)
  or 0.33 m/s (k 0.8), so the fixed 0.35 m/s is plausible only with the higher Doppler
  mean-to-peak factor. The rule (revisit if 0.35 m/s is outside the IQR at both k) did not
  fire. P1's 0.25–0.45 m/s range contains 27% (k 0.5) to 56% (k 0.8) of lesions.
- **FAVOR Pilot conversion:** median relative error +28% (k 0.5) or +4% (k 0.8), 5th–95th
  percentile −29% to +72% (k 0.8). Individual conversions are imprecise, which supports keeping
  the fixed velocity as the default (D55).
- **Pa at hyperemia:** median 92 mmHg; 78% inside P1's 70–110 mmHg.

**Decision:** nothing changes, because no pre-registered rule fired. Two candidate changes are
recorded, each to be pre-registered before it is tested:
- an artery-specific phasic prior (RCA about 1.35, left coronary about 1.8), evaluated on
  P7's development seeds and then on its held-out report;
- wider velocity and Pa ranges in P1's input Monte Carlo, so that they cover real patients.

**Reason:** a check that is pre-registered and then followed is what makes it evidence; a change
seen in the data and applied at once would not be.

**Limits:** Doppler measures a peak velocity at the wire tip, distal to the lesion, and the wire
alters the flow; the mean-to-peak factor is uncertain (0.5–0.8). The population is stable
patients with intermediate lesions. No angiograms are included, so FFR from images is not
measured.

### D57 — Automatic single-frame measurements are not repeatable enough yet (2026-11-02)
**Finding** (`reports/repeatability.md`; CADICA, 42 patients, 414 lesions with ≥ 2 keyframes,
targets pre-registered in docs/PLAN.md, P4f, from accepted methods' own variability):
- **Frame to frame, not met.** The same lesion on different keyframes of its run gives a
  within-lesion SD of 10 %DS points (CV 44.6%; target ≤ 11.9%) and a segment-FFR SD of 0.124
  (CV 18.7%; target ≤ 9.4%, QFR's inter-observer CV in QREP). It is worst in severe lesions
  (FFR CV 27–28% at 70–98%). Pairs of keyframes agree on the side of 0.80 in 85%.
- **Operator to operator.** Two independent choices of the segment ends give an FFR CV of
  17.1% (not met; 90% agree at 0.80), but an SD of %DS differences of 6.0 points (met;
  QCA's inter-observer range is about 5–7).
- **Part of this is the design:**
  - the keyframes span heart phases, where the product's cine workflow uses end-diastolic
    frames only;
  - the FFR is a short segment's, at an assumed pixel size;
  - nothing is corrected by hand, whereas QFR's analysts edit contours.
- **Part is a real weakness:** one automatic frame does not measure a lesion stably. This
  agrees with the weak agreement with the experts' grades (D44) and is now a hazard input
  (H-01, H-02).

**Decision:** nothing changes by rule; the report stands as measured. The next step,
pre-registered before it is tried, is **multi-frame consensus**: measure the lesion on several
frames (the run's end-diastolic frames, or the keyframes) and report the median. If frame
errors were independent, three frames would cut the spread by about √3. The number of frames is
to be chosen on the CADICA development patients and the result reported on the test patients.
Also: contour editing in the app (as QFR analysts do), and a lesion-measurement model trained
on open stenosis annotations (AngioCAD, CARDIAG).

**Alternatives:** retune the pass marks (rejected: they were set from accepted methods before
measuring); claim the lower operator variability only (rejected: the frame-to-frame result is
the one a user meets).

### D58 — Multi-frame consensus: K = 5; FFR repeatability near QFR's, %DS not (2026-11-02)
**Finding** (`reports/consensus.md`; rules pre-registered in docs/PLAN.md, P4g, commit 4ecbd0b,
before any consensus was computed). Data: P4f's CADICA measurements, re-made with the rows
saved; the single-frame result reproduced exactly (%DS CV 44.6%, FFR CV 18.7%).
- **Development patients:** no K in 1–5 met both targets. The rule took the lowest FFR CV:
  **K = 5** (FFR CV 12.0%, 95% 4.7–18.5%; %DS CV 21.7%). One frame: FFR CV 23.1%.
- **Test patients, once, K = 5** (111 lesions, 20 patients):
  - FFR CV **5.1%** (95% 3.4–6.8%): **met** (≤ 9.4%, QFR's inter-observer CV in QREP);
    within-lesion FFR SD 0.040, against 0.124 for one frame;
  - %DS CV **27.0%** (95% 20.1–32.5%): **not met** (≤ 11.9%);
  - the two medians agree on the side of 0.80 in 91%.
- **How to read it:**
  - Development and test differ (12.0% vs 5.1%). The development interval is wide and holds
    both, because a root-mean-square CV is driven by a few lesions with a large spread. The
    5-frame median brings segment-FFR repeatability to about QFR's inter-observer level; that
    it is below that level is not established.
  - A CV depends on the mean: FFR sits near 1 and %DS near 40%, so equal noise gives FFR a
    smaller CV. The SD (0.040) is the more direct number.
  - CADICA's keyframes span heart phases. The product's end-diastolic frames should differ
    less, but CADICA has no ECG, so this is untested on real ED-only frames.
- **Flag** (frames on both sides of 0.80): all 10 test lesions whose medians disagreed were
  flagged, but 31.5% of consensus sets were flagged and 38% of concordant lesions too. It is
  sensitive, not specific.

**Decision:**
- **Product K = 5** (`inference/consensus.py`, `CONSENSUS_K`). A cine run is measured on up
  to five end-diastolic frames, with the user's points following the heart. The report gives
  the median distal FFR with every frame's value, and the flag. The frame closest to the
  median gives the profile for the intervals, the plans and the figures. One function serves
  `vstent analyse` (single path and tree), the app and `vstent validate`.
- **Validation (before any real data):**
  - the index tests of a cine run are the medians, as the product reports them;
  - the analysed frame alone is kept in the `frame_*` columns as an exploratory comparison
    (PROTOCOL updated);
  - an analyst's mask covers one frame, so such vessels are measured once.
- **Two rules beyond the pre-registered text, set on principle before anything was measured
  with them:**
  1. Only frames on which the contrast has filled the analysed path: 95% of its points at 90%
     of their plateau contrast (the P7 transit), up to the end of the opacification plateau.
     A frame before filling cannot show the lumen. On the first synthetic run it changed
     nothing.
  2. A frame whose traced path differs by more than 30% in length from the reference frame's
     is left out, with the reason: the moved points probably found another vessel.
- **On a synthetic beating-heart run** (breathing 3–4 mm):
  - the moved points were within 1 px of the true vessel position, also from a systolic
    reference frame;
  - one end-diastolic frame read the lesion's MLD as 0.41 mm against 1.16 mm on the others
    (FFR 0.35 vs 0.92); the median kept 0.92 and the flag fired. This is the single-frame
    failure P4f measured.

**Alternatives:**
- K = 3 (the value assumed while building): replaced by the rule's choice.
- The mean: not robust to such outlier frames.
- A flag on the frames' spread instead of the 0.80 straddle: it would need its own
  pre-registration.

**Not solved:** %DS repeatability; the severity compression every frame shares (D44); accuracy
against wire FFR.

### D59 — A guided app as the default; the research view behind --advanced (2026-11-02)
**Decision:** `vstent app` opens a guided, single-page workflow for one cine run
(`app/guided.py` for the logic, `app/guided_app.py` for the screen); the previous app stays as
`vstent app --advanced`. The steps follow a cath-lab reading:
1. load the run;
2. pick the frame;
3. check the vessel path and its lesions;
4. read the virtual FFR;
5. plan a stent;
6. download the report.

Choices:
- **"Virtual FFR", not "QFR"**: QFR is another company's trademark.
- **The headline is the multi-frame median** (D58), with the heartbeats' agreement in words.
  The stent step uses the frame closest to the median, shown with the path moved there.
- **The plain-language reading** uses the grey-zone flag. *Borderline* when the input
  interval contains 0.80; otherwise *significant* (≤ 0.80) or *not significant*.
- **Frame choices** are the end-diastolic frames on the opacification plateau: the same frames
  the median can use.
- **Automatic path: the main trunk** (`centerline.main_trunk_endpoints`). It starts at the
  wider end of the longest path, takes the widest branch over the next 15 px at every fork,
  and stops at 100 mm.
  - On a real CADICA run (patient 24, run 15) the longest path alone was about 270 mm and
    crossed vessels: 9 "lesions" and FFR 0.49. The trunk path was 100 mm, with 2 lesions of
    50% or more and FFR 0.74.
  - This was found by trying the demo, not chosen on any evaluation.
- **Only narrowings of ≥ 50% are numbered**; milder ones are counted.
- **Manual stent lengths are limited to those that fit the path.** The flow test found a
  crash on a 7 mm path.

**Alternatives:**
- Simplifying the research app in place: rejected, because its options are needed for
  research and the reports.
- Tabs or several pages: rejected; one page in order keeps the dependence between steps
  visible.

**Not solved:**
- The automatic path can still pick the wrong vessel; the user is asked to check it.
- Without an ECG, the motion tracking between heartbeats often fails on real runs (CADICA:
  the moved points landed 86 px away). The result then rests on one heartbeat and says so.

### D60 — Guided app, second version: case library, frame scroller, lesion close-ups, FFR guide (2026-11-02)
**Decision** (after the user tried D59's version):
- **Case library** (`guided.case_library`): 10 synthetic beating hearts (seeds 9500–9509, with
  ECG and C-arm geometry), then CADICA's 382 selected runs if downloaded. They are ordered
  90–98%, 99%, 70–90%, 50–70%, 20–50%, 0–20%, then occlusions (no valid FFR) and runs
  without a marked lesion. *Random* and *New study* reset the study; uploading stays
  possible.
- **Frame scroller:** a slider over the end-diastolic frames (★ recommended, ◆ expert box,
  "faint" outside the opacification plateau; every frame on request). It runs in a fragment:
  scrolling measures nothing; *Analyse this frame* commits.
- **Lesions by eye:**
  - a close-up of each lesion with the measured lumen edges (`guided.lesion_zoom`), and the
    plain angiogram on request;
  - on CADICA keyframes, the experts' boxes, and *Through the expert box*: a path through
    the most severe box (`guided.box_path`, the P4b path).
- **FFR guide and reading by fixed bands** (`guided.FFR_GUIDE`, `guided.verdict`), replacing
  D59's grey-zone-flag reading so that the label and the guide agree:
  - above 0.80: not significant, no stent needed;
  - 0.76–0.80: grey zone, significant;
  - 0.75 or below: significant;
  - after a stent, 0.90 or more is a good result.

  The bands come from the wire-FFR literature (FAME; ESC 2019). The likely range is shown
  beside the value. Above 0.80 the stent step says no stent is needed, and still allows one.
- **Less text:** short labels; explanations in tooltips.

**Seen with it:** on CADICA runs graded 90–98% (p11 run 16, p12 run 1), the path through the
expert box gives 32–35% DS and FFR above 0.9. The close-ups show why:
- the lesion is visible mainly as fading of the contrast column (lower density), not as a
  measurable narrowing of its width;
- in vessels of 7–16 px, the edges of a tight lesion lie inside the blur.

Measuring density (densitometry) is the candidate fix; D44 dropped a simple version because
it overestimated severity on rendered vessels. Not done here.

### D61 — Densitometric lumen: not adopted by the rule; agreement improves, severe lesions still read mild (2026-11-02)
**Finding.** Pre-registered in docs/PLAN.md (P4h, commit 031074a). Amendment 1 (the IOD
smoothing, commit 8aa3def) was chosen on development renders before any CADICA measurement.
Reports: `reports/stenosis_grading.md`, `reports/lumen_synthetic.md`; development rows in
`data/derived/p4h/` (`scripts/p4h_dev.py`, commit 5c38b64).
- **Domain:** `log` by the rule (development kappa 0.294 against 0.281 for `linear`).
- **Development patients** (21 patients, 248 lesions):
  - weighted kappa 0.214 → 0.294 (higher);
  - median %DS of 0–20% lesions 19.6 (≤ 30, met);
  - **AUC for ≥ 70% lesions 0.818 → 0.803 (lower)**.

  The adoption rule needs both kappa and AUC higher: **not met**.
- **Rendered lesions** (the report's seeds): tight lesions ≥ 85% mean |error| 5.9 points
  (edges 30.7; ≤ 10 met); no lesion 5.5 (edges 6.5; ≤ 8 met).
- **Test patients** (21 patients, 208 lesions; run once; nothing chosen on them):

  | | Edges | `dense` |
  |---|---:|---:|
  | Weighted kappa | 0.13 | 0.25 |
  | AUC for ≥ 70% lesions | 0.71 | 0.77 |
  | Median %DS, lesions graded 70–90% | 44 | 46 |
  | Median %DS, lesions graded 90–98% | 40 | 49 |
  | Median %DS, lesions graded 0–20% | 18 | 19 |
  | 0–50% lesions read as critical | 0 / 219 | 1 / 219 |

  The pre-registered targets were not met (kappa ≥ 0.40; 70–90% ≥ 50; 90–98% ≥ 60).
- **Reading:**
  - Densitometry improves agreement on both halves; the development AUC drop of 0.015 lies
    well inside the interval.
  - But it moves severe lesions by only 2–9 points. On rendered images it is accurate. On
    real angiograms a tight lesion's density deficit is far smaller than the physics
    predicts. Likely causes: the X-ray systems' processing before export (local contrast
    equalisation, edge enhancement, 8-bit images) evens out density, and visual grades read
    above QCA.
  - It is the second physics-based attempt at lesion severity that does not meet its rule
    (D44 was the first).

**Decision:** the product's lumen is unchanged. `dense` stays an option (`--lumen dense`, the
app's research view, `vstent stenosis-grading --edges dense`). Following CLAUDE.md, after two
attempts the analysis and options are written in PROGRESS and the human decides the next step.

**Alternatives:**
- Adopting it on the test result: rejected. The decision is made on development data; using
  the test to choose would leave no unbiased result.
- Relaxing the rule after seeing it: rejected.

### D62 — Hybrid vessel segmentation: not adopted; vessel coverage is not the main gap (2026-11-02)
**Finding** (`reports/vessel_coverage_dev.md`; pre-registered in docs/PLAN.md, P4j, commit
c6623be; CADICA development patients and ARCADE val):

| Segmentation | CADICA development boxes crossed by a measurable path | ARCADE val recall | Added pixels near annotated vessels |
|---|---:|---:|---:|
| Model alone | 95.4% | 80.8% | – |
| Hybrid, T = 0.05–0.20 | 98.3–99.3% | 81.9–85.9% | 15.5–24.1% |

- No threshold met the guard (≥ 50% of the added pixels near annotated vessels), so none was
  eligible and the hybrid is **not adopted**. It stays an option
  (`SegmentationRunner(hybrid_threshold=...)`).
- The added structure could be real small branches (ARCADE annotates only the SYNTAX
  segments) or noise from ribs and lung edges; the data cannot tell them apart.
- **The model already crosses 95% of the boxes.** The faint mid-vessel in the app's overlay is
  often marked only thinly, but enough for a path. Vessel coverage is not the main reason
  severe lesions read mild.

**Seen with it (the user's screenshot, CADICA p12):** a long diffuse lesion graded 90–98% on
three boxes, measured on the path "through the expert box" (box ± its size), reads 17–36%.
That path starts inside the diseased segment, so the healthy reference along it is itself
narrow. Experts compare with the healthy proximal vessel. The app's expert-box path now starts
at the tree's proximal end and runs through the box (as *Target* does). The grading protocol's
short path has the same weakness; a pre-registered comparison follows.

**Alternatives:**
- Relaxing the guard after the result: rejected.
- Adopting the hybrid for display only: possible later, as an option.

### D63 — Learned lesion reading: much better on severe lesions, too many false alarms; an option, not the default (2026-11-02)
**Finding** (`reports/lesion_reading.md`; pre-registered in docs/PLAN.md, P4i, commit 5bf07b2;
amendment 2: P4j did not adopt the hybrid segmentation, so the run on the model's segmentation
was evaluated):

| | Edges | Learned |
|---|---:|---:|
| **Development (out-of-fold)** | | |
| Weighted kappa | 0.21 | 0.39 |
| AUC for ≥ 70% lesions | 0.82 | 0.84 |
| Median %DS of lesions graded 70–90% / 90–98% | 39 / 46 | 51 / 63 |
| ARCADE val stenoses read ≥ 50% | 36% | 79% |
| ARCADE val distant candidates read ≥ 50% | 7% | 46% |
| **Test (once)** | | |
| CADICA test: kappa | 0.13 | 0.32 |
| CADICA test: AUC for ≥ 70% lesions | 0.71 | 0.82 |
| CADICA test: median %DS 70–90% / 90–98% / 0–20% | 44 / 40 / 18 | 53 / 65 / 11 |
| ARCADE test: stenoses read ≥ 50% | 29% | 73% |
| ARCADE test: distant candidates read ≥ 50% | 4% | 29% |
| Kemerovo (external): significant lesions read ≥ 50% | 18% | 54% |

- The development rule failed on one condition, the false alarms (46% against a limit of 17%):
  **not adopted**.
- The improvement on severe lesions is large and consistent across the CADICA test patients,
  ARCADE and a third hospital (Kemerovo).
- **Likely cause of the false alarms:** the %DS head learned only from expert boxes (CADICA
  lesions at every grade). It never had a %DS target at an ordinary vessel point, so it
  over-reads unmarked narrowings. The significance head, which saw ARCADE's distant candidates
  as negatives, is not used for %DS.

**Decision:**
- The product's lumen is unchanged.
- The reader is a research option: `analyse_vessel(..., lesion_reader=...)`, and the guided
  app's "Learned lesion reading (research)" switch.
- A second version needs its own pre-registration. Candidates:
  - a %DS target on non-lesion candidates;
  - the significance head as a gate;
  - the clinician-style path from the proximal vessel (D62).

  The test sets used here (CADICA test, ARCADE test, Kemerovo) have now been seen once. A
  fresh external set would make the next test credible.

**Alternatives:** adopting on the test result, or relaxing the false-alarm limit after seeing
it: both rejected.

### D64 — Guided app: stent plans from the solver; each hand-placed stent has its own controls (2026-11-02)
**Seen** (the user's screenshot, a CADICA run with two lesions):
- The plan list showed a post-PCI FFR of 0.52, and the big number 0.61 for nearly the same plan.
- "Adjust it myself" changed only the first of the plan's two stents.
- The report kept only the first stent.

**Cause:** the plans were ranked and labelled by the ONNX surrogate, while every other number on
the page comes from the reduced-order solver. On a 250 mm real path the surrogate is outside
its training range (synthetic vessels about 70 mm long), so the two disagree.

**Decision:**
- The guided app plans with the solver backend. It is also faster: 0.15 s vs 0.84 s for 5,000
  candidates (`reports/planner_benchmark.md`). The research view keeps the surrogate option.
- Hand placement offers 1–3 stents, each with its own diameter, length and start. Stents are
  labelled S1, S2… on the image and in the text, and overlaps are noted.
- When the FFR stays ≤ 0.80 after stenting and most of the remaining loss lies outside the
  marked lesions, the app says so. A stent cannot fix diffuse loss; a path that runs into thin
  distal branches adds such loss.
- The report includes every stent.

**Alternatives:** showing both FFRs in the plan list was rejected: two numbers for one plan
confuse the guided view.

### D65 — The automatic path starts at the tree's widest end, not at an end of the longest path (2026-11-02)
**Seen** (a CADICA left-coronary run, P4k-D's display): the path started at the tip of a
diagonal branch, ran back through the left main, then down the LAD. The reference diameter and
the flow direction over that first part were wrong.

**Cause:** the proximal end was the wider end of the longest skeleton path. With a short left
main, the longest path runs from a side branch's tip to the LAD's end, so the left main is not
on it.

**Decision:**
- `proximal_end` chooses, among all ends of the largest vessel tree, the one where the vessel
  is widest over its first 15 px.
- `main_trunk_endpoints` (the automatic path), `target_endpoints` (Target) and
  `guided.box_path` use it.
- `longest_path_endpoints` is unchanged for `vstent analyse` and the research view.
- Found before P4k-C's development stage was looked at. Its first run used the old start and
  was stopped unread. The stage is re-run with the fix.

**Alternatives:** asking the user to click the ostium every time was rejected (the guided app
must work automatically). A catheter segmented as vessel could still win; the user can move the
start.

### D66 — Side-branch outflow: realistic for lesion-free vessels, but lesions then read too mild; not adopted (2026-11-02)
**Finding** (`reports/p4k_physics_dev.md`; pre-registered in docs/PLAN.md, P4k-C, commits
aa14e80 and 9de7e2b; CADICA development patients, product steps, pixel size assumed):

| | No outflow (product) | Murray outflow | Anchor |
|---|---:|---:|---:|
| Lesion-free vessels: FFR ≤ 0.80 | 52% | 6% | ≈ 13% (De Bruyne 2001) |
| Lesion-free vessels: median FFR | 0.73 | 0.95 | 0.89 |
| Lesions 50–70 / 70–90 / 90–98%: FFR ≤ 0.80 | 88 / 78 / 81% | 0 / 14 / 19% | 35 / 80 / 96% (FAME) |

**What it shows:**
- **The product's physics:** it calls half of the lesion-free vessels significant. Its high
  shares for graded lesions come largely from the same diffuse loss along long, thin paths:
  even 50–70% lesions read ≤ 0.80 in 88%, far above FAME's 35%.
- **The outflow model:** lesion-free vessels become realistic, but the lesions' own losses
  are then too small, because their depth is under-read (D63: 90–98% lesions measure about
  40% by edges).
- **The bottleneck:** lesion measurement, not the flow model. Neither physics can be right
  while severe lesions measure mild.

**Decision:**
- Not adopted, by the pre-registered rule. The product keeps `outflow: none`.
- The product's virtual FFR has poor specificity on real angiograms (52% of lesion-free
  vessels ≤ 0.80). This belongs in every limitations section until it is fixed.
- Amendment 1: the check is run again if a lesion reading (A or E) is adopted.

**Alternatives:**
- Adopting `murray` for its lesion-free result alone: rejected by the rule.
- A partial outflow exponent tuned to the anchors: rejected (tuning physics constants to
  data).

### D67 — Guided app display: every vessel outlined, faint vessels from the vesselness filter, mild narrowings amber (2026-11-02)
**Seen** (the user's feedback):
- Contours should cover every vessel as well as the target vessel.
- A stent was sometimes advised without any lesion marked.

**Decision (display only; no measurement changes):**
- **Outlines:** every vessel the model segments is outlined.
- **Faint vessels:** the Frangi vesselness components joined to the model's mask (T = 0.15,
  P4j's middle candidate) are outlined light blue. They are never measured; as a measurement
  change they failed P4j's guard (D62).
- **Lesion marks:**
  - narrowings of 30–50% are filled amber, with close-ups;
  - ≥ 50% stays red, and only red counts as "found" in the expert-box check;
  - when the FFR is ≤ 0.80 without any narrowing ≥ 30%, the app says the loss is spread
    along the vessel.

**Alternatives:** measuring on the faint vessels was rejected until a pre-registered check
supports it.

### D68 — The FFR is read where a pressure wire would sit (2026-11-02)
**Finding** (`reports/p4k_wire_dev.md`; pre-registered in docs/PLAN.md, P4k-C2, commit 526c0cd;
CADICA development patients): the FFR is read at the last point of the path whose QCA
reference diameter is ≥ 2.0 mm (FAVOR II's vessel size).
- Lesion-free vessels read ≤ 0.80: 52% → 37% (De Bruyne anchor about 13%).
- Lesions 50–70 / 70–90 / 90–98% read ≤ 0.80: 88 / 78 / 81% → 53 / 60 / 69% (FAME 35 / 80 / 96%).
- FAME distance 23 → 22 points.

**Decision:**
- The product reads the FFR, plans stents and evaluates them on the vessel up to the wire
  position. The path beyond it is still measured and shown.
- Physics unchanged (`outflow: none`).

**Corrected implementation** (the vessel cut at the wire point, as the plan's text says; the
first run solved the whole path and read the FFR there): lesion-free 42%, FAME distance 20
points. The rule is met again. The product uses the corrected rule (`qca.at_wire`).

**Still open:** lesion-free vessels are significant in 42%, about three times the anchor. The
remaining gap is lesion depth and diffuse loss: P4k-A, C3, E.

**Alternatives:** a different diameter limit chosen on the data was rejected (tuning). Murray
outflow failed its own rule (D66).

### D69 — The product's FFR pipeline: learned lesion reading (v2) with Murray side-branch outflow (2026-11-02)
**Finding** (pre-registered in docs/PLAN.md, P4k-C3 and its amendment; CADICA development
patients; anchors De Bruyne 2001 and FAME). Product before the change: lesion-free vessels
read ≤ 0.80 in 52%; FAME distance 23 points.

| Combination | Lesion-free ≤ 0.80 | 50–70 / 70–90 / 90–98% ≤ 0.80 | FAME distance | Combined |
|---|---:|---:|---:|---:|
| Reader v2 + Murray (adopted) | 19% | 35 / 52 / 66% | 20 | 0.258 |
| Reader v1 + Murray at the wire | 21% | 32 / 48 / 75% | 19 | 0.267 |
| Reader v2 + Murray at the wire | 15% | 29 / 42 / 66% | 25 | 0.270 |
| (FAME) | (≈ 13%) | (35 / 80 / 96%) | | |

**Decision:** `configs/product.yaml`:
- `lesion_reader`: models/lesion/lesion_cnn_v2-20261102-194645;
- `outflow`: murray;
- `wire`: false.

The app, `vstent analyse` and `vstent validate` use it. The rule was the end-to-end one, so
this supersedes D63's and P4k-A's patch-level decisions for the FFR pipeline. D68's wire
position is superseded for the product: with outflow, thin distal segments carry little flow.
It stays an option (`wire: true`).

**Still open:**
- Severe lesions are under-called: 70–90% → 52% and 90–98% → 66% read ≤ 0.80, against
  FAME's 80% and 96%. Lesion depth is still under-read.
- **Test patients** (run once): lesion-free 54% → 20%; 50–70 / 70–90 / 90–98% → 36 / 40 /
  52% (FAME 35 / 80 / 96%); FAME distance 22 → 28 points. The rule still holds, but severe
  lesions are under-called: the weakest point of the product.
- The virtual stenting check (P4m) re-run with this pipeline.

**Alternatives:**
- Reader v1 + Murray at the wire: fewer false negatives on 90–98% lesions (75% vs 66%), but a
  larger combined distance.
- Choosing by eye after seeing the results was rejected. The rule chose.

### D70 — The learned lesion depth only in vessels ≥ 2 mm (2026-11-03)
**Finding** (pre-registered in docs/PLAN.md, P4n, commit aeab59f; disclosed as found on
development data after D69; CADICA development patients).
- **Cause:** reader v2 read narrowings in thin distal segments (reference about 1.6–1.9 mm)
  as near-occlusions (lumen about 0.17 mm). The pre-PCI FFR fell to ~0.06, and no stent could
  fix it (the smallest stent is 2.25 mm).
- **Guard:** the reader's depth is applied only where the candidate's reference diameter is
  ≥ 2.0 mm (FAVOR II's vessel size); thinner vessels keep the edges' depth.

| D69 pipeline | Without guard | With guard |
|---|---:|---:|
| Lesion-free vessels ≤ 0.80 | 19.2% | 15.4% |
| 50–70 / 70–90 / 90–98% ≤ 0.80 | 35 / 52 / 66% | 32 / 44 / 66% |
| Combined distance (C3) | 0.2579 | 0.2539 |
| Post-PCI ≤ 0.80 (planned lesions) | 28% | 23% |
| Plans reaching 0.90 | 66% | 72% |
| Median post-PCI FFR (FFR-SEARCH 0.91) | 0.91 | 0.93 |

**Decision:** `configs/product.yaml` `lesion_min_ref_mm: 2.0`. Test patients once (C3 and
P4m).

**Note:** 70–90% lesions read ≤ 0.80 a little less often (52 → 44%). Some under-call remains;
the rule accepted the trade.

