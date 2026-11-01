# Software development plan (draft)

> **Research use only — not for clinical decisions.** Draft regulatory groundwork for review by
> a qualified person; not regulatory advice. Version 0.1, 2026-11-01, not reviewed.

IEC 62304 §5.1 for a class C software system ([SAFETY_CLASS.md](SAFETY_CLASS.md)). It
describes how the repository already works, and marks with **Gap** what a regulated release
needs on top.

## 1. Life-cycle model

Incremental. Work is planned as milestones with written acceptance criteria in
`docs/PLAN.md` (M0–M8 for the MVP, P1–P8 for phase 2). For each milestone:

1. implement with tests;
2. run the checks (section 5);
3. regenerate the affected reports from a clean commit;
4. tick the plan, log the result in `docs/PROGRESS.md` (including targets that were missed),
   record design decisions in `docs/DECISIONS.md`;
5. review with the founder before the next milestone.

Methods that are chosen on data are chosen on development data only and fixed in
`docs/DECISIONS.md` before held-out data are scored (e.g. D43–D47 on CADICA).

## 2. Deliverables (IEC 62304 → this repository)

| Deliverable | Clause | Where |
|---|---|---|
| Development plan | §5.1 | this file |
| Requirements | §5.2 | [REQUIREMENTS.md](REQUIREMENTS.md) from `requirements.yaml` |
| Architecture, SOUP | §5.3 | [ARCHITECTURE.md](ARCHITECTURE.md), [SOUP.md](SOUP.md), [sbom.cdx.json](sbom.cdx.json) |
| Detailed design | §5.4 | Docstrings (units in every docstring), `docs/PHYSICS.md`, `docs/DECISIONS.md`. **Gap:** a unit-level design description and its review (class C) |
| Unit implementation and verification | §5.5 | `src/vstent/`, `tests/` (305 tests; `slow` marks the long ones). **Gap:** §5.5.4 criteria per unit |
| Integration and system tests | §5.6–5.7 | Pipeline, CLI and app tests (`tests/test_pipeline.py`, `tests/test_app.py`); generated reports in `reports/`. **Gap:** system test plan and per-release records |
| Risk management | §7 | [RISK_MANAGEMENT_PLAN.md](RISK_MANAGEMENT_PLAN.md), [HAZARD_ANALYSIS.md](HAZARD_ANALYSIS.md), [TRACEABILITY.md](TRACEABILITY.md) |
| Release | §5.8 | Git tags (`v0.1.0`). **Gap:** release record, residual anomaly list, archived models |

## 3. Standards, methods and tools

- Python 3.14 (`.python-version`), packaged with `pyproject.toml`; environment from `uv.lock`
  (exact versions and hashes).
- `pytest` (tests), `ruff` (lint, format), `git` / GitHub (private repository).
- SI units inside, conversion only at the boundaries (`units.py`); fixed seeds; every model and
  report saved with its configuration and the git commit.
- **Gap:** the tools that produce evidence (pytest, the report generators, training) need a
  documented validation under ISO 13485 §4.1.6 (intended use of each tool, its risk, how it
  is checked).

## 4. Configuration management (§8)

- **Source and documents:** git. Development on `phaseN` branches, merged into `master` with
  `--no-ff`; releases tagged.
- **SOUP:** `uv.lock` fixes every third-party package; [SOUP.md](SOUP.md) and the SBOM are
  generated from it, and the build fails if the code imports an undeclared package (SR-39).
- **Data and models:** never in git (`data/`, `models/`, `runs/`, `checkpoints/`). Each model
  directory holds its configuration, the git commit and the ONNX file's SHA-256; each dataset
  is recorded in `docs/DATA.md` with its source, licence and hash. **Gap:** controlled,
  versioned storage for the released model files.
- **Patient data:** never in the repository; the validation tool refuses tracked locations
  (SR-33).

## 5. Verification

The checks before every commit:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -m "not slow" -q
```

These include the traceability check (`tests/test_regulatory.py`): every requirement,
hazard, test, report and code reference resolves, and the generated documents match their
sources. `uv run vstent regulatory --check` runs the same check alone.

**Gaps:**
- the checks run on the developer's computer; a continuous-integration server should run them
  on every push and keep the records;
- timing reports are valid only on an idle machine, so they are re-measured before a release.

## 6. Problem resolution (§9)

Today open issues are logged per milestone in `docs/PROGRESS.md`, and known limits in the
reports and `reports/mvp_report.md`. **Gap:** an issue tracker in which each problem gets a
safety assessment (does it affect a hazard?), a fix, a verification and a link to the
commit.

## 7. Risk management in development (§7)

Each change that touches an item updates `hazards.yaml` and `requirements.yaml` in the same
commit. A new measured weakness becomes a hazard or changes one (as the CADICA results did
for H-01, H-04 and H-11).

## 8. Machine-learning components

- **Data management:** datasets, splits and seeds recorded (`docs/DATA.md`, dataset cards);
  public data under their licences; held-out splits scored once.
- **Training:** reproducible from saved configurations; resumable checkpoints.
- **Verification:** ONNX parity with the trained model; evaluation reports on held-out and
  out-of-distribution data; conformal coverage.
- **Change control:** models are fixed at release. Any retraining follows a predetermined
  change control plan (FDA) or an algorithm change protocol (CDSCO software guidance, section
  12.4: data management, performance evaluation, retraining, update and rollback plans).
  **Gap:** neither is written yet.

## 9. Release (§5.8)

For each release:

1. tag the commit;
2. regenerate every report and `vstent regulatory` from it;
3. build the analysis-path package without the tools (ARCHITECTURE.md, section 6);
4. archive the model files with their hashes and evaluation reports;
5. write the release record: version, verification results, known residual anomalies, SBOM.

## 10. Maintenance (§6)

- SOUP updates: refresh `uv.lock`, review the SOUP.md difference, re-run the checks and the
  affected reports.
- Feedback from users and the literature enters problem resolution (section 6).
- Security: monitor published vulnerabilities of the SBOM components (IEC 81001-5-1).
  **Gap:** no monitoring yet.

## 11. Roles

| Role | Today |
|---|---|
| Development | One developer (the founder), working with an AI coding assistant |
| Verification | The same developer: automated tests and generated reports |
| Independent review | **Gap:** a reviewer other than the author for design, code, tests and documents (class C) |
| Clinical input | **Gap:** an interventional cardiologist for the intended use, hazards and usability |
| Regulatory | **Gap:** a regulatory adviser to review these drafts |
