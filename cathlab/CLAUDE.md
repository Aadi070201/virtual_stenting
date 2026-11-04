# vcathlab: the virtual cath lab (separate app)

> **Research use only.** Every screen, report and README says so.

- A separate app from vstent (the parent folder). **Never import `vstent` here, and never
  import `vcathlab` from vstent.** The only link is the exported dataset format
  (`docs/DATA_FORMAT.md`), read on the vstent side by a small loader.
- Code copied from vstent (the C-arm geometry, the X-ray chain's fitted ranges) is attributed
  in `docs/DECISIONS.md` and then evolves here on its own.
- Own environment: `cd cathlab && uv sync --all-extras`. Checks before every commit:
  `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`.
- The parent's ground rules apply: SI units inside (m, Pa, s), docstrings state units, fixed
  seeds, outputs saved with their config and git commit, no patient data, generic catheter and
  stent sizes only, never fabricate results.
- The plan is `docs/PLAN.md` (C1–C6); decisions go to `docs/DECISIONS.md`.
- Generated data goes to `cathlab/data/` (git-ignored).
