# Commit map: rewrite of 2026-11-02

On 2026-11-02 the messages of 20 commits on `phase2` were rewritten to remove an attribution
trailer (`git filter-branch --msg-filter`). Only the messages changed: every commit has the
same files (the final tree is identical), but each hash changed. Generated reports keep the
commit they were made from, so a report made from one of these commits cites its old hash;
this table maps it to the commit now in the history.

| Old | New | Commit |
|---|---|---|
| `acd6a5c` | `3fe4ec0` | P8: CADICA stenosis grading regenerated with the adopted segmenter |
| `cb3f600` | `ceb0c11` | P8: CADICA cine frames regenerated with the adopted segmenter (frame choice worse) |
| `28fb19b` | `02d0e9e` | P8: app latency re-measured with the adopted segmenter |
| `dd07ea8` | `309482f` | P8: hazard numbers from the regenerated reports; D54 (frame choice regression, recovered in P7) |
| `198d44f` | `8dd6820` | P8: MVP report regenerated |
| `d4484b6` | `dc16be1` | P7: plan with pre-registered targets (ED frames, measured flow velocity) |
| `c63c29a` | `fcb0506` | P7: plan amended before any data (two-view velocity targets, single-view foreshortening) |
| `ffb2173` | `9d1a30e` | P7: ED frames (R-wave pointer, ECG, images), motion-tracked transit, phasic flow velocity, beating-heart synthetic runs |
| `f7b9a33` | `964bb6a` | P7: plateau motion tracking, measured flow in vstent analyse (--flow, --hyperemic-run), measured-velocity input interval |
| `f5924e5` | `46dad08` | P7: app: ED frame of cine runs, measured flow velocity and the option to use it, case report flow section |
| `87ab49b` | `bf612c8` | P7: SR-31 covers the measured flow velocity |
| `1c65d1c` | `61dd6e8` | P7: run curves at confident pixels (0.9), ED rule in the CADICA cine report, docs and invention note |
| `9ddcbac` | `cc58272` | P7: velocity interval on the log ratio, two-run agreement guard (×1.5); calibration moved to fresh seeds 9164-9199 |
| `b5d16a3` | `4fd20de` | P7: velocity interval calibration on development seeds 9164-9199 |
| `dd511aa` | `420a104` | P7: calibrated velocity intervals in use (x/÷ 1.68 per run, 1.23 for two agreeing runs); D55 |
| `47409d8` | `4ebd43e` | P7: beating-heart synthetic report (held-out seeds 9200-9259; 8 of 9 targets met) |
| `f19c790` | `954784b` | P7: requirements SR-06, SR-44-SR-46 and hazard H-12 for the ED frames and the measured flow velocity |
| `804a8da` | `2bcf0fb` | P7: CADICA cine report with the end-diastolic rule (test 149/177 within the clinicians' span; target met) |
| `eb0c71a` | `9f2c14a` | P7: done; opacification chart with ED frames in the app, H-11, plan, progress, README, human actions |
| `2e8ac4c` | `9e72fe9` | P7: MVP report regenerated |
