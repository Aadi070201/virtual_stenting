"""App update latency (``vstent app-latency``) → ``reports/app_latency.{md,json}``.

Drives the real app script with ``streamlit.testing.v1.AppTest`` and the exported models:
synthetic mode and image mode (an ARCADE example after segmentation and an automatic path).
After the case is set up, the virtual-stent controls are changed repeatedly and two times are
recorded per update: the server time the app reports for its rerun, and the wall time of
``AppTest.run`` (script rerun + test harness). Browser rendering is not included.
"""

import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vstent import DISCLAIMER, __version__
from vstent.config import git_info
from vstent.sysinfo import cpu_name

APP_SCRIPT = Path(__file__).resolve().parent / "streamlit_app.py"
TARGET_MS = 300.0


def _server_ms(at) -> float:
    text = [c.value for c in at.caption if c.value.startswith("Server time")][-1]
    return float(re.search(r"(\d+) ms", text).group(1))


def _updates(at, n_slider: int) -> list[dict[str, float]]:
    """Change the stent controls and time every rerun."""
    actions = []
    max_start = at.slider(key="stent_start").max
    for k in range(n_slider):
        value = round((k % 10) / 10 * max_start * 2) / 2
        actions.append(("start", lambda v=value: at.slider(key="stent_start").set_value(v)))
    for length in (12.0, 22.0, 15.0):
        actions.append(("length", lambda v=length: at.selectbox(key="stent_len").set_value(v)))
    for dia in (2.75, 3.25, 3.0):
        actions.append(("diameter", lambda v=dia: at.selectbox(key="stent_dia").set_value(v)))
    out = []
    for name, act in actions:
        act()
        t0 = time.perf_counter()
        at.run()
        wall = (time.perf_counter() - t0) * 1e3
        if at.exception:
            raise RuntimeError(f"app raised during '{name}' update: {at.exception[0].value}")
        out.append({"control": name, "server_ms": _server_ms(at), "wall_ms": wall})
    return out


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    server = np.array([r["server_ms"] for r in rows])
    wall = np.array([r["wall_ms"] for r in rows])
    return {
        "n": len(rows),
        "server_median_ms": float(np.median(server)),
        "server_max_ms": float(server.max()),
        "wall_median_ms": float(np.median(wall)),
        "wall_max_ms": float(wall.max()),
    }


def measure(n_slider: int = 20, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Synthetic and image-mode update latencies (needs the exported models)."""
    from streamlit.testing.v1 import AppTest

    log = log or (lambda _msg: None)
    results: dict[str, Any] = {}

    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=300).run()
    at.toggle(key="place").set_value(True).run()
    rows = _updates(at, n_slider)
    results["synthetic"] = {"case": "Focal lesion preset", **_summary(rows), "rows": rows}
    log(f"synthetic: median {results['synthetic']['server_median_ms']:.0f} ms")

    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=300).run()
    at.radio(key="mode").set_value("Angiogram image").run()
    at.radio(key="source").set_value("ARCADE example").run()
    if at.exception or not at.selectbox(key="example").options:
        results["image"] = None
        log("image mode skipped: no ARCADE example available")
    else:
        example = at.selectbox(key="example").value
        t0 = time.perf_counter()
        at.button(key="auto").click().run()
        setup_ms = (time.perf_counter() - t0) * 1e3
        at.toggle(key="place").set_value(True).run()
        rows = _updates(at, n_slider)
        results["image"] = {
            "case": f"ARCADE stenosis/val {example}, automatic path",
            "path_setup_ms": setup_ms,
            **_summary(rows),
            "rows": rows,
        }
        log(f"image: median {results['image']['server_median_ms']:.0f} ms")
    return results


def write_latency_report(report_dir: str | Path = "reports", n_slider: int = 20, log=None) -> dict:
    """Measure and write ``app_latency.{md,json}``."""
    provenance = {
        **git_info(),
        "vstent_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu": cpu_name(),
    }
    results = measure(n_slider, log)
    modes = [m for m in ("synthetic", "image") if results.get(m)]
    worst = max(results[m]["server_max_ms"] for m in modes)
    report = {
        "target_ms": TARGET_MS,
        "passed": worst < TARGET_MS,
        "results": results,
        "provenance": provenance,
    }
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "app_latency.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    lines = [
        "# App update latency",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Generated by `vstent app-latency`. Do not edit by hand.",
        "",
        "The app script is driven with `streamlit.testing.v1.AppTest` and the exported models. "
        "After the case is set up (and, in image mode, segmented and traced), the virtual-stent "
        "controls are changed repeatedly: the start slider and the length / diameter selectors. "
        "*Server* = the rerun time the app reports itself; *wall* = `AppTest.run` including the "
        "test harness. Browser rendering is not included.",
        "",
        "| Mode | Case | Updates | Server median | Server max | Wall median | Wall max "
        f"| Target (< {TARGET_MS:g} ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode in modes:
        r = results[mode]
        ok = r["server_max_ms"] < TARGET_MS
        lines.append(
            f"| {mode} | {r['case']} | {r['n']} | {r['server_median_ms']:.0f} ms "
            f"| {r['server_max_ms']:.0f} ms | {r['wall_median_ms']:.0f} ms "
            f"| {r['wall_max_ms']:.0f} ms | {'yes' if ok else 'NO'} |"
        )
    if results.get("image"):
        lines += [
            "",
            f"Image mode, one-off: tracing the automatic path and running QCA + both FFR models "
            f"took {results['image']['path_setup_ms']:.0f} ms (cached afterwards; segmentation "
            "itself is cached per image).",
        ]
    lines += [
        "",
        f"CPU: {provenance['cpu']}. Commit `{provenance['git_commit']}` (dirty: "
        f"{provenance['git_dirty']}), vstent {provenance['vstent_version']}, "
        f"{provenance['created_utc']}.",
        "",
    ]
    (report_dir / "app_latency.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report
