"""Tree mode of the app (P3): side branches on the image, FFR along every path, stents on a
path. No Streamlit here, so it can be tested and timed; ``streamlit_app.py`` wires it up.
Units: SI internally, mm for display."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go

from vstent import DISCLAIMER, __version__
from vstent.config import git_info
from vstent.imaging.qca_tree import TreeQCA
from vstent.imaging.segment import THRESHOLD
from vstent.inference.models import latest_model_dir
from vstent.inference.pipeline import TreeAnalysis
from vstent.inference.tree_runtime import TREE_MODELS, TREE_ONNX_FILE
from vstent.physics.tree import TreeFFRResult
from vstent.planning.stent import Stent
from vstent.planning.tree_planner import TreePlanResult
from vstent.plotting import INK_MUTED, INK_SECONDARY, SERIES_1, SERIES_2, SERIES_3
from vstent.units import m_to_mm, pa_to_mmhg

ENV_TREE_SURROGATE = "VSTENT_TREE_SURROGATE_MODEL"

TREE_NOTE = (
    "Tree model (reduced-order, side branches with a Murray flow split; junction losses "
    "neglected). FFR from the tree solver; no uncertainty intervals for trees yet (the P1 "
    "intervals are for single paths)."
)


def path_samples(tq: TreeQCA, outlet: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """For every branch piece on the path to ``outlet``: (branch, centreline px (n, 2) as
    (row, col), path coordinate (m) of each sample)."""
    out = []
    for k, i0, i1, offset in tq.tree.path_offsets(outlet):
        pts = tq.branches[k].centerline.points[i0 : i1 + 1]
        out.append((k, pts, offset + tq.tree.branches[k].s[i0 : i1 + 1]))
    return out


def tree_overlay(
    image: np.ndarray,
    prob: np.ndarray | None,
    tq: TreeQCA | None,
    outlet: int = 0,
    stents: list[Stent] | None = None,
    points: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """RGB image: vessel mask tinted blue, main centreline yellow, side branches green (the
    selected path thicker), stents on the path orange, clicked points."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    if prob is not None:
        mask = prob >= THRESHOLD
        rgb[mask] = 0.65 * rgb[mask] + 0.35 * np.array([42, 120, 214], np.float32)
    out = rgb.astype(np.uint8)
    if tq is not None:
        on_path = {k for k, *_ in tq.tree.path_offsets(outlet)}
        for k, q in enumerate(tq.branches):
            pts = np.round(q.centerline.points[:, ::-1]).astype(np.int32)
            color = (255, 214, 10) if k == 0 else (27, 175, 122)
            cv2.polylines(out, [pts], False, color, 2 if k in on_path else 1, cv2.LINE_AA)
            cv2.putText(out, tq.tree.names[k], tuple(int(v) for v in pts[-1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)  # fmt: skip
        for _, pts, x in path_samples(tq, outlet):
            xy = np.round(pts[:, ::-1]).astype(np.int32)
            for st in stents or []:
                sel = (x >= st.s_start) & (x <= st.s_end)
                if sel.sum() >= 2:
                    cv2.polylines(out, [xy[sel]], False, (235, 104, 52), 3, cv2.LINE_AA)
    for k, (r, c) in enumerate(points or []):
        cv2.circle(out, (int(c), int(r)), 5, (27, 175, 122) if k == 0 else (235, 104, 52), -1)
    return out


def tree_ffr_figure(
    analysis: TreeAnalysis, outlet: int = 0, post: TreeFFRResult | None = None
) -> go.Figure:
    """FFR along every root-to-outlet path (tree solver), the main path as a single vessel,
    and (``post``) after the virtual stent; the selected path is drawn thicker."""
    tree, pre = analysis.qca.tree, analysis.result
    on_path = {k for k, *_ in tree.path_offsets(outlet)}
    fig = go.Figure()
    fig.add_hline(y=0.80, line={"color": INK_MUTED, "dash": "dash", "width": 1})
    x0 = m_to_mm(tree.branches[0].s)
    fig.add_scatter(x=x0, y=analysis.single.ffr, name="main as a single path",
                    line={"color": SERIES_2, "dash": "dash", "width": 1.5})  # fmt: skip
    for k in range(tree.n_branches):
        offset = tree.path_offsets(k)[-1][3]
        x = m_to_mm(offset + tree.branches[k].s)
        color = SERIES_1 if k == 0 else SERIES_3
        width = 3 if k in on_path else 1.5
        fig.add_scatter(x=x, y=pre.ffr[k], name=f"{tree.names[k]}, pre-PCI",
                        line={"color": color, "width": width})  # fmt: skip
        if post is not None:
            fig.add_scatter(x=x, y=post.ffr[k], name=f"{tree.names[k]}, after stent",
                            line={"color": color, "width": width, "dash": "dot"})  # fmt: skip
    fig.update_layout(
        height=380, margin={"l": 50, "r": 10, "t": 30, "b": 40},
        xaxis_title="Distance from the inlet along each path (mm)", yaxis_title="FFR",
        legend={"orientation": "h", "y": -0.25}, font={"color": INK_SECONDARY},
    )  # fmt: skip
    return fig


def branch_table(
    analysis: TreeAnalysis, post: TreeFFRResult | None = None, surrogate: list | None = None
) -> list[dict[str, Any]]:
    """One row per branch: length, MLD, max DS, outlet FFR (solver; surrogate; after stent)."""
    tq, pre = analysis.qca, analysis.result
    rows = []
    for k, (q, name) in enumerate(zip(tq.branches, tq.tree.names, strict=True)):
        row = {
            "Branch": name,
            "From (mm)": round(m_to_mm(tq.tree.path_offsets(k)[-1][3]), 1),
            "Length (mm)": round(float(q.s_mm[-1]), 1),
            "MLD (mm)": round(q.mld_mm, 2),
            "Max DS (%)": round(q.max_percent_stenosis),
            "Outlet FFR (solver)": round(float(pre.ffr[k][-1]), 3),
            "Share of inflow": f"{pre.q_outlet[k] / pre.q_inlet[0]:.0%}",
        }
        if surrogate is not None:
            row["Outlet FFR (tree surrogate)"] = round(float(surrogate[k][-1]), 3)
        if post is not None:
            row["After stent (solver)"] = round(float(post.ffr[k][-1]), 3)
        rows.append(row)
    return rows


def tree_report_json(
    analysis: TreeAnalysis,
    outlet: int,
    stents: list[Stent],
    post: TreeFFRResult | None,
    plans: TreePlanResult | None,
    image_name: str,
) -> str:
    """Case report (JSON) of a tree analysis: inputs, assumptions, branches, stents, plans."""
    tq, pre = analysis.qca, analysis.result
    report = {
        "disclaimer": DISCLAIMER,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "software": {"vstent": __version__, **git_info()},
        "input": {
            "mode": "image, tree",
            "image": image_name,
            "mm_per_px": tq.mm_per_px,
            "pixel_size_assumed": tq.calibration_assumed,
        },  # fmt: skip
        "assumptions": {
            "model": TREE_NOTE,
            "pa_mmhg": round(pa_to_mmhg(pre.pa), 2),
            "pv_mmhg": round(pa_to_mmhg(pre.pv), 2),
            "physics": analysis.params.model_dump(mode="json"),
            "outlet_resistances_kept_after_stent": True,
        },
        "branches": branch_table(analysis, post),
        "main_path_single_vessel_ffr": round(float(analysis.single.ffr[-1]), 4),
        "side_paths_not_modelled": len(tq.dropped),
        "path_to": tq.tree.names[outlet],
        "stents": [
            {
                "start_mm": round(m_to_mm(s.s_start), 2),
                "length_mm": round(m_to_mm(s.length), 1),
                "diameter_mm": round(m_to_mm(s.diameter), 2),
            }
            for s in stents
        ],  # fmt: skip
        "plans": None
        if plans is None
        else {
            "message": plans.message,
            "backend": plans.backend,
            "top": [
                {
                    "stents": [
                        [
                            round(m_to_mm(s.s_start), 1),
                            round(m_to_mm(s.length)),
                            round(m_to_mm(s.diameter), 2),
                        ]
                        for s in p.stents
                    ],
                    "ffr_outlets": [round(f, 4) for f in p.ffr_outlets],
                    "meets_target": p.meets_target,
                }
                for p in plans.plans
            ],
        },  # fmt: skip
        "warnings": list(analysis.warnings),
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def tree_model_dir() -> Path | None:
    """Tree surrogate to use: ``$VSTENT_TREE_SURROGATE_MODEL`` or the newest export."""
    env = os.environ.get(ENV_TREE_SURROGATE)
    return Path(env) if env else latest_model_dir(TREE_MODELS, TREE_ONNX_FILE)
