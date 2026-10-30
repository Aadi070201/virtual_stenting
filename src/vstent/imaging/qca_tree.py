"""QCA on the vessel tree (P2): the main path from a start point plus its side branches.

On the pruned skeleton (:func:`vessel_graph`), the shortest paths from the start point reach
every endpoint. The main path ends at ``end`` (or at the farthest endpoint). The other
endpoints are taken in order of decreasing path length. Each joins the tree at the last
skeleton pixel it shares with an accepted branch, and the rest of its path becomes a side
branch if it is long and wide enough. Longer paths go first, so a branch's trunk is accepted
before its own sub-branches.

A side branch is measured from where its centreline leaves the parent's lumen (the parent's
half-width at the take-off), so an ostial lesion is kept; only its far end is trimmed. Every
branch gets the single-path QCA (:func:`qca_from_centerline`). The take-off is the parent's
measured sample nearest to the junction pixel. In the tree model the side branch starts at
that sample; the short piece inside the parent's lumen is not modelled. Units: px in the image,
m in the profiles.
"""

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

from vstent.imaging.centerline import (
    clean_mask,
    smooth_and_resample,
    snap_to_graph,
    vessel_graph,
)
from vstent.imaging.qca import DEFAULT_MM_PER_PX, LEVEL, QCAResult, qca_from_centerline
from vstent.physics.tree import VesselTree

MIN_BRANCH_MM = 5.0  # side branch length outside the parent's lumen
MIN_BRANCH_RATIO = 0.3  # side branch median diameter / parent diameter at the take-off
MAX_DEPTH = 1  # side branches of the main path only (deeper levels are dropped and listed)
MIN_BRANCH_POINTS = 12  # centreline samples needed for a measurable side branch


@dataclass(frozen=True, eq=False)
class TreeQCA:
    """QCA of a vessel tree.

    Attributes:
        tree: the branches' SI profiles and how they connect (branch 0 = the main path).
        branches: the single-path QCA of every branch (centrelines in px).
        junction_px: (row, col) take-off pixel of every side branch (``None`` for the main).
        dropped: side paths found on the skeleton but not used, with the reason.
        mm_per_px: pixel size used (mm).
        calibration_assumed: whether the default pixel size was assumed.
    """

    tree: VesselTree
    branches: tuple[QCAResult, ...]
    junction_px: tuple[tuple[int, int] | None, ...]
    dropped: tuple[dict[str, Any], ...]
    mm_per_px: float
    calibration_assumed: bool


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def tree_qca(
    prob: np.ndarray,
    start,
    end=None,
    *,
    mm_per_px: float | None = None,
    level: float = LEVEL,
    min_branch_mm: float = MIN_BRANCH_MM,
    min_branch_ratio: float = MIN_BRANCH_RATIO,
    max_depth: int = MAX_DEPTH,
) -> TreeQCA:
    """QCA of the main path from ``start`` (to ``end``, or the farthest point) and its side
    branches, on the probability map ``prob`` (H, W). Points are (row, col) px."""
    mask = np.asarray(prob) >= level
    graph = vessel_graph(mask)
    a = snap_to_graph(graph, start)
    component = graph.subgraph(nx.node_connected_component(graph, a))
    dist, paths = nx.single_source_dijkstra(component, a, weight="weight")
    b = snap_to_graph(component, end) if end is not None else max(dist, key=dist.get)
    if b == a:
        raise ValueError("start and end points snap to the same skeleton pixel")
    scale = DEFAULT_MM_PER_PX if mm_per_px is None else float(mm_per_px)
    half_width = distance_transform_edt(clean_mask(mask))

    main = qca_from_centerline(
        prob, smooth_and_resample(np.asarray(paths[b])), mm_per_px=mm_per_px, level=level
    )
    accepted: list[dict[str, Any]] = [{"qca": main, "depth": 0, "parent": -1, "attach": 0,
                                       "junction": None}]  # fmt: skip
    owner = {node: 0 for node in paths[b]}
    dropped = []
    ends = [n for n, d in component.degree() if d == 1 and n not in owner]
    for e in sorted(ends, key=lambda n: -dist[n]):
        path = paths[e]
        j = max(i for i, node in enumerate(path) if node in owner)
        parent = owner[path[j]]
        junction = path[j]
        info = {"end_px": e, "junction_px": junction, "parent": parent}
        if accepted[parent]["depth"] + 1 > max_depth:
            dropped.append({**info, "reason": f"deeper than {max_depth} level(s)"})
            continue
        pts = np.asarray(path[j:], dtype=float)
        outside = np.linalg.norm(pts - np.asarray(junction), axis=1) > half_width[junction] + 1.0
        if not outside.any():
            dropped.append({**info, "reason": "does not leave the parent's lumen"})
            continue
        pts = pts[int(np.argmax(outside)) :]
        length_mm = _path_length(pts) * scale
        if length_mm < min_branch_mm or len(pts) < MIN_BRANCH_POINTS:
            dropped.append({**info, "reason": f"shorter than {min_branch_mm:g} mm",
                            "length_mm": length_mm})  # fmt: skip
            continue
        qca = qca_from_centerline(
            prob, smooth_and_resample(pts), mm_per_px=mm_per_px, level=level,
            trim_start_diameters=0.0,
        )  # fmt: skip
        ratio = float(np.median(qca.diameters.diameter_px)) / (2.0 * half_width[junction])
        if ratio < min_branch_ratio:
            dropped.append({**info, "reason": f"thinner than {min_branch_ratio:g} × the parent",
                            "diameter_ratio": ratio})  # fmt: skip
            continue
        parent_pts = accepted[parent]["qca"].centerline.points
        t = int(np.argmin(np.linalg.norm(parent_pts - np.asarray(junction), axis=1)))
        if not 1 <= t <= len(parent_pts) - 2:
            dropped.append({**info, "reason": "take-off outside the parent's measured part"})
            continue
        accepted.append({"qca": qca, "depth": accepted[parent]["depth"] + 1, "parent": parent,
                         "attach": t, "junction": junction})  # fmt: skip
        for node in path[j + 1 :]:
            owner.setdefault(node, len(accepted) - 1)

    tree = VesselTree(
        tuple(x["qca"].profile for x in accepted),
        tuple(x["parent"] for x in accepted),
        tuple(x["attach"] for x in accepted),
        ("main", *(f"side {k}" for k in range(1, len(accepted)))),
    )
    return TreeQCA(
        tree=tree,
        branches=tuple(x["qca"] for x in accepted),
        junction_px=tuple(x["junction"] for x in accepted),
        dropped=tuple(dropped),
        mm_per_px=scale,
        calibration_assumed=mm_per_px is None,
    )
