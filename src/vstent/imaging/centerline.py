"""Vessel centreline between two user points on a binary vessel mask.

Steps: clean the mask → skeletonise → 8-connected pixel graph → prune spurs shorter than
``min_spur_px`` → bridge gaps between aligned vessel ends (:func:`bridge_gaps`) → snap the
start / end points to the nearest skeleton pixels → shortest path (networkx, Euclidean edge
weights) → Savitzky–Golay smoothing → uniform arc-length resampling.

Gaps: a segmentation often breaks a vessel exactly at its tightest narrowing or at a total
occlusion, where the lumen is faint or absent. Two vessel ends that face each other across a
short gap are joined by a straight bridge; the centreline marks bridged points (``bridged``),
and QCA measures them from the image, never from the mask (P4c).
A point farther from its skeleton pixel than the vessel's half-width there plus
``SNAP_TOLERANCE_PX`` is refused: it is not on the segmented vessel, and snapping it would
silently measure a different (usually shorter) path.

All coordinates are image pixels as (row, col) floats; arc length is in pixels.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

MIN_SPUR_PX = 10.0
MIN_OBJECT_PX = 64
MAX_HOLE_PX = 64
SMOOTH_WINDOW_PX = 15
SMOOTH_POLYORDER = 3
STEP_PX = 1.0
SNAP_TOLERANCE_PX = 5.0  # beyond the vessel wall
# Gap bridging: two skeleton ends of different components at most BRIDGE_MAX_GAP_PX apart, each
# pointing at the other within BRIDGE_MAX_ANGLE_DEG. An end's direction is taken from the
# skeleton BRIDGE_DIRECTION_PX back to BRIDGE_END_SKIP_PX back (a skeleton hooks towards a corner
# in its last pixels). Bridge edges cost BRIDGE_PENALTY × their length, so a real vessel path is
# preferred.
BRIDGE_MAX_GAP_PX = 40.0
BRIDGE_MAX_ANGLE_DEG = 45.0
BRIDGE_DIRECTION_PX = 15
TRUNK_LOOKAHEAD_PX = 15  # main_trunk_endpoints: how far ahead a fork's branches are compared
BRIDGE_END_SKIP_PX = 4
BRIDGE_PENALTY = 2.0

# Forward half of the 8-neighbourhood (each undirected edge is generated once).
_OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))

Pixel = tuple[int, int]


@dataclass(frozen=True, eq=False)
class Centerline:
    """Smoothed centreline sampled at uniform arc length.

    Attributes:
        points: (n, 2) positions (row, col) in pixels.
        s: (n,) arc length from the start point (px), uniformly spaced.
        bridged: (n,) True where the path crosses a bridged gap in the mask (``None``: no
            information, e.g. a path not traced on a mask).
    """

    points: np.ndarray
    s: np.ndarray
    bridged: np.ndarray | None = None

    @property
    def bridged_mask(self) -> np.ndarray:
        """(n,) ``bridged``, all False when unknown."""
        return np.zeros(len(self.s), bool) if self.bridged is None else self.bridged

    @property
    def length(self) -> float:
        """Arc length (px)."""
        return float(self.s[-1])

    @property
    def tangents(self) -> np.ndarray:
        """(n, 2) unit tangents (row, col)."""
        t = np.gradient(self.points, axis=0)
        return t / np.linalg.norm(t, axis=1, keepdims=True)

    @property
    def normals(self) -> np.ndarray:
        """(n, 2) unit normals (row, col): tangents rotated by 90°."""
        t = self.tangents
        return np.stack([t[:, 1], -t[:, 0]], axis=1)


def clean_mask(
    mask: np.ndarray, min_object_px: int = MIN_OBJECT_PX, max_hole_px: int = MAX_HOLE_PX
) -> np.ndarray:
    """Boolean mask without specks smaller than ``min_object_px`` or holes up to ``max_hole_px``."""
    m = np.asarray(mask, dtype=bool)
    m = remove_small_objects(m, max_size=min_object_px - 1)
    return remove_small_holes(m, max_size=max_hole_px)


def skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    """8-connected graph of skeleton pixels, edge weight = Euclidean distance (px).

    A diagonal edge is dropped when both pixels share a 4-neighbour on the skeleton, so
    "staircase" corners do not look like junctions.
    """
    sk = np.asarray(skeleton, dtype=bool)
    rows, cols = np.nonzero(sk)
    graph = nx.Graph()
    graph.add_nodes_from(zip(rows.tolist(), cols.tolist(), strict=True))
    padded = np.pad(sk, 1)
    pr, pc = rows + 1, cols + 1
    for dr, dc in _OFFSETS:
        hit = padded[pr + dr, pc + dc]
        if dr != 0 and dc != 0:  # diagonal: skip if a 4-connected detour exists
            hit &= ~(padded[pr + dr, pc] | padded[pr, pc + dc])
        w = float(np.hypot(dr, dc))
        graph.add_weighted_edges_from(
            ((int(r), int(c)), (int(r + dr), int(c + dc)), w)
            for r, c in zip(rows[hit], cols[hit], strict=True)
        )
    return graph


def prune_spurs(graph: nx.Graph, min_length: float = MIN_SPUR_PX, max_rounds: int = 3) -> nx.Graph:
    """Remove branches shorter than ``min_length`` (px) that end in a free endpoint.

    A branch runs from an endpoint (degree 1) through degree-2 pixels to a junction
    (degree ≥ 3); the junction is kept. Isolated segments are never removed.
    """
    g = graph.copy()
    for _ in range(max_rounds):
        removed = False
        for end in [n for n, d in g.degree() if d == 1]:
            if end not in g or g.degree(end) != 1:
                continue
            branch, length, prev, node = [end], 0.0, None, end
            while True:
                nxt = [m for m in g.neighbors(node) if m != prev]
                if len(nxt) != 1:
                    break
                length += g[node][nxt[0]]["weight"]
                prev, node = node, nxt[0]
                if g.degree(node) != 2:
                    break
                branch.append(node)
            if g.degree(node) >= 3 and length < min_length:
                g.remove_nodes_from(branch)
                removed = True
        if not removed:
            break
    return g


def snap_to_graph(graph: nx.Graph, point) -> Pixel:
    """Skeleton pixel nearest to ``point`` = (row, col)."""
    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("the vessel mask has no skeleton")
    _, i = cKDTree(np.asarray(nodes, dtype=float)).query(np.asarray(point, dtype=float))
    return nodes[int(i)]


def snap_point(graph: nx.Graph, point, half_width: np.ndarray, label: str = "point") -> Pixel:
    """:func:`snap_to_graph`, refusing a point that is not on the segmented vessel: farther from
    its skeleton pixel than the half-width there (``half_width``: distance transform of the
    cleaned mask, px) plus ``SNAP_TOLERANCE_PX``. ``label`` names the point in the error."""
    node = snap_to_graph(graph, point)
    dist = float(np.hypot(*(np.asarray(point, dtype=float) - node)))
    if dist > half_width[node] + SNAP_TOLERANCE_PX:
        raise ValueError(
            f"{label} point {tuple(int(round(v)) for v in point)} is {dist:.0f} px from the "
            "nearest segmented vessel: it is not on the vessel mask (the segmentation may have "
            "missed that part of the vessel)"
        )
    return node


def smooth_and_resample(
    path: np.ndarray,
    window_px: int = SMOOTH_WINDOW_PX,
    polyorder: int = SMOOTH_POLYORDER,
    step_px: float = STEP_PX,
    bridged: np.ndarray | None = None,
) -> Centerline:
    """Savitzky–Golay smoothing of a pixel path (n, 2), then resampling every ``step_px``.
    ``bridged``: (n,) flags of the path pixels, carried to the resampled points (nearest)."""
    pts = np.asarray(path, dtype=float)
    window = min(window_px, pts.shape[0] - (1 - pts.shape[0] % 2))  # odd, ≤ n
    if window > polyorder:
        pts = savgol_filter(pts, window, polyorder, axis=0, mode="interp")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        raise ValueError("start and end points snap to the same skeleton pixel")
    n = max(2, int(round(s[-1] / step_px)) + 1)
    s_new = np.linspace(0.0, s[-1], n)
    resampled = np.stack([np.interp(s_new, s, pts[:, k]) for k in range(2)], axis=1)
    flags = None
    if bridged is not None:
        flags = np.asarray(bridged, dtype=float)[
            np.clip(np.searchsorted(s, s_new), 0, len(s) - 1)
        ].astype(bool)
    return Centerline(points=resampled, s=s_new, bridged=flags)


def trace_centerline(
    mask: np.ndarray,
    start,
    end,
    *,
    min_spur_px: float = MIN_SPUR_PX,
    smooth_window_px: int = SMOOTH_WINDOW_PX,
    step_px: float = STEP_PX,
) -> Centerline:
    """Centreline from ``start`` to ``end`` (each (row, col) px) along the vessel mask.

    Raises ``ValueError`` if a point is not on the vessel mask (:func:`snap_point`) or the two
    points do not lie on a connected part of the skeleton.
    """
    graph = vessel_graph(mask, min_spur_px, bridge=True)
    half_width = distance_transform_edt(clean_mask(mask))
    a = snap_point(graph, start, half_width, "start")
    b = snap_point(graph, end, half_width, "end")
    try:
        path = nx.shortest_path(graph, a, b, weight="weight")
    except nx.NetworkXNoPath as exc:
        raise ValueError(f"no vessel path connects {tuple(start)} and {tuple(end)}") from exc
    return smooth_and_resample(np.asarray(path), smooth_window_px, step_px=step_px,
                               bridged=bridge_flags(graph, path))  # fmt: skip


def vessel_graph(
    mask: np.ndarray, min_spur_px: float = MIN_SPUR_PX, *, bridge: bool = False
) -> nx.Graph:
    """Pruned skeleton graph of a vessel mask (the graph ``trace_centerline`` searches), with
    gaps bridged if ``bridge`` (:func:`bridge_gaps`)."""
    graph = prune_spurs(skeleton_graph(skeletonize(clean_mask(mask))), min_spur_px)
    return bridge_gaps(graph) if bridge else graph


def bridge_flags(graph: nx.Graph, path) -> np.ndarray:
    """(n,) True for the path's pixels that lie on a bridge (:func:`bridge_gaps`)."""
    return np.array([graph.nodes[n].get("bridge", False) for n in path], dtype=bool)


def _end_walk(graph: nx.Graph, end: Pixel, n_back: int = BRIDGE_DIRECTION_PX) -> list[Pixel]:
    """Skeleton pixels from ``end`` back along its branch, at most ``n_back`` steps (stops at a
    junction)."""
    walk, prev, node = [end], None, end
    while len(walk) <= n_back:
        nxt = [m for m in graph.neighbors(node) if m != prev]
        if len(nxt) != 1:
            break
        prev, node = node, nxt[0]
        walk.append(node)
    return walk


def bridge_gaps(
    graph: nx.Graph,
    *,
    max_gap_px: float = BRIDGE_MAX_GAP_PX,
    max_angle_deg: float = BRIDGE_MAX_ANGLE_DEG,
) -> nx.Graph:
    """Copy of ``graph`` in which gaps between aligned vessel ends are bridged.

    Each skeleton end (degree 1) gets an anchor ``BRIDGE_END_SKIP_PX`` steps back along its
    branch (a skeleton hooks towards a corner of the mask's end in its last pixels) and a
    direction from ``BRIDGE_DIRECTION_PX`` steps back to the anchor. Candidates: ends of
    different connected components at most ``max_gap_px`` apart, each pointing at the other
    within ``max_angle_deg``. Closest pairs first, each end bridged at most once. The two hooks
    are removed and the anchors joined by a straight pixel line whose pixels get
    ``bridge=True`` and whose edges cost ``BRIDGE_PENALTY`` × length. A bridge that would run
    through another skeleton pixel is not made.
    """
    g = graph.copy()
    component = {n: k for k, comp in enumerate(nx.connected_components(g)) for n in comp}
    walks = {e: _end_walk(g, e) for e, d in g.degree() if d == 1}
    walks = {e: w for e, w in walks.items() if len(w) - 1 >= BRIDGE_END_SKIP_PX + 3}
    if len(walks) < 2:
        return g
    ends = list(walks)
    anchors = np.asarray([walks[e][BRIDGE_END_SKIP_PX] for e in ends], dtype=float)
    dirs = anchors - np.asarray([walks[e][-1] for e in ends], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    cos_max = float(np.cos(np.deg2rad(max_angle_deg)))
    candidates = []
    tips = np.asarray(ends, dtype=float)
    for i, j in cKDTree(tips).query_pairs(max_gap_px):
        if component[ends[i]] == component[ends[j]]:
            continue
        v = anchors[j] - anchors[i]
        dist = float(np.linalg.norm(v))
        if dist > 0 and dirs[i] @ v / dist >= cos_max and dirs[j] @ -v / dist >= cos_max:
            candidates.append((dist, i, j))
    used: set[int] = set()
    for dist, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        hooks = set(walks[ends[i]][:BRIDGE_END_SKIP_PX]) | set(walks[ends[j]][:BRIDGE_END_SKIP_PX])
        n = max(2, int(np.ceil(dist)) + 1)
        raw = np.round(np.linspace(anchors[i], anchors[j], n)).astype(int)
        keep = np.r_[True, (np.diff(raw, axis=0) != 0).any(axis=1)]
        line = [tuple(map(int, p)) for p in raw[keep]]
        inner = line[1:-1]
        if any(p in g and p not in hooks for p in inner):
            continue
        g.remove_nodes_from(hooks)
        g.add_nodes_from(inner, bridge=True)
        for p, q in zip(line[:-1], line[1:], strict=True):
            g.add_edge(p, q, weight=BRIDGE_PENALTY * float(np.hypot(p[0] - q[0], p[1] - q[1])))
        used |= {i, j}
    return g


def longest_path_endpoints(
    mask: np.ndarray, min_spur_px: float = MIN_SPUR_PX
) -> tuple[Pixel, Pixel]:
    """Two far-apart skeleton points of the largest vessel component, wider end first.

    Double sweep: from any node, the farthest node ``a`` (path length); from ``a``, the
    farthest ``b`` (exact for tree-shaped skeletons). The pair is ordered so that the end where
    the vessel is wider (median distance transform over the first / last tenth of the path)
    comes first: flow enters at the proximal, wider end. Used to pick a path automatically.
    """
    graph = vessel_graph(mask, min_spur_px)
    if graph.number_of_nodes() == 0:
        raise ValueError("the vessel mask has no skeleton")
    component = graph.subgraph(max(nx.connected_components(graph), key=len))
    start = next(iter(component.nodes))
    dist = nx.single_source_dijkstra_path_length(component, start, weight="weight")
    a = max(dist, key=dist.get)
    dist, paths = nx.single_source_dijkstra(component, a, weight="weight")
    b = max(dist, key=dist.get)
    width = distance_transform_edt(clean_mask(mask))[tuple(np.asarray(paths[b]).T)]
    k = max(3, len(width) // 10)
    return (a, b) if np.median(width[:k]) >= np.median(width[-k:]) else (b, a)


def _ahead_width(graph: nx.Graph, node: Pixel, blocked: set, width: np.ndarray) -> float:
    """Mean half-width over up to ``TRUNK_LOOKAHEAD_PX`` skeleton pixels from ``node``, always
    stepping to the widest unvisited neighbour."""
    seen, vals, cur = set(blocked), [], node
    for _ in range(TRUNK_LOOKAHEAD_PX):
        seen.add(cur)
        vals.append(width[cur])
        nxt = [m for m in graph.neighbors(cur) if m not in seen]
        if not nxt:
            break
        cur = max(nxt, key=lambda m: width[m])
    return float(np.mean(vals))


def proximal_end(graph: nx.Graph, width: np.ndarray) -> Pixel:
    """The vessel tree's proximal end: of the largest skeleton component's ends, the one where
    the vessel is widest over its first ``TRUNK_LOOKAHEAD_PX`` pixels (the ostium or left main).
    An end of the longest path is not enough: with a short left main, the longest path runs
    from a side branch's tip to the LAD's end."""
    if graph.number_of_nodes() == 0:
        raise ValueError("the vessel mask has no skeleton")
    component = graph.subgraph(max(nx.connected_components(graph), key=len))
    ends = sorted(n for n in component.nodes if component.degree(n) <= 1)
    return max(ends or sorted(component.nodes), key=lambda n: _ahead_width(component, n, set(),
                                                                          width))  # fmt: skip


def main_trunk_endpoints(
    mask: np.ndarray, max_length_px: float | None = None, min_spur_px: float = MIN_SPUR_PX
) -> tuple[Pixel, Pixel]:
    """An automatic path along a main vessel: from the tree's proximal end
    (:func:`proximal_end`), follow the skeleton, taking at every fork the branch that is widest
    over the next ``TRUNK_LOOKAHEAD_PX`` pixels, until a vessel end or ``max_length_px`` of
    path. The longest path alone wanders through side branches and crossing vessels on real
    angiograms; the main vessel stays the widest."""
    graph = vessel_graph(mask, min_spur_px)
    width = distance_transform_edt(clean_mask(mask))
    a = proximal_end(graph, width)
    return a, _walk_trunk(graph, width, a, {a}, max_length_px)


def _walk_trunk(graph: nx.Graph, width: np.ndarray, start: Pixel, visited: set,
                max_length_px: float | None, length: float = 0.0) -> Pixel:  # fmt: skip
    """From ``start``, the widest branch at every fork (avoiding ``visited``) to a vessel end
    or ``max_length_px`` of path in all; the last pixel."""
    cur = start
    while max_length_px is None or length < max_length_px:
        nxt = [n for n in graph.neighbors(cur) if n not in visited]
        if not nxt:
            break
        n = max(nxt, key=lambda m: _ahead_width(graph, m, visited | {cur}, width))
        length += float(graph[cur][n].get("weight", 1.0))
        visited.add(n)
        cur = n
    return cur


def target_endpoints(
    mask: np.ndarray, point, max_length_px: float | None = None,
    min_spur_px: float = MIN_SPUR_PX,
) -> tuple[Pixel, Pixel]:  # fmt: skip
    """A path to a chosen vessel: from the vessel tree's proximal end
    (:func:`proximal_end`) through the skeleton pixel nearest ``point`` ((row, col)
    px, which must lie on the vessel), then on along the widest branch to that vessel's end,
    at most ``max_length_px`` in all. Raises ``ValueError`` off the vessel or without a
    connection."""
    graph = vessel_graph(mask, min_spur_px)
    width = distance_transform_edt(clean_mask(mask))
    node = snap_point(graph, point, width, "target")
    a = proximal_end(graph, width)
    try:
        lengths, paths = nx.single_source_dijkstra(graph, a, node, weight="weight")
    except nx.NetworkXNoPath as exc:
        raise ValueError("the chosen vessel is not connected to the main vessel tree") from exc
    return a, _walk_trunk(graph, width, node, set(paths), max_length_px, float(lengths))
