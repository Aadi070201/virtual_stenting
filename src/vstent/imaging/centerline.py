"""Vessel centreline between two user points on a binary vessel mask.

Steps: clean the mask → skeletonise → 8-connected pixel graph → prune spurs shorter than
``min_spur_px`` → snap the start / end points to the nearest skeleton pixels → shortest path
(networkx, Euclidean edge weights) → Savitzky–Golay smoothing → uniform arc-length resampling.
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

# Forward half of the 8-neighbourhood (each undirected edge is generated once).
_OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))

Pixel = tuple[int, int]


@dataclass(frozen=True, eq=False)
class Centerline:
    """Smoothed centreline sampled at uniform arc length.

    Attributes:
        points: (n, 2) positions (row, col) in pixels.
        s: (n,) arc length from the start point (px), uniformly spaced.
    """

    points: np.ndarray
    s: np.ndarray

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
) -> Centerline:
    """Savitzky–Golay smoothing of a pixel path (n, 2), then resampling every ``step_px``."""
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
    return Centerline(points=resampled, s=s_new)


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
    graph = vessel_graph(mask, min_spur_px)
    half_width = distance_transform_edt(clean_mask(mask))
    a = snap_point(graph, start, half_width, "start")
    b = snap_point(graph, end, half_width, "end")
    try:
        path = nx.shortest_path(graph, a, b, weight="weight")
    except nx.NetworkXNoPath as exc:
        raise ValueError(f"no vessel path connects {tuple(start)} and {tuple(end)}") from exc
    return smooth_and_resample(np.asarray(path), smooth_window_px, step_px=step_px)


def vessel_graph(mask: np.ndarray, min_spur_px: float = MIN_SPUR_PX) -> nx.Graph:
    """Pruned skeleton graph of a vessel mask (the graph ``trace_centerline`` searches)."""
    return prune_spurs(skeleton_graph(skeletonize(clean_mask(mask))), min_spur_px)


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
