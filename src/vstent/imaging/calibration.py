"""Frame-level calibration of the imaging chain (P4d).

The imaging-chain lumen model (:mod:`vstent.imaging.lumen_model`) has a blur σ_b and an edge
enhancement (a, σ_e) that belong to the X-ray system and its image processing, so they are the
same for every vessel of a frame. Fitted to one vessel they trade off against its lumen (on
real ARCADE frames, a single frame's vessels gave edge amounts from 1 to 4). Here they are
fitted once per frame, jointly over its longest branch-free vessel segments, and then held
fixed while each lumen is measured.

Units: px.
"""

import networkx as nx
import numpy as np

from vstent.imaging.centerline import smooth_and_resample, vessel_graph
from vstent.imaging.lumen_model import ChainCalibration, calibrate_chain
from vstent.imaging.qca import LEVEL, measure_diameters, measure_diameters_intensity

MIN_CHAIN_PX = 60  # branch-free skeleton chains at least this long
TRIM_PX = 10  # skipped at both ends (junctions, tips)
MAX_SEGMENTS = 6  # the longest chains are used
MAX_POINTS = 150  # a longer chain contributes its middle part
MIN_DIAMETER_PX = 2.5  # thinner segments carry too little signal


def branch_free_chains(graph: nx.Graph, min_px: int = MIN_CHAIN_PX) -> list[np.ndarray]:
    """Pixel chains (k, 2) of a skeleton graph between nodes of degree ≠ 2 (ends, junctions),
    at least ``min_px`` pixels long, longest first."""
    seen, out = set(), []
    for a in (n for n in graph if graph.degree(n) != 2):
        for b in graph.neighbors(a):
            if (a, b) in seen:
                continue
            path, prev, node = [a], a, b
            while graph.degree(node) == 2:
                path.append(node)
                prev, node = node, next(m for m in graph.neighbors(node) if m != prev)
            path.append(node)
            seen.update({(a, b), (path[-1], path[-2])})
            if len(path) >= min_px:
                out.append(np.asarray(path, dtype=float))
    return sorted(out, key=len, reverse=True)


def calibration_vessels(
    prob: np.ndarray, image: np.ndarray, *, max_segments: int = MAX_SEGMENTS
) -> list[tuple]:
    """Inputs of :func:`~vstent.imaging.lumen_model.calibrate_chain` from a frame: centreline,
    intensity-edge diameter, edge offset and reference (the median edge diameter) of up to
    ``max_segments`` long branch-free segments of the vessel map ``prob``."""
    out = []
    for chain in branch_free_chains(vessel_graph(prob >= LEVEL)):
        part = chain[TRIM_PX:-TRIM_PX]
        if len(part) > MAX_POINTS:
            mid = len(part) // 2
            part = part[mid - MAX_POINTS // 2 : mid + MAX_POINTS // 2]
        if len(part) < MIN_CHAIN_PX - 2 * TRIM_PX:
            continue
        cl = smooth_and_resample(part)
        edges = measure_diameters_intensity(image, cl, measure_diameters(prob, cl))
        d_med = float(np.median(edges.diameter_px))
        if d_med < MIN_DIAMETER_PX:
            continue
        offset = (edges.right_px + edges.left_px) / 2.0
        out.append((cl, edges.diameter_px, offset, np.full(len(cl.s), d_med)))
        if len(out) == max_segments:
            break
    return out


def calibrate_frame(prob: np.ndarray, image: np.ndarray) -> ChainCalibration | None:
    """The frame's blur and edge enhancement, fitted jointly over its vessels (``None`` if the
    vessel map has no usable segment)."""
    vessels = calibration_vessels(prob, image)
    return calibrate_chain(image, vessels) if vessels else None
