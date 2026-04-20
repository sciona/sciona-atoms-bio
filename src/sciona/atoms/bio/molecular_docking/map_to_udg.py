from __future__ import annotations

from itertools import combinations
from math import cos, pi, sin
from typing import cast

import icontract
import networkx as nx
import numpy as np
from sciona.ghost.registry import register_atom

from .map_to_udg_witnesses import witness_graphtoudgmapping


def _position_from_attrs(attrs: dict[str, object]) -> np.ndarray | None:
    for key in ("pos", "position", "coord", "coords"):
        value = attrs.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.shape == (2,) and np.all(np.isfinite(array)):
            return array
    return None


def _fallback_positions(G: nx.Graph) -> dict[object, np.ndarray] | None:
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: np.zeros(2, dtype=float)}
    if G.number_of_edges() == 0:
        return {node: np.array([2.0 * idx, 0.0], dtype=float) for idx, node in enumerate(nodes)}
    if G.number_of_edges() == len(nodes) * (len(nodes) - 1) // 2:
        return {
            node: np.array([0.25 * cos(2.0 * pi * idx / len(nodes)), 0.25 * sin(2.0 * pi * idx / len(nodes))])
            for idx, node in enumerate(nodes)
        }
    layout = nx.spring_layout(G, seed=0, dim=2)
    return {node: np.asarray(coords, dtype=float) for node, coords in layout.items()}


def _extract_positions(G: nx.Graph) -> dict[object, np.ndarray]:
    positions: dict[object, np.ndarray] = {}
    missing = False
    for node, attrs in G.nodes(data=True):
        position = _position_from_attrs(cast(dict[str, object], attrs))
        if position is None:
            missing = True
            break
        positions[node] = position
    if not missing:
        return positions
    fallback = _fallback_positions(G)
    if fallback is None:
        raise ValueError("graph must provide 2D node positions for UDG certification")
    return fallback


def _scaled_positions(G: nx.Graph, positions: dict[object, np.ndarray]) -> dict[object, np.ndarray]:
    max_edge_distance = 0.0
    for left, right in G.edges():
        distance = float(np.linalg.norm(positions[left] - positions[right]))
        max_edge_distance = max(max_edge_distance, distance)
    scale = 1.0 / max_edge_distance if max_edge_distance > 1.0 else 1.0
    return {node: coords * scale for node, coords in positions.items()}


def _certify_unit_disk_edges(
    G: nx.Graph,
    positions: dict[object, np.ndarray],
    radius: float,
    tolerance: float,
) -> None:
    edge_keys = {frozenset(edge) for edge in G.edges()}
    for left, right in combinations(G.nodes(), 2):
        distance = float(np.linalg.norm(positions[left] - positions[right]))
        has_input_edge = frozenset((left, right)) in edge_keys
        has_unit_disk_edge = distance <= radius + tolerance
        if has_input_edge and not has_unit_disk_edge:
            raise ValueError("input edge is longer than the certified unit-disk radius")
        if not has_input_edge and has_unit_disk_edge:
            raise ValueError("embedding would add a unit-disk edge not present in the input graph")


@register_atom(witness_graphtoudgmapping)  # type: ignore[untyped-decorator]
@icontract.require(lambda G: G is not None, "Input graph G cannot be None")
@icontract.require(lambda G: hasattr(G, "nodes") and hasattr(G, "edges"), "G must be NetworkX-like")
@icontract.ensure(lambda result: result is not None, "GraphToUDGMapping output must not be None")
def graphtoudgmapping(G: nx.Graph) -> nx.Graph:
    """Certify a graph as a unit-disk graph without adding layout-induced edges.

    The atom uses existing 2D node positions when present, falls back only for
    simple certifiable cases, scales edge lengths to radius one when safe, and
    raises ``ValueError`` if the resulting unit-disk rule would add or drop an
    edge.
    """
    H = G.copy()
    if H.number_of_nodes() == 0:
        H.graph["udg_radius"] = 1.0
        H.graph["udg_certified"] = True
        return H

    positions = _scaled_positions(H, _extract_positions(H))
    _certify_unit_disk_edges(H, positions, radius=1.0, tolerance=1e-9)

    for node, coords in positions.items():
        H.nodes[node]["pos"] = coords
    H.graph["udg_radius"] = 1.0
    H.graph["udg_certified"] = True
    H.graph["udg_edge_rule"] = "edge iff Euclidean node distance is <= udg_radius"
    return H
