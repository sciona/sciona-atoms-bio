from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Iterable, Mapping
from typing import cast

import icontract
import networkx as nx
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_construct_mapping_state_via_greedy_expansion,
    witness_init_problem_context,
    witness_orchestrate_generation_and_validate,
)

Graph = nx.Graph
LatticeInstance = object
Subgraph = Collection[Hashable] | Mapping[Hashable, object]
GreedyMappingContext = dict[str, object]
NodeId = Hashable
MappingState = dict[str, object]
ScoredNode = dict[str, object]

def _as_lattice_graph(lattice_instance: object) -> nx.Graph:
    lattice = getattr(lattice_instance, "lattice", lattice_instance)
    if not hasattr(lattice, "nodes") or not hasattr(lattice, "neighbors"):
        raise TypeError("lattice_instance must expose a NetworkX-like lattice graph")
    return cast(nx.Graph, lattice)

def _ordered_nodes(graph: nx.Graph) -> list[Hashable]:
    nodes = list(graph.nodes())
    try:
        return sorted(nodes)
    except TypeError:
        return nodes

def _copy_state(mapping_state: MappingState) -> MappingState:
    return {
        "mapping": dict(cast(Mapping[Hashable, Hashable], mapping_state.get("mapping", {}))),
        "unmapping": dict(cast(Mapping[Hashable, Hashable], mapping_state.get("unmapping", {}))),
        "unexpanded_nodes": set(
            cast(Iterable[Hashable], mapping_state.get("unexpanded_nodes", set()))
        ),
        "removed_nodes": set(cast(Iterable[Hashable], mapping_state.get("removed_nodes", set()))),
        "rng_counter": int(cast(int, mapping_state.get("rng_counter", 0))),
    }

def _mapping(state: MappingState) -> dict[Hashable, Hashable]:
    return cast(dict[Hashable, Hashable], state["mapping"])

def _unmapping(state: MappingState) -> dict[Hashable, Hashable]:
    return cast(dict[Hashable, Hashable], state["unmapping"])

def _unexpanded(state: MappingState) -> set[Hashable]:
    return cast(set[Hashable], state["unexpanded_nodes"])

def _removed(state: MappingState) -> set[Hashable]:
    return cast(set[Hashable], state["removed_nodes"])

def _lattice_start_node(lattice: nx.Graph, used_sites: set[Hashable]) -> Hashable:
    nodes = _ordered_nodes(lattice)
    if not nodes:
        raise ValueError("lattice must contain at least one node")

    lattice_n = len(nodes)
    lattice_grid_size = int(math.sqrt(lattice_n))
    center_index = int(lattice_n / 2 + lattice_grid_size / 4)
    center_index = min(max(center_index, 0), lattice_n - 1)

    offsets = [0]
    for delta in range(1, lattice_n):
        offsets.extend((-delta, delta))
    for offset in offsets:
        candidate_index = center_index + offset
        if 0 <= candidate_index < lattice_n and nodes[candidate_index] not in used_sites:
            return nodes[candidate_index]
    raise ValueError("no free lattice site is available for the starting node")

def _average_lattice_degree(lattice_instance: object, lattice: nx.Graph) -> float:
    if hasattr(lattice_instance, "avg_degree"):
        return float(getattr(lattice_instance, "avg_degree"))
    if lattice.number_of_nodes() == 0:
        return 0.0
    return float(sum(dict(lattice.degree()).values()) / lattice.number_of_nodes())

def _score_candidates(
    graph: nx.Graph,
    lattice_instance: object,
    lattice: nx.Graph,
    previous_subgraphs: Collection[Subgraph],
    unplaced_nodes: list[Hashable],
    mapping: Mapping[Hashable, Hashable],
    remove_invalid_placement_nodes: bool,
    rng: random.Random,
) -> list[ScoredNode]:
    import random
    graph_n = max(graph.number_of_nodes(), 1)
    avg_degree = _average_lattice_degree(lattice_instance, lattice)
    scored_nodes: list[ScoredNode] = []

    for node in unplaced_nodes:
        degree_score = 1.0 - (abs(avg_degree - float(graph.degree(node))) / graph_n)
        non_adj_score = 0.0
        if not remove_invalid_placement_nodes:
            non_neighbors = [neighbor for neighbor in nx.non_neighbors(graph, node) if neighbor in mapping]
            non_adj_score = len(non_neighbors) / graph_n

        previous_count = sum(1 for subgraph in previous_subgraphs if node in subgraph)
        previous_score = (
            1.0 - (previous_count / len(previous_subgraphs)) if previous_subgraphs else 0.0
        )
        score = degree_score + non_adj_score + previous_score
        scored_nodes.append(
            {
                "node": node,
                "score": float(score),
                "tie_breaker": float(rng.random()),
                "degree_score": float(degree_score),
                "non_adjacent_score": float(non_adj_score),
                "previous_subgraph_score": float(previous_score),
            }
        )

    return scored_nodes

def _place_candidates_on_frontier(
    graph: nx.Graph,
    lattice: nx.Graph,
    state: MappingState,
    ordered_candidates: list[Hashable],
    current_lattice_node: Hashable,
    remove_invalid_placement_nodes: bool,
) -> None:
    mapping = _mapping(state)
    unmapping = _unmapping(state)
    unexpanded_nodes = _unexpanded(state)
    removed_nodes = _removed(state)
    already_placed_nodes = set(mapping)
    unplaced_nodes = list(ordered_candidates)
    free_lattice_neighbors = [
        neighbor for neighbor in lattice.neighbors(current_lattice_node) if neighbor not in unmapping
    ]

    for free_lattice_neighbor in free_lattice_neighbors:
        for unplaced_node in list(unplaced_nodes):
            valid_placement = True

            mapped_lattice_neighbors = [
                neighbor for neighbor in lattice.neighbors(free_lattice_neighbor) if neighbor in unmapping
            ]
            for mapped_neighbor in mapped_lattice_neighbors:
                if not graph.has_edge(unplaced_node, unmapping[mapped_neighbor]):
                    valid_placement = False
                    break

            if valid_placement:
                for graph_neighbor in graph.neighbors(unplaced_node):
                    if graph_neighbor in already_placed_nodes and not lattice.has_edge(
                        mapping[graph_neighbor], free_lattice_neighbor
                    ):
                        valid_placement = False
                        break

            if valid_placement:
                mapping[unplaced_node] = free_lattice_neighbor
                unmapping[free_lattice_neighbor] = unplaced_node
                already_placed_nodes.add(unplaced_node)
                unexpanded_nodes.add(unplaced_node)
                unplaced_nodes.remove(unplaced_node)
                break

    if remove_invalid_placement_nodes:
        removed_nodes.update(unplaced_nodes)

def _validate_mapping(context: GreedyMappingContext, state: MappingState) -> bool:
    graph = cast(nx.Graph, context["graph"])
    lattice = cast(nx.Graph, context["lattice"])
    mapping = _mapping(state)
    unmapping = _unmapping(state)

    for graph_node, lattice_node in mapping.items():
        if unmapping.get(lattice_node) != graph_node:
            return False
    for lattice_node, graph_node in unmapping.items():
        if mapping.get(graph_node) != lattice_node:
            return False

    for graph_u, graph_v in graph.edges():
        if graph_u in mapping and graph_v in mapping and not lattice.has_edge(
            mapping[graph_u], mapping[graph_v]
        ):
            return False

    for lattice_u, lattice_v in lattice.edges():
        if lattice_u in unmapping and lattice_v in unmapping and not graph.has_edge(
            unmapping[lattice_u], unmapping[lattice_v]
        ):
            return False

    return True

@register_atom(witness_init_problem_context)
@icontract.require(lambda graph: graph is not None, "graph cannot be None")
@icontract.require(lambda lattice_instance: lattice_instance is not None, "lattice_instance cannot be None")
@icontract.require(lambda previously_generated_subgraphs: previously_generated_subgraphs is not None, "previously_generated_subgraphs cannot be None")
@icontract.ensure(lambda result: result is not None, "init_problem_context output must not be None")
def init_problem_context(
    graph: Graph,
    lattice_instance: LatticeInstance,
    previously_generated_subgraphs: Collection[Subgraph],
    seed: int,
) -> GreedyMappingContext:
    """Collect the graph, lattice, historical subgraphs, and seed for D12 greedy mapping."""
    lattice = getattr(lattice_instance, "lattice", lattice_instance)
    return {
        "graph": graph,
        "lattice": lattice,
        "lattice_instance": lattice_instance,
        "previously_generated_subgraphs": list(previously_generated_subgraphs),
        "seed": int(seed),
    }

@register_atom(witness_construct_mapping_state_via_greedy_expansion)
@icontract.require(lambda problem_context: problem_context is not None, "problem_context cannot be None")
@icontract.require(lambda starting_node: starting_node is not None, "starting_node cannot be None")
@icontract.require(lambda mapping_state_in: mapping_state_in is not None, "mapping_state_in cannot be None")
@icontract.require(lambda considered_nodes: considered_nodes is not None, "considered_nodes cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "construct_mapping_state_via_greedy_expansion outputs must not be None")
def construct_mapping_state_via_greedy_expansion(
    problem_context: GreedyMappingContext,
    starting_node: NodeId,
    mapping_state_in: MappingState,
    considered_nodes: Collection[NodeId],
    remove_invalid_placement_nodes: bool,
    rank_nodes: bool,
) -> tuple[MappingState, list[ScoredNode]]:
    import random
    """Advance one D12 greedy-lattice mapping step without mutating the input state."""
    graph = cast(nx.Graph, problem_context["graph"])
    lattice = _as_lattice_graph(problem_context["lattice"])
    lattice_instance = problem_context["lattice_instance"]
    previous_subgraphs = cast(Collection[Subgraph], problem_context["previously_generated_subgraphs"])
    state = _copy_state(mapping_state_in)
    mapping = _mapping(state)
    unmapping = _unmapping(state)
    unexpanded_nodes = _unexpanded(state)

    if starting_node not in mapping:
        starting_lattice_node = _lattice_start_node(lattice, set(mapping.values()))
        mapping[starting_node] = starting_lattice_node
        unmapping[starting_lattice_node] = starting_node
        unexpanded_nodes.add(starting_node)

    current_node = cast(Hashable, mapping_state_in.get("current_node", starting_node))
    current_lattice_node = cast(Hashable, mapping_state_in.get("current_lattice_node", mapping[current_node]))
    removed_nodes = _removed(state)
    unplaced_nodes = [
        node for node in considered_nodes if node not in mapping and node not in removed_nodes
    ]

    rng_seed = int(cast(int, problem_context.get("seed", 0))) + int(state["rng_counter"])
    rng = random.Random(rng_seed)
    scored_nodes = _score_candidates(
        graph,
        lattice_instance,
        lattice,
        previous_subgraphs,
        list(unplaced_nodes),
        mapping,
        remove_invalid_placement_nodes,
        rng,
    )
    state["rng_counter"] = int(state["rng_counter"]) + 1

    if rank_nodes:
        scored_nodes.sort(
            key=lambda item: (cast(float, item["score"]), cast(float, item["tie_breaker"])),
            reverse=True,
        )
    ordered_candidates = [cast(Hashable, scored["node"]) for scored in scored_nodes]

    _place_candidates_on_frontier(
        graph,
        lattice,
        state,
        ordered_candidates,
        current_lattice_node,
        remove_invalid_placement_nodes,
    )

    return state, scored_nodes

@register_atom(witness_orchestrate_generation_and_validate)
@icontract.require(lambda problem_context: problem_context is not None, "problem_context cannot be None")
@icontract.require(lambda starting_node: starting_node is not None, "starting_node cannot be None")
@icontract.require(lambda mapping_state: mapping_state is not None, "mapping_state cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "orchestrate_generation_and_validate outputs must not be None")
def orchestrate_generation_and_validate(
    problem_context: GreedyMappingContext,
    starting_node: NodeId,
    remove_invalid_placement_nodes: bool,
    rank_nodes: bool,
    mapping_state: MappingState,
) -> tuple[MappingState, bool]:
    """Run the D12 greedy expansion loop from ``starting_node`` and validate the final map."""
    graph = cast(nx.Graph, problem_context["graph"])
    state = _copy_state(mapping_state)

    if starting_node not in _mapping(state):
        state, _ = construct_mapping_state_via_greedy_expansion(
            problem_context,
            starting_node,
            state,
            (),
            remove_invalid_placement_nodes,
            rank_nodes,
        )

    while _unexpanded(state):
        current_node = next(iter(_unexpanded(state)))
        _unexpanded(state).remove(current_node)
        current_lattice_node = _mapping(state)[current_node]
        removed_nodes = _removed(state)
        considered_nodes = [
            neighbor for neighbor in graph.neighbors(current_node) if neighbor not in removed_nodes
        ]
        step_state = dict(state)
        step_state["current_node"] = current_node
        step_state["current_lattice_node"] = current_lattice_node
        state, _ = construct_mapping_state_via_greedy_expansion(
            problem_context,
            starting_node,
            cast(MappingState, step_state),
            considered_nodes,
            remove_invalid_placement_nodes,
            rank_nodes,
        )

    return state, _validate_mapping(problem_context, state)
