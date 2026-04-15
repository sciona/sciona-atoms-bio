from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""

from typing import Any, Collection

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_construct_mapping_state_via_greedy_expansion, witness_init_problem_context, witness_orchestrate_generation_and_validate

# Fallback type aliases for domain-specific types.
Graph = Any
LatticeInstance = Any
Subgraph = Any
GreedyMappingContext = Any
NodeId = Any
MappingState = Any
ScoredNode = Any


@register_atom(witness_init_problem_context)
@icontract.require(lambda graph: graph is not None, "graph cannot be None")
@icontract.require(lambda lattice_instance: lattice_instance is not None, "lattice_instance cannot be None")
@icontract.require(lambda previously_generated_subgraphs: previously_generated_subgraphs is not None, "previously_generated_subgraphs cannot be None")
@icontract.require(lambda seed: seed is not None, "seed cannot be None")
@icontract.ensure(lambda result: result is not None, "init_problem_context output must not be None")
def init_problem_context(graph: Graph, lattice_instance: LatticeInstance, previously_generated_subgraphs: Collection[Subgraph], seed: int) -> GreedyMappingContext:
    """Bootstraps immutable problem context for all later kernels: graph topology, lattice abstraction, lattice instance, previously generated subgraphs, and seed.

    Args:
        graph: Required; treated as immutable input state.
        lattice_instance: Required; used for placement/scoring context.
        previously_generated_subgraphs: Required; historical constraints for scoring.
        seed: Deterministic initialization input.

    Returns:
        Immutable state object carrying graph, lattice, lattice_instance,
        previously_generated_subgraphs, seed.
    """
    lattice = getattr(lattice_instance, 'lattice', lattice_instance)
    return {
        'graph': graph,
        'lattice': lattice,
        'lattice_instance': lattice_instance,
        'previously_generated_subgraphs': list(previously_generated_subgraphs),
        'seed': seed,
    }


@register_atom(witness_construct_mapping_state_via_greedy_expansion)
@icontract.require(lambda problem_context: problem_context is not None, "problem_context cannot be None")
@icontract.require(lambda starting_node: starting_node is not None, "starting_node cannot be None")
@icontract.require(lambda mapping_state_in: mapping_state_in is not None, "mapping_state_in cannot be None")
@icontract.require(lambda considered_nodes: considered_nodes is not None, "considered_nodes cannot be None")
@icontract.require(lambda remove_invalid_placement_nodes: remove_invalid_placement_nodes is not None, "remove_invalid_placement_nodes cannot be None")
@icontract.require(lambda rank_nodes: rank_nodes is not None, "rank_nodes cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "construct_mapping_state_via_greedy_expansion all outputs must not be None")
def construct_mapping_state_via_greedy_expansion(problem_context: GreedyMappingContext, starting_node: NodeId, mapping_state_in: MappingState, considered_nodes: Collection[NodeId], remove_invalid_placement_nodes: bool, rank_nodes: bool) -> tuple[MappingState, Collection[ScoredNode]]:
    """Builds and evolves immutable mapping state: initialize seed placement, score candidate graph nodes greedily, and extend mapping/frontier with feasible placements.

    Args:
        problem_context: Must come from init_problem_context.
        starting_node: Used by initialization stage.
        mapping_state_in: Threaded immutable state.
        considered_nodes: Nodes considered for expansion.
        remove_invalid_placement_nodes: Controls invalid-placement pruning.
        rank_nodes: Controls ranking behavior.

    Returns:
        mapping_state_out: New immutable state after extension.
        scored_nodes: Greedy scores computed from graph/lattice context.
    """
    graph = problem_context.get('graph') if isinstance(problem_context, dict) else getattr(problem_context, 'graph', None)
    lattice = problem_context.get('lattice', problem_context.get('lattice_instance')) if isinstance(problem_context, dict) else getattr(problem_context, 'lattice', None)

    # Extract current state
    if isinstance(mapping_state_in, dict):
        cur_mapping = dict(mapping_state_in.get('mapping', {}))
        cur_unmapping = dict(mapping_state_in.get('unmapping', {}))
        cur_unexpanded = set(mapping_state_in.get('unexpanded_nodes', set()))
    else:
        cur_mapping = {}
        cur_unmapping = {}
        cur_unexpanded = set()

    # Initialize seed placement if not yet mapped
    if starting_node not in cur_mapping:
        used_sites = set(cur_mapping.values())
        if hasattr(lattice, 'nodes'):
            available = [n for n in lattice.nodes() if n not in used_sites]
        elif hasattr(lattice, '__iter__'):
            available = [n for n in lattice if n not in used_sites]
        else:
            available = [i for i in range(1000) if i not in used_sites]
        if available:
            site = available[0]
            cur_mapping[starting_node] = site
            cur_unmapping[site] = starting_node
            cur_unexpanded.add(starting_node)

    # Score candidates
    scored_nodes = []
    for node in considered_nodes:
        if node in cur_mapping:
            continue
        if hasattr(graph, 'neighbors'):
            neighbors = list(graph.neighbors(node))
        elif hasattr(graph, 'adj'):
            neighbors = list(graph.adj.get(node, []))
        else:
            neighbors = []
        score = sum(1 for nb in neighbors if nb in cur_mapping)
        scored_nodes.append({'node': node, 'score': float(score)})

    # Filter invalid placements
    if remove_invalid_placement_nodes:
        scored_nodes = [s for s in scored_nodes if s['score'] > 0 or len(cur_mapping) <= 1]

    # Sort by score if ranking requested
    if rank_nodes:
        scored_nodes.sort(key=lambda s: -s['score'])

    # Greedily extend mapping
    used_sites = set(cur_mapping.values())
    if hasattr(lattice, 'nodes'):
        free_sites = [n for n in lattice.nodes() if n not in used_sites]
    elif hasattr(lattice, '__iter__'):
        free_sites = [n for n in lattice if n not in used_sites]
    else:
        free_sites = [i for i in range(1000) if i not in used_sites]

    for sn in scored_nodes:
        node = sn['node']
        if not free_sites or node in cur_mapping:
            continue
        site = free_sites.pop(0)
        cur_mapping[node] = site
        cur_unmapping[site] = node
        cur_unexpanded.add(node)

    mapping_state_out = {
        'mapping': cur_mapping,
        'unmapping': cur_unmapping,
        'unexpanded_nodes': cur_unexpanded,
    }
    return (mapping_state_out, scored_nodes)


@register_atom(witness_orchestrate_generation_and_validate)
@icontract.require(lambda problem_context: problem_context is not None, "problem_context cannot be None")
@icontract.require(lambda starting_node: starting_node is not None, "starting_node cannot be None")
@icontract.require(lambda remove_invalid_placement_nodes: remove_invalid_placement_nodes is not None, "remove_invalid_placement_nodes cannot be None")
@icontract.require(lambda rank_nodes: rank_nodes is not None, "rank_nodes cannot be None")
@icontract.require(lambda mapping_state: mapping_state is not None, "mapping_state cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "orchestrate_generation_and_validate all outputs must not be None")
def orchestrate_generation_and_validate(problem_context: GreedyMappingContext, starting_node: NodeId, remove_invalid_placement_nodes: bool, rank_nodes: bool, mapping_state: MappingState) -> tuple[MappingState, bool]:
    """Entry-point orchestration kernel (GreedyMapping): drives iterative expansion, invokes validity checks, and returns final generated subgraph mapping and inverse mapping.

    Args:
        problem_context: Immutable context from initialization.
        starting_node: Required entry node.
        remove_invalid_placement_nodes: Forwarded to expansion kernel.
        rank_nodes: Forwarded to expansion kernel.
        mapping_state: Threaded immutable state from expansion kernel.

    Returns:
        final_mapping_state: Final immutable mapping/unmapping state.
        is_valid: Result of final mapping validity check.
    """
    graph = problem_context.get('graph') if isinstance(problem_context, dict) else getattr(problem_context, 'graph', None)
    lattice = problem_context.get('lattice', problem_context.get('lattice_instance')) if isinstance(problem_context, dict) else getattr(problem_context, 'lattice', None)

    if isinstance(mapping_state, dict):
        cur_mapping = mapping_state.get('mapping', {})
        cur_unmapping = mapping_state.get('unmapping', {})
    else:
        cur_mapping = {}
        cur_unmapping = {}

    # Validate bijectivity
    is_valid = True
    for g_node, l_node in cur_mapping.items():
        if l_node not in cur_unmapping or cur_unmapping[l_node] != g_node:
            is_valid = False
            break
    if is_valid:
        for l_node, g_node in cur_unmapping.items():
            if g_node not in cur_mapping or cur_mapping[g_node] != l_node:
                is_valid = False
                break

    # Validate edge preservation
    if is_valid and hasattr(graph, 'edges'):
        for u, v in graph.edges():
            if u in cur_mapping and v in cur_mapping:
                lu, lv = cur_mapping[u], cur_mapping[v]
                if hasattr(lattice, 'has_edge') and not lattice.has_edge(lu, lv):
                    is_valid = False
                    break

    return (mapping_state, is_valid)
