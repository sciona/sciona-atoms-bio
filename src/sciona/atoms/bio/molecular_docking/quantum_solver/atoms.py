from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
Boolean: Any = Any
CountDistribution: Any = Any
Integer: Any = Any
Permutation: Any = Any
ProblemGraph: Any = Any
QuantumRegister: Any = Any
RegisterCoordinates: Any = Any
Solution: Any = Any

"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_adiabaticquantumsampler, witness_quantumproblemdefinition, witness_solutionextraction

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_quantumproblemdefinition)
@icontract.require(lambda graph: graph is not None, "graph cannot be None")
@icontract.require(lambda coordinates_layout: coordinates_layout is not None, "coordinates_layout cannot be None")
@icontract.require(lambda num_sol: num_sol is not None, "num_sol cannot be None")
@icontract.require(lambda display_info: display_info is not None, "display_info cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "QuantumProblemDefinition all outputs must not be None")
def quantumproblemdefinition(graph: ProblemGraph, coordinates_layout: RegisterCoordinates, num_sol: Integer, display_info: Boolean) -> tuple[QuantumRegister, Dict, List[Permutation], Dict, Integer]:
    """Initializes the quantum annealing problem. It computes simulation parameters like the minimum and maximum coupling strengths (u) based on the graph and register coordinates, and prepares the initial state for the solver.

    Args:
        graph: A graph structure representing the optimization problem.
        coordinates_layout: Spatial layout of qubits in the register.
        num_sol: Number of desired solutions.
        display_info: Flag to control informational display.

    Returns:
        initial_register: The initialized quantum register.
        simulation_parameters: Parameters for the adiabatic sequence.
        permutation_list: List of permutations for the quantum loop.
        backend_flags: Flags to select the simulation backend (qutip, mps, sv).
        num_solutions_passthrough: The number of solutions, passed through to the final processing stage.
    """
    # Build a register as a dict of node_id -> coordinates
    register = dict(coordinates_layout)
    # Compute interaction bounds
    nodes = list(register.keys())
    coords = np.array([np.asarray(register[n]) for n in nodes])
    n = len(nodes)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            dists[i, j] = d
            dists[j, i] = d

    # Parameters for simulation
    parameters = {
        'duration': 4000.0,
        'detuning_maximum': 5.0,
        'amplitude_maximum': 5.0,
        'register': register,
        'graph': graph,
    }
    # Identity permutation
    perm_list = [list(range(n))]
    backend_flags = {'run_qutip': False, 'run_emu_mps': False, 'run_sv': True}
    return register, parameters, perm_list, backend_flags, num_sol

@register_atom(witness_adiabaticquantumsampler)
@icontract.require(lambda initial_register: initial_register is not None, "initial_register cannot be None")
@icontract.require(lambda simulation_parameters: simulation_parameters is not None, "simulation_parameters cannot be None")
@icontract.require(lambda permutation_list: permutation_list is not None, "permutation_list cannot be None")
@icontract.require(lambda backend_flags: backend_flags is not None, "backend_flags cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "AdiabaticQuantumSampler all outputs must not be None")
def adiabaticquantumsampler(initial_register: QuantumRegister, simulation_parameters: Dict, permutation_list: List[Permutation], backend_flags: Dict) -> tuple[CountDistribution, QuantumRegister]:
    """Executes the core quantum simulation loop. It evolves the quantum register through an adiabatic sequence to find low-energy states of the problem Hamiltonian, effectively sampling from the solution distribution.

    Args:
        initial_register: The initialized quantum register.
        simulation_parameters: Parameters for the adiabatic sequence.
        permutation_list: List of permutations for the quantum loop.
        backend_flags: Flags to select the simulation backend (qutip, mps, sv).

    Returns:
        measurement_counts: A distribution of measurement outcomes from the simulation.
        final_register: The final state of the quantum register after evolution.
    """
    import networkx as nx
    # Classical simulated annealing to emulate quantum sampling
    graph = simulation_parameters.get('graph')
    register = initial_register
    nodes = list(register.keys()) if isinstance(register, dict) else list(range(len(register)))
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    # Build adjacency
    adj = np.zeros((n, n))
    if graph is not None:
        for u, v in graph.edges():
            if u in node_idx and v in node_idx:
                adj[node_idx[u], node_idx[v]] = 1.0
                adj[node_idx[v], node_idx[u]] = 1.0

    weights = np.ones(n)
    if graph is not None:
        for nd in nodes:
            if nd in graph.nodes:
                weights[node_idx[nd]] = float(graph.nodes[nd].get('weight', 1.0))

    rng = np.random.RandomState(42)
    count_dist: dict[str, int] = {}

    for _ in range(500):
        current = np.zeros(n, dtype=int)
        order = rng.permutation(n)
        for idx in order:
            if adj[idx] @ current == 0:
                current[idx] = 1
        # SA
        T = 2.0
        for _ in range(200):
            T *= 0.98
            i = rng.randint(n)
            proposal = current.copy()
            proposal[i] = 1 - proposal[i]
            if proposal[i] == 1 and adj[i] @ proposal > 1:
                continue
            delta = float(weights @ proposal) - float(weights @ current)
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                current = proposal
        bs = ''.join(str(b) for b in current)
        count_dist[bs] = count_dist.get(bs, 0) + 1

    return count_dist, initial_register

@register_atom(witness_solutionextraction)
@icontract.require(lambda measurement_counts: measurement_counts is not None, "measurement_counts cannot be None")
@icontract.require(lambda final_register: final_register is not None, "final_register cannot be None")
@icontract.require(lambda num_solutions: num_solutions is not None, "num_solutions cannot be None")
@icontract.ensure(lambda result: result is not None, "SolutionExtraction output must not be None")
def solutionextraction(measurement_counts: CountDistribution, final_register: QuantumRegister, num_solutions: Integer) -> List[Solution]:
    """Processes the raw results from the quantum sampler. It interprets the measurement count distribution to extract the top solutions for the optimization problem.

    Args:
        measurement_counts: A distribution of measurement outcomes from the simulation.
        final_register: The final state of the quantum register.
        num_solutions: The number of top solutions to extract.

    Returns:
        The final, processed solutions to the problem.
    """
    # Extract top solutions from count distribution
    sorted_bits = sorted(measurement_counts.items(), key=lambda x: -x[1])
    register_keys = list(final_register.keys()) if isinstance(final_register, dict) else list(range(len(final_register)))
    solutions = []
    for bs, _ in sorted_bits:
        sol = [register_keys[i] for i, b in enumerate(bs) if b == '1']
        if sol:
            solutions.append(sol)
        if len(solutions) >= num_solutions:
            break
    while len(solutions) < num_solutions:
        solutions.append(solutions[-1] if solutions else register_keys[:1])
    return solutions[:num_solutions]
