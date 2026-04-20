from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_adiabaticpulseassembler, witness_interactionboundscomputer, witness_quantumcircuitsampler, witness_quantumsolutionextractor, witness_quantumsolverorchestrator
from .._pulser_optional import (
    build_adiabatic_sequence,
    compute_interaction_bounds,
    extract_solutions,
    prepare_quantum_problem,
    sample_adiabatic_sequence,
)

# Type aliases for domain-specific types not in sciona.ghost.abstract.
PulseSequence = object  # pulser.Sequence (pulse-level quantum control object)
QuantumRegister = object  # pulser.Register (atom arrangement on QPU)


def _is_nx_graph(obj: object) -> bool:
    """Check if obj is a networkx Graph without a top-level import."""
    try:
        import networkx  # lazy import
        return isinstance(obj, networkx.Graph)
    except ImportError:
        return False


@register_atom(witness_quantumsolverorchestrator)
@icontract.require(lambda graph: _is_nx_graph(graph), "graph must be a networkx Graph")
@icontract.require(lambda coordinates_layout: isinstance(coordinates_layout, dict), "coordinates_layout must be a dict")
@icontract.require(lambda num_sol: num_sol >= 1, "num_sol must be >= 1")
@icontract.ensure(lambda result: isinstance(result[0], list), "solutions must be a list")
@icontract.ensure(lambda result: isinstance(result[1], dict), "count_dist must be a dict")
def quantumsolverorchestrator(graph: nx.Graph, coordinates_layout: dict[str, np.ndarray], num_sol: int, display_info: bool) -> tuple[list[list[str]], dict[str, int]]:
    """Solve the Maximum Weight Independent Set (MWIS) problem end-to-end on a neutral-atom quantum device.

    Builds an atomic register from graph node coordinates. Computes
    interaction-energy bounds to set pulse parameters. Runs an adiabatic
    quantum evolution on the chosen backend. Post-processes bitstring counts
    into ranked independent-set solutions.

    Args:
        graph: Weighted networkx graph for the molecular docking problem.
        coordinates_layout: Mapping from node id to spatial coordinates
            (in micrometres) for the atom register layout.
        num_sol: Number of top-ranked MWIS solutions to return; must be >= 1.
        display_info: If True, print progress and draw the register and graph.

    Returns:
        solutions: List of MWIS solutions, each a list of node ids; length
            equals num_sol.
        count_dist: Bitstring count distribution from the quantum simulation,
            mapping bitstrings to their measured counts.
    """
    register, parameters, permutation, backend_flags = prepare_quantum_problem(
        graph,
        coordinates_layout,
    )
    count_dist = sample_adiabatic_sequence(
        parameters,
        register,
        permutation,
        run_qutip=backend_flags["run_qutip"],
        run_emu_mps=backend_flags["run_emu_mps"],
        run_sv=backend_flags["run_sv"],
    )
    solutions = extract_solutions(count_dist, register, num_sol)
    return [[str(node) for node in solution] for solution in solutions], count_dist


@register_atom(witness_interactionboundscomputer)
@icontract.require(lambda register_coord: isinstance(register_coord, dict) and len(register_coord) > 0, "register_coord must be a non-empty dict")
@icontract.require(lambda graph: _is_nx_graph(graph), "graph must be a networkx Graph")
@icontract.ensure(lambda result: result[0] > 0, "u_min must be positive")
@icontract.ensure(lambda result: result[1] > 0, "u_max must be positive")
def interactionboundscomputer(register_coord: dict[str, np.ndarray], graph: nx.Graph) -> tuple[float, float]:
    """Compute the min and max interaction energy bounds across all atom pairs.

    u_min is the weakest interaction between connected nodes (largest edge
    distance). u_max is the strongest interaction between non-connected
    nodes (smallest complement distance). Both use the van-der-Waals
    coefficient divided by distance to the sixth power rule.

    Args:
        register_coord: Mapping from node id to spatial coordinates in
            micrometres for each atom.
        graph: Networkx graph whose edges define connected pairs; complement
            edges define non-connected pairs.

    Returns:
        u_min: Min interaction energy among connected pairs (u_min > 0).
        u_max: Max interaction energy among non-connected pairs
            (u_max >= u_min).
    """
    return compute_interaction_bounds(register_coord, graph)


@register_atom(witness_adiabaticpulseassembler)
@icontract.require(lambda register: register is not None, "register cannot be None")
@icontract.require(lambda parameters: isinstance(parameters, dict), "parameters must be a dict")
@icontract.require(lambda parameters: "duration" in parameters, "parameters must contain 'duration'")
@icontract.ensure(lambda result: result is not None, "pulse sequence must not be None")
def adiabaticpulseassembler(register: QuantumRegister, parameters: dict[str, object]) -> PulseSequence:
    """Build the time-dependent adiabatic pulse sequence for the neutral-atom device.

    Creates drive-frequency and detuning ramp waveforms from interpolated
    envelopes. Configures the Detuning Modulation Map (DMM) channel for
    per-atom weight control. Returns a locked pulse-schedule object ready for
    the emulator or hardware backend.

    Args:
        register: Pulser Register defining the spatial arrangement of atoms
            on the quantum processor.
        parameters: Dictionary of pulse settings including 'duration',
            'detuning_maximum', 'amplitude_maximum', 'dmm_map', and
            'dmm_detuning'.

    Returns:
        Locked pulse sequence object ready to pass to emulator or hardware.
    """
    return build_adiabatic_sequence(register, parameters)


@register_atom(witness_quantumcircuitsampler)
@icontract.require(lambda parameters: isinstance(parameters, dict), "parameters must be a dict")
@icontract.require(lambda register: register is not None, "register cannot be None")
@icontract.require(lambda list_perm: isinstance(list_perm, list), "list_perm must be a list")
@icontract.require(lambda run_qutip, run_emu_mps, run_sv: run_qutip or run_emu_mps or run_sv, "one backend flag must be true")
@icontract.ensure(lambda result: isinstance(result, dict) and len(result) > 0, "counts must be a non-empty dict")
@icontract.ensure(lambda result: all(isinstance(v, int) and v > 0 for v in result.values()), "all counts must be positive integers")
def quantumcircuitsampler(parameters: dict[str, object], register: QuantumRegister, list_perm: list[int], run_qutip: bool, run_emu_mps: bool, run_sv: bool) -> dict[str, int]:
    """Run the adiabatic pulse sequence on the chosen quantum backend and return bitstring counts.

    Builds the pulse sequence, then sends it to one of three backends:
    full-density-matrix, tensor-network, or state-vector. For tensor-network
    runs, applies an inverse permutation to restore the original qubit order.

    Args:
        parameters: Pulse settings (duration, amplitude, detuning, Detuning
            Modulation Map (DMM) values).
        register: Pulser Register layout for the quantum processor.
        list_perm: Permutation indices for bandwidth-optimised qubit ordering;
            inverted for tensor-network bitstrings.
        run_qutip: If True, use the full-density-matrix backend.
        run_emu_mps: If True, use the tensor-network backend.
        run_sv: If True, use the state-vector backend.

    Returns:
        Bitstring count distribution mapping measured bitstrings to their
        occurrence counts; total counts equal the number of shots (5000).
    """
    return sample_adiabatic_sequence(
        parameters,
        register,
        list_perm,
        run_qutip=run_qutip,
        run_emu_mps=run_emu_mps,
        run_sv=run_sv,
    )


@register_atom(witness_quantumsolutionextractor)
@icontract.require(lambda count_dist: isinstance(count_dist, dict) and len(count_dist) > 0, "count_dist must be a non-empty dict")
@icontract.require(lambda register: register is not None, "register cannot be None")
@icontract.require(lambda num_solutions: num_solutions >= 1, "num_solutions must be >= 1")
@icontract.ensure(lambda result: isinstance(result[0], list), "solutions must be a list")
@icontract.ensure(lambda result: isinstance(result[1], list), "solution_counts must be a list")
def quantumsolutionextractor(count_dist: dict[str, int], register: QuantumRegister, num_solutions: int) -> tuple[list[list[str]], list[int]]:
    """Post-processes the raw measurement count distribution to decode, rank, and filter the top-k bitstring solutions.

    Sorts bitstrings by descending count, decodes each bitstring into a list of
    qubit ids where the bit is 1 (corresponding to selected nodes in the
    independent set), and returns the top num_solutions results mapped back to
    the original graph node labelling.

    Args:
        count_dist: Bitstring-to-count mapping from the quantum sampler,
            sorted by descending frequency.
        register: Quantum register providing the qubit_ids mapping from
            bit positions to graph node labels.
        num_solutions: Number of top-ranked solutions to extract; must be >= 1.

    Returns:
        solutions: List of solutions, each a list of node ids selected in
            that bitstring; length equals num_solutions.
        solution_counts: Corresponding measurement counts for each solution.
    """
    solutions = [[str(node) for node in solution] for solution in extract_solutions(count_dist, register, num_solutions)]
    solution_counts = [
        int(count_dist[bitstring])
        for bitstring in sorted(count_dist, key=lambda key: count_dist[key], reverse=True)[:num_solutions]
    ]
    return solutions, solution_counts
