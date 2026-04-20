from __future__ import annotations
from typing import Hashable, Mapping, Sequence

import numpy as np

Boolean = bool
CountDistribution = dict[str, int]
Integer = int
Permutation = int
ProblemGraph = object
QuantumRegister = object
RegisterCoordinates = Mapping[Hashable, Sequence[float] | np.ndarray]
Solution = list[Hashable]

"""Auto-generated atom wrappers following the sciona pattern."""

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_adiabaticquantumsampler, witness_quantumproblemdefinition, witness_solutionextraction
from .._pulser_optional import (
    extract_solutions,
    prepare_quantum_problem,
    sample_adiabatic_sequence,
)

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_quantumproblemdefinition)
@icontract.require(lambda graph: graph is not None, "graph cannot be None")
@icontract.require(lambda coordinates_layout: coordinates_layout is not None, "coordinates_layout cannot be None")
@icontract.require(lambda num_sol: num_sol is not None, "num_sol cannot be None")
@icontract.require(lambda display_info: display_info is not None, "display_info cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "QuantumProblemDefinition all outputs must not be None")
def quantumproblemdefinition(graph: ProblemGraph, coordinates_layout: RegisterCoordinates, num_sol: Integer, display_info: Boolean) -> tuple[QuantumRegister, dict[str, object], list[Permutation], dict[str, bool], Integer]:
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
    register, parameters, permutation, backend_flags = prepare_quantum_problem(
        graph,  # type: ignore[arg-type]
        coordinates_layout,
    )
    return register, parameters, permutation, backend_flags, num_sol

@register_atom(witness_adiabaticquantumsampler)
@icontract.require(lambda initial_register: initial_register is not None, "initial_register cannot be None")
@icontract.require(lambda simulation_parameters: simulation_parameters is not None, "simulation_parameters cannot be None")
@icontract.require(lambda permutation_list: permutation_list is not None, "permutation_list cannot be None")
@icontract.require(lambda backend_flags: backend_flags is not None, "backend_flags cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "AdiabaticQuantumSampler all outputs must not be None")
def adiabaticquantumsampler(initial_register: QuantumRegister, simulation_parameters: dict[str, object], permutation_list: list[Permutation], backend_flags: dict[str, bool]) -> tuple[CountDistribution, QuantumRegister]:
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
    count_dist = sample_adiabatic_sequence(
        simulation_parameters,
        initial_register,
        permutation_list,
        run_qutip=backend_flags.get("run_qutip", False),
        run_emu_mps=backend_flags.get("run_emu_mps", False),
        run_sv=backend_flags.get("run_sv", False),
    )
    return count_dist, initial_register

@register_atom(witness_solutionextraction)
@icontract.require(lambda measurement_counts: measurement_counts is not None, "measurement_counts cannot be None")
@icontract.require(lambda final_register: final_register is not None, "final_register cannot be None")
@icontract.require(lambda num_solutions: num_solutions is not None, "num_solutions cannot be None")
@icontract.ensure(lambda result: result is not None, "SolutionExtraction output must not be None")
def solutionextraction(measurement_counts: CountDistribution, final_register: QuantumRegister, num_solutions: Integer) -> list[Solution]:
    """Processes the raw results from the quantum sampler. It interprets the measurement count distribution to extract the top solutions for the optimization problem.

    Args:
        measurement_counts: A distribution of measurement outcomes from the simulation.
        final_register: The final state of the quantum register.
        num_solutions: The number of top solutions to extract.

    Returns:
        The final, processed solutions to the problem.
    """
    return extract_solutions(measurement_counts, final_register, num_solutions)
