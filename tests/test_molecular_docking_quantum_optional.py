from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("pulser")
pytest.importorskip("emu_sv")

from sciona.atoms.bio.molecular_docking.quantum_solver.atoms import (
    adiabaticquantumsampler,
    quantumproblemdefinition,
    solutionextraction,
)
from sciona.atoms.bio.molecular_docking.quantum_solver_d12.atoms import (
    adiabaticpulseassembler,
    interactionboundscomputer,
    quantumcircuitsampler,
    quantumsolutionextractor,
)


def _weighted_edge_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("a", weight=1.0)
    graph.add_node("b", weight=0.5)
    graph.add_edge("a", "b")
    return graph


def test_quantum_solver_d12_builds_real_pulser_sequence_and_samples_sv() -> None:
    graph = _weighted_edge_graph()
    coordinates = {"a": np.array([0.0, 0.0]), "b": np.array([6.0, 0.0])}

    u_min, u_max = interactionboundscomputer(coordinates, graph)
    assert u_min > 0
    assert u_max > 0

    register, parameters, permutation, backend_flags, _ = quantumproblemdefinition(
        graph,
        coordinates,
        1,
        False,
    )
    parameters["n_samples"] = 12
    parameters["dt"] = 1000

    sequence = adiabaticpulseassembler(register, parameters)
    assert sequence.get_duration() == 4000
    assert {"rydberg_global", "dmm_0"} <= set(sequence.declared_channels)

    counts = quantumcircuitsampler(
        parameters,
        register,
        permutation,
        backend_flags["run_qutip"],
        backend_flags["run_emu_mps"],
        backend_flags["run_sv"],
    )
    assert sum(counts.values()) == 12
    assert all(set(bitstring) <= {"0", "1"} for bitstring in counts)

    solutions, solution_counts = quantumsolutionextractor(counts, register, 1)
    assert len(solutions) == 1
    assert len(solution_counts) == 1


def test_quantum_solver_pipeline_uses_optional_pulser_backend() -> None:
    graph = _weighted_edge_graph()
    coordinates = {"a": np.array([0.0, 0.0]), "b": np.array([6.0, 0.0])}
    register, parameters, permutation, backend_flags, num_sol = quantumproblemdefinition(
        graph,
        coordinates,
        1,
        False,
    )
    parameters["n_samples"] = 10
    parameters["dt"] = 1000

    counts, final_register = adiabaticquantumsampler(
        register,
        parameters,
        permutation,
        backend_flags,
    )

    assert final_register is register
    assert sum(counts.values()) == 10
    assert len(solutionextraction(counts, final_register, num_sol)) == 1
