from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.bio.molecular_docking.mwis_sa.atoms import (
    calculate_weight,
    is_independent_set,
    load_graphs_from_folder,
    to_qubo,
)
from sciona.atoms.bio.molecular_docking.mwis_sa.witnesses import (
    witness_calculate_weight,
    witness_is_independent_set,
    witness_load_graphs_from_folder,
    witness_to_qubo,
)
from sciona.ghost.abstract import AbstractArray, AbstractScalar


def test_load_graphs_from_folder_loads_npy_and_csv(tmp_path) -> None:
    npy_graph = np.array([[1.0, 0.0], [0.0, 2.0]])
    csv_graph = np.array([[3.0, 1.0], [1.0, 4.0]])

    np.save(tmp_path / "a.npy", npy_graph)
    np.savetxt(tmp_path / "b.csv", csv_graph, delimiter=",")

    graphs = load_graphs_from_folder(str(tmp_path))

    assert len(graphs) == 2
    assert np.array_equal(graphs[0], npy_graph)
    assert np.array_equal(graphs[1], csv_graph)


def test_is_independent_set_detects_edge_conflicts() -> None:
    graph = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 5.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )

    assert is_independent_set(graph, [0, 2]) is True
    assert is_independent_set(graph, [0, 1]) is False


def test_calculate_weight_sums_diagonal_node_weights() -> None:
    graph = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 5.0, 1.0],
            [0.0, 1.0, 3.5],
        ]
    )

    assert calculate_weight(graph, [0, 2]) == pytest.approx(7.5)


def test_to_qubo_encodes_weights_and_edge_penalties() -> None:
    graph = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 5.0, 1.0],
            [0.0, 1.0, 3.0],
        ]
    )

    qubo = to_qubo(graph, penalty=10.0)

    expected = np.array(
        [
            [-4.0, 10.0, 0.0],
            [10.0, -5.0, 10.0],
            [0.0, 10.0, -3.0],
        ]
    )
    assert np.array_equal(qubo, expected)


def test_to_qubo_rejects_underpowered_penalty() -> None:
    graph = np.array(
        [
            [4.0, 1.0],
            [1.0, 5.0],
        ]
    )

    with pytest.raises(ValueError, match="penalty must be at least 2 \\* max\\(node weight\\)"):
        to_qubo(graph, penalty=9.9)


def test_witnesses_match_scalar_and_array_shapes() -> None:
    graph = AbstractArray(shape=(3, 3), dtype="float64")
    nodes = AbstractArray(shape=(2,), dtype="int64")

    folder_meta = witness_load_graphs_from_folder(AbstractScalar(dtype="str"))
    independence_meta = witness_is_independent_set(graph, nodes)
    weight_meta = witness_calculate_weight(graph, nodes)
    qubo_meta = witness_to_qubo(graph, AbstractScalar(dtype="float64"))

    assert isinstance(folder_meta, AbstractArray)
    assert folder_meta.shape == (1,)
    assert isinstance(independence_meta, AbstractScalar)
    assert independence_meta.dtype == "bool"
    assert isinstance(weight_meta, AbstractScalar)
    assert weight_meta.dtype == "float64"
    assert isinstance(qubo_meta, AbstractArray)
    assert qubo_meta.shape == graph.shape
