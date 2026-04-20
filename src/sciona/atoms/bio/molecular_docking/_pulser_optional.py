from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import replace
from typing import Hashable, Mapping, Sequence

import networkx as nx
import numpy as np


class MissingQuantumOptionalDependency(ImportError):
    """Raised when the Pulser quantum optional dependency stack is absent."""


def _require_quantum_stack() -> dict[str, object]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import pulser as pl
        from emu_mps import BitStrings as MPSBitStrings
        from emu_mps import MPSBackend, MPSConfig
        from emu_sv import BitStrings as SVBitStrings
        from emu_sv import SVBackend, SVConfig, StateResult
        from pulser import Pulse, Register
        from pulser.channels.dmm import DMM
        from pulser.devices import AnalogDevice, DigitalAnalogDevice
        from pulser.waveforms import ConstantWaveform, InterpolatedWaveform
        from pulser_simulation import QutipEmulator
    except ImportError as exc:
        raise MissingQuantumOptionalDependency(
            "Install the Pulser quantum extra with `pip install sciona-atoms-bio[quantum]` "
            "to build and sample source-aligned neutral-atom quantum docking atoms."
        ) from exc

    return {
        "AnalogDevice": AnalogDevice,
        "ConstantWaveform": ConstantWaveform,
        "DMM": DMM,
        "DigitalAnalogDevice": DigitalAnalogDevice,
        "InterpolatedWaveform": InterpolatedWaveform,
        "MPSBackend": MPSBackend,
        "MPSBitStrings": MPSBitStrings,
        "MPSConfig": MPSConfig,
        "Pulse": Pulse,
        "QutipEmulator": QutipEmulator,
        "Register": Register,
        "SVBackend": SVBackend,
        "SVBitStrings": SVBitStrings,
        "SVConfig": SVConfig,
        "StateResult": StateResult,
        "pl": pl,
    }


def _coordinates_for_graph(
    graph: nx.Graph,
    coordinates_layout: Mapping[Hashable, Sequence[float] | np.ndarray],
) -> dict[Hashable, tuple[float, float]]:
    coords: dict[Hashable, tuple[float, float]] = {}
    for node in graph.nodes:
        if node not in coordinates_layout:
            continue
        coord = np.asarray(coordinates_layout[node], dtype=float)
        if coord.shape[0] < 2:
            raise ValueError("each coordinate must have at least two entries")
        coords[node] = (float(coord[0]), float(coord[1]))
    if not coords:
        raise ValueError("coordinates_layout must contain coordinates for graph nodes")
    return coords


def build_register(
    graph: nx.Graph,
    coordinates_layout: Mapping[Hashable, Sequence[float] | np.ndarray],
) -> object:
    """Build the Pulser Register used by the upstream quantum solver."""
    stack = _require_quantum_stack()
    register_cls = stack["Register"]
    return register_cls(_coordinates_for_graph(graph, coordinates_layout))  # type: ignore[operator]


def compute_interaction_bounds(
    register_coord: Mapping[Hashable, Sequence[float] | np.ndarray],
    graph: nx.Graph,
) -> tuple[float, float]:
    """Compute source-style C6 / distance**6 bounds for edges and non-edges."""
    stack = _require_quantum_stack()
    digital_analog = stack["DigitalAnalogDevice"]
    interaction_coefficient = float(digital_analog.interaction_coeff)  # type: ignore[attr-defined]

    nodes = list(register_coord.keys())
    edge_distances: list[float] = []
    non_edge_distances: list[float] = []

    for idx, left in enumerate(nodes):
        left_coord = np.asarray(register_coord[left], dtype=float)
        for right in nodes[idx + 1 :]:
            right_coord = np.asarray(register_coord[right], dtype=float)
            distance = float(np.linalg.norm(left_coord - right_coord))
            if distance <= 0.0:
                continue
            if graph.has_edge(left, right):
                edge_distances.append(distance)
            else:
                non_edge_distances.append(distance)

    if not edge_distances:
        raise ValueError("graph must contain at least one edge for quantum interaction bounds")

    u_min = interaction_coefficient / (max(edge_distances) ** 6)
    u_max = (
        interaction_coefficient / (min(non_edge_distances) ** 6)
        if non_edge_distances
        else u_min
    )
    return float(u_min), float(u_max)


def prepare_quantum_problem(
    graph: nx.Graph,
    coordinates_layout: Mapping[Hashable, Sequence[float] | np.ndarray],
    *,
    n_samples: int = 5000,
    dt: int = 10,
) -> tuple[object, dict[str, object], list[int], dict[str, bool]]:
    """Prepare the Pulser register and source-derived adiabatic parameters."""
    register_coord = _coordinates_for_graph(graph, coordinates_layout)
    register = build_register(graph, coordinates_layout)
    qubit_ids = list(getattr(register, "qubit_ids"))
    weights_by_node = nx.get_node_attributes(graph, "weight")
    weights = np.asarray([float(weights_by_node.get(node, 1.0)) for node in qubit_ids])
    max_weight = float(np.max(weights)) if weights.size else 1.0
    if max_weight <= 0.0:
        max_weight = 1.0

    u_min, u_max = compute_interaction_bounds(register_coord, graph)
    normalised_weights = weights / max_weight
    min_weight = float(np.min(normalised_weights)) if normalised_weights.size else 1.0
    weights_rev = 1.0 - normalised_weights
    spread_detuning = 31.06390344878028 * (1.0 - min_weight)

    params: dict[str, object] = {
        "amplitude_maximum": u_max,
        "detuning_maximum": 2.0 * u_max,
        "dmm_detuning": -spread_detuning,
        "dmm_map": dict(zip(qubit_ids, [float(value) for value in weights_rev])),
        "dt": int(dt),
        "duration": 4000,
        "graph": graph,
        "n_samples": int(n_samples),
        "u_min": u_min,
        "u_max": u_max,
    }
    return register, params, list(range(len(qubit_ids))), {
        "run_qutip": False,
        "run_emu_mps": False,
        "run_sv": True,
    }


def build_adiabatic_sequence(register: object, parameters: Mapping[str, object]) -> object:
    """Compile the upstream source's Pulser adiabatic sequence with DMM detuning."""
    stack = _require_quantum_stack()
    analog_device = stack["AnalogDevice"]
    constant_waveform = stack["ConstantWaveform"]
    dmm_cls = stack["DMM"]
    digital_analog_device = stack["DigitalAnalogDevice"]
    interpolated_waveform = stack["InterpolatedWaveform"]
    pulse_cls = stack["Pulse"]
    pl = stack["pl"]

    dmm = dmm_cls(
        clock_period=4,
        min_duration=16,
        max_duration=2**26,
        mod_bandwidth=8,
        bottom_detuning=-2 * np.pi * 20,
        total_bottom_detuning=-2 * np.pi * 2000,
    )
    mock_device = replace(
        digital_analog_device.to_virtual(),
        dmm_objects=(dmm, dmm_cls()),
        reusable_channels=True,
    )

    duration = int(parameters.get("duration", 4000))
    channel_name = "rydberg_global"
    dmm_channel_name = "dmm_0"
    qubit_ids = list(getattr(register, "qubit_ids"))
    raw_dmm_map = parameters.get("dmm_map", {})
    dmm_map = {
        qubit_id: float(raw_dmm_map.get(qubit_id, 0.0))  # type: ignore[union-attr]
        for qubit_id in qubit_ids
    }

    detuning_map = register.define_detuning_map(dmm_map, dmm_channel_name)  # type: ignore[attr-defined]
    omega_list = [1e-9, 4.850055051963065, 12.566370614359172, 11.882582923954192, 1e-9]
    delta_list = [
        -analog_device.channels[channel_name].max_abs_detuning,  # type: ignore[attr-defined]
        -23.735663070454464,
        2.429518510655143,
        13.330733571729787,
        31.06390344878028,
    ]

    sequence = pl.Sequence(register, mock_device)  # type: ignore[attr-defined]
    sequence.declare_channel(channel_name, channel_name)
    sequence.config_detuning_map(detuning_map, dmm_channel_name)
    sequence.add(
        pulse_cls(
            interpolated_waveform(duration, omega_list),
            interpolated_waveform(duration, delta_list),
            0,
        ),
        channel_name,
    )
    sequence.add_dmm_detuning(
        constant_waveform(duration, float(parameters.get("dmm_detuning", 0.0))),
        dmm_channel_name,
    )
    return sequence


def _evaluation_times(sequence: object, dt: int) -> tuple[int, list[float], float]:
    duration = int(sequence.get_duration())  # type: ignore[attr-defined]
    safe_dt = max(1, min(int(dt), max(1, duration // 2)))
    sequence_duration = duration // safe_dt * safe_dt
    eval_times = [time / sequence_duration for time in range(safe_dt, sequence_duration, safe_dt)]
    if not eval_times:
        eval_times = [1.0]
    return safe_dt, eval_times, eval_times[-1]


def _invert_permutation(permutation: Sequence[int]) -> list[int]:
    inverse = [0] * len(permutation)
    for idx, value in enumerate(permutation):
        inverse[int(value)] = idx
    return inverse


def _restore_permutation(bitstring: str, permutation: Sequence[int]) -> str:
    bits = list(bitstring)
    inverse = _invert_permutation(permutation)
    return "".join(bits[idx] for idx in inverse)


def sample_adiabatic_sequence(
    parameters: Mapping[str, object],
    register: object,
    list_perm: Sequence[int],
    *,
    run_qutip: bool,
    run_emu_mps: bool,
    run_sv: bool,
) -> dict[str, int]:
    """Run the source-style Pulser sequence on the requested emulator backend."""
    stack = _require_quantum_stack()
    sequence = build_adiabatic_sequence(register, parameters)
    n_samples = int(parameters.get("n_samples", 5000))
    dt = int(parameters.get("dt", 10))

    counts: Counter[str] | dict[str, int] | None = None
    if run_qutip:
        emulator = stack["QutipEmulator"].from_sequence(sequence)  # type: ignore[attr-defined]
        result = emulator.run()
        counts = result.sample_final_state(N_samples=n_samples)

    if run_sv:
        safe_dt, _, final_time = _evaluation_times(sequence, dt)
        bitstrings = stack["SVBitStrings"](evaluation_times=[final_time], num_shots=n_samples)  # type: ignore[operator]
        state = stack["StateResult"](evaluation_times=[final_time])  # type: ignore[operator]
        config = stack["SVConfig"](
            dt=safe_dt,
            observables=[bitstrings, state],
            log_level=logging.WARN,
        )  # type: ignore[operator]
        results = stack["SVBackend"](sequence, config=config).run()  # type: ignore[operator]
        counts = results.state[-1].sample(num_shots=n_samples)

    if run_emu_mps:
        safe_dt, _, final_time = _evaluation_times(sequence, dt)
        bitstrings = stack["MPSBitStrings"](evaluation_times=[final_time], num_shots=n_samples)  # type: ignore[operator]
        config = stack["MPSConfig"](
            dt=safe_dt,
            max_bond_dim=int(parameters.get("max_bond_dim", 200)),
            observables=[bitstrings],
            log_level=logging.WARN,
        )  # type: ignore[operator]
        results = stack["MPSBackend"](sequence, config=config).run()  # type: ignore[operator]
        counts_mps = results.get_result(bitstrings, 1.0)
        counts = {
            _restore_permutation(str(bitstring), list_perm): int(count)
            for bitstring, count in counts_mps.items()
        }

    if counts is None:
        raise ValueError("at least one quantum backend flag must be true")
    return {str(bitstring): int(count) for bitstring, count in counts.items()}


def extract_solutions(count_dist: Mapping[str, int], register: object, num_solutions: int) -> list[list[Hashable]]:
    """Decode the highest-count source bitstrings into selected qubit ids."""
    qubit_ids = list(getattr(register, "qubit_ids", []))
    if not qubit_ids and isinstance(register, Mapping):
        qubit_ids = list(register.keys())
    if not qubit_ids:
        qubit_ids = list(range(len(next(iter(count_dist)))))

    solutions: list[list[Hashable]] = []
    for bitstring in sorted(count_dist, key=lambda key: count_dist[key], reverse=True)[:num_solutions]:
        solution = [qubit_ids[idx] for idx, bit in enumerate(bitstring) if int(bit) == 1]
        solutions.append(solution)
    return solutions
