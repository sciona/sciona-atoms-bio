import importlib


def test_molecular_docking_quantum_solver_d12_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.molecular_docking.quantum_solver_d12") is not None
    assert importlib.import_module("sciona.probes.bio.molecular_docking_quantum_solver_d12") is not None
