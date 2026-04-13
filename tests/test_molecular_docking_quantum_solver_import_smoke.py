import importlib


def test_molecular_docking_quantum_solver_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.molecular_docking.quantum_solver") is not None
    assert importlib.import_module("sciona.probes.bio.molecular_docking_quantum_solver") is not None
