import importlib


def test_molecular_docking_add_quantum_link_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.molecular_docking.add_quantum_link") is not None
    assert importlib.import_module("sciona.probes.bio.molecular_docking_add_quantum_link") is not None
