import importlib


def test_molecular_docking_minimize_bandwidth_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.molecular_docking.minimize_bandwidth") is not None
    assert importlib.import_module("sciona.probes.bio.molecular_docking_minimize_bandwidth") is not None
