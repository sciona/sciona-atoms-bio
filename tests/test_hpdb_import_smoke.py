import importlib


def test_hpdb_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.hpdb") is not None
    assert importlib.import_module("sciona.probes.bio.hpdb") is not None
