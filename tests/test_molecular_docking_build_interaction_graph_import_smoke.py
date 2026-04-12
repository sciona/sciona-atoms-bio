import importlib


def test_molecular_docking_build_interaction_graph_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.bio.molecular_docking.build_interaction_graph") is not None
    assert importlib.import_module("sciona.probes.bio.molecular_docking_build_interaction_graph") is not None
