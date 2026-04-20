from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "src" / "sciona" / "atoms" / "bio" / "molecular_docking"
BUNDLE = ROOT / "review_bundle.json"
REMEDIATION = ROOT / "REMEDIATION.md"

SAFE_PUBREV005_FQDNS = {
    "sciona.atoms.bio.molecular_docking.add_quantum_link.addquantumlink",
    "sciona.atoms.bio.molecular_docking.build_complementary.constructcomplementarygraph",
    "sciona.atoms.bio.molecular_docking.build_interaction_graph.networkx_weighted_graph_materialization",
    "sciona.atoms.bio.molecular_docking.build_interaction_graph.pair_distance_compatibility_check",
    "sciona.atoms.bio.molecular_docking.build_interaction_graph.weighted_interaction_edge_derivation",
    "sciona.atoms.bio.molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion",
    "sciona.atoms.bio.molecular_docking.greedy_mapping_d12.init_problem_context",
    "sciona.atoms.bio.molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate",
    "sciona.atoms.bio.molecular_docking.greedy_subgraph.greedy_maximum_subgraph",
    "sciona.atoms.bio.molecular_docking.map_to_udg.graphtoudgmapping",
}

REMEDIATED_CLASSICAL_FQDNS = {
    "sciona.atoms.bio.molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion",
    "sciona.atoms.bio.molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate",
    "sciona.atoms.bio.molecular_docking.greedy_subgraph.greedy_maximum_subgraph",
    "sciona.atoms.bio.molecular_docking.map_to_udg.graphtoudgmapping",
}

QUANTUM_PUBREV005_FQDNS = {
    "sciona.atoms.bio.molecular_docking.quantum_solver.adiabaticquantumsampler",
    "sciona.atoms.bio.molecular_docking.quantum_solver.quantumproblemdefinition",
    "sciona.atoms.bio.molecular_docking.quantum_solver.solutionextraction",
}

QUANTUM_PUBREV028_FQDNS = {
    "sciona.atoms.bio.molecular_docking.quantum_solver_d12.adiabaticpulseassembler",
    "sciona.atoms.bio.molecular_docking.quantum_solver_d12.interactionboundscomputer",
    "sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumcircuitsampler",
    "sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolutionextractor",
    "sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolverorchestrator",
}


def _bundle() -> dict:
    return json.loads(BUNDLE.read_text())


def _rows_by_base_fqdn() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in _bundle()["rows"]:
        base_fqdn = row["atom_fqdn"].split("@", 1)[0]
        assert base_fqdn not in rows
        rows[base_fqdn] = row
    return rows


def test_molecular_docking_review_bundle_pubrev005_safe_rows_are_ready() -> None:
    bundle = _bundle()
    rows = _rows_by_base_fqdn()

    assert bundle["family"] == "bio.molecular_docking"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["review_status"] == "reviewed"
    assert bundle["trust_readiness"] == "ready_for_manifest_merge"

    for fqdn in SAFE_PUBREV005_FQDNS:
        row = rows[fqdn]
        assert row["atom_fqdn"] == f"{fqdn}@{row['source_path']}"
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "publishable_candidate"
        assert row["developer_semantic_verdict"] == "source_aligned_with_direct_audit"
        assert row["trust_readiness"] == "ready_for_manifest_merge"
        assert row["limitations"] == []
        assert row["required_actions"] == []
        if fqdn in REMEDIATED_CLASSICAL_FQDNS:
            assert row["audit_batch"] == "pubrev-005-remediation"
            assert row["audit_scope"] == "classical_molecular_docking_remediation"
        else:
            assert row["audit_batch"] == "pubrev-005"
            assert row["audit_scope"] == "wave_1_audit_completion"

        source_rel, _, line_text = row["source_path"].partition(":")
        assert line_text
        assert (REPO_ROOT / "src" / source_rel).exists()
        for source in row["authoritative_sources"]:
            assert (REPO_ROOT / source).exists()

        module_name, function_name = fqdn.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        source_text, _ = inspect.getsourcelines(function)
        assert f"def {function_name}(" in "".join(source_text)
        source_lines = (REPO_ROOT / "src" / source_rel).read_text().splitlines()
        assert source_lines[int(line_text) - 1].lstrip().startswith(f"def {function_name}(")


def test_molecular_docking_review_bundle_pubrev005_quantum_rows_are_ready() -> None:
    rows = _rows_by_base_fqdn()
    remediation = REMEDIATION.read_text()

    for fqdn in QUANTUM_PUBREV005_FQDNS:
        row = rows[fqdn]
        assert fqdn in remediation
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "publishable_candidate"
        assert row["developer_semantic_verdict"] == "source_aligned_optional_pulser_backend"
        assert row["trust_readiness"] == "ready_for_manifest_merge"
        assert row["audit_batch"] == "pubrev-005-quantum-optional"
        assert row["audit_scope"] == "quantum_optional_dependency_remediation"
        assert row["limitations"]
        assert row["required_actions"] == []
        source_rel, _, line_text = row["source_path"].partition(":")
        assert line_text
        assert (REPO_ROOT / "src" / source_rel).exists()
        for source in row["authoritative_sources"]:
            assert (REPO_ROOT / source).exists()

    for fqdn in REMEDIATED_CLASSICAL_FQDNS:
        assert fqdn in remediation
        assert rows[fqdn]["trust_readiness"] == "ready_for_manifest_merge"


def test_molecular_docking_review_bundle_pubrev028_quantum_solver_d12_rows_are_ready() -> None:
    rows = _rows_by_base_fqdn()
    remediation = REMEDIATION.read_text()

    for fqdn in QUANTUM_PUBREV028_FQDNS:
        row = rows[fqdn]
        assert fqdn in remediation
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "publishable_candidate"
        assert row["developer_semantic_verdict"] == "source_aligned_optional_pulser_backend"
        assert row["trust_readiness"] == "ready_for_manifest_merge"
        assert row["audit_batch"] == "pubrev-028-quantum-optional"
        assert row["audit_scope"] == "quantum_optional_dependency_remediation"
        assert row["limitations"]
        assert row["required_actions"] == []

        source_rel, _, line_text = row["source_path"].partition(":")
        assert line_text
        assert (REPO_ROOT / "src" / source_rel).exists()
        for source in row["authoritative_sources"]:
            assert (REPO_ROOT / source).exists()
