from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "src" / "sciona" / "atoms" / "bio" / "molecular_docking"
BUNDLE = ROOT / "review_bundle.json"


def _expected_atom_keys() -> set[str]:
    keys: set[str] = set()
    for refs_path in sorted(ROOT.rglob("references.json")):
        data = json.loads(refs_path.read_text())
        keys.update(data.get("atoms", {}).keys())
    return keys


def _subfamilies_with_uncertainty() -> set[str]:
    return {path.parent.name for path in ROOT.rglob("uncertainty.json")}


def test_molecular_docking_review_bundle_covers_all_rows() -> None:
    bundle = json.loads(BUNDLE.read_text())
    expected_keys = _expected_atom_keys()
    ready_subfamilies = _subfamilies_with_uncertainty()
    blocked_subfamilies = [
        summary["subfamily"]
        for summary in bundle["subfamily_summaries"]
        if not summary["has_uncertainty"]
    ]

    assert bundle["family"] == "bio.molecular_docking"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["review_status"] == "partial"
    assert bundle["trust_readiness"] == "blocked_on_uncertainty_backfill"
    assert bundle["authoritative_sources"] == [
        "src/sciona/atoms/bio/molecular_docking/**/references.json",
        "src/sciona/atoms/bio/molecular_docking/**/cdg.json",
        "src/sciona/atoms/bio/molecular_docking/**/matches.json",
        "src/sciona/atoms/bio/molecular_docking/**/uncertainty.json",
    ]
    assert bundle["limitations"] == [
        "Some subfamilies still lack uncertainty evidence: build_interaction_graph, greedy_mapping_d12, minimize_bandwidth, quantum_solver, quantum_solver_d12."
    ]
    assert bundle["required_actions"] == [
        "Backfill uncertainty evidence for: build_interaction_graph, greedy_mapping_d12, minimize_bandwidth, quantum_solver, quantum_solver_d12."
    ]
    assert len(bundle["rows"]) == len(expected_keys)
    assert {row["atom_fqdn"] for row in bundle["rows"]} == expected_keys
    assert ready_subfamilies == {"greedy_mapping", "mwis_sa"}
    assert blocked_subfamilies == [
        "build_interaction_graph",
        "greedy_mapping_d12",
        "minimize_bandwidth",
        "quantum_solver",
        "quantum_solver_d12",
    ]

    for idx, row in enumerate(bundle["rows"]):
        assert row["review_record_path"] == f"src/sciona/atoms/bio/molecular_docking/review_bundle.json#rows[{idx}]"
        assert row["source_path"].startswith("sciona/atoms/bio/molecular_docking/")
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "publishable_candidate"
        for rel in row["authoritative_sources"]:
            assert (REPO_ROOT / rel).exists()

        assert row["authoritative_sources"][0].startswith("src/sciona/atoms/bio/molecular_docking/")
        assert row["authoritative_sources"][1].endswith("cdg.json")
        assert row["authoritative_sources"][2].endswith("matches.json")

        if row["subfamily"] in ready_subfamilies:
            assert row["developer_semantic_verdict"] == "source_aligned_with_evidence"
            assert row["trust_readiness"] == "ready_for_manifest_merge"
            assert row["limitations"] == []
            assert row["required_actions"] == []
            assert row["authoritative_sources"][3].endswith("uncertainty.json")
        else:
            assert row["developer_semantic_verdict"] == "source_aligned_with_gap"
            assert row["trust_readiness"] == "blocked_on_uncertainty_backfill"
            assert row["required_actions"]
