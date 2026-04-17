from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILY_ROOT = ROOT / "src" / "sciona" / "atoms" / "bio" / "alphafold"
BUNDLE = FAMILY_ROOT / "review_bundle.json"


def _expected_atom_keys() -> set[str]:
    data = json.loads((FAMILY_ROOT / "references.json").read_text())
    return set(data.get("atoms", {}).keys())


def test_alphafold_review_bundle_covers_all_rows() -> None:
    bundle = json.loads(BUNDLE.read_text())
    expected_keys = _expected_atom_keys()

    assert bundle["schema_version"] == "1.0"
    assert bundle["bundle_id"] == "bio.alphafold.family_batch.review.v1"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["family"] == "bio.alphafold"
    assert bundle["review_status"] == "reviewed"
    assert bundle["semantic_verdict"] == "publishable_candidate"
    assert bundle["developer_semantic_verdict"] == "source_aligned_with_evidence"
    assert bundle["trust_readiness"] == "ready_for_manifest_merge"
    assert bundle["authoritative_sources"] == [
        "src/sciona/atoms/bio/alphafold/references.json",
        "src/sciona/atoms/bio/alphafold/cdg.json",
    ]
    for rel in bundle["authoritative_sources"]:
        assert (ROOT / rel).exists()
    assert bundle["limitations"] == []
    assert bundle["required_actions"] == []
    assert len(bundle["rows"]) == len(expected_keys)
    assert {row["atom_fqdn"] for row in bundle["rows"]} == expected_keys

    for idx, row in enumerate(bundle["rows"]):
        assert row["review_record_path"] == f"src/sciona/atoms/bio/alphafold/review_bundle.json#rows[{idx}]"
        assert row["source_path"] == row["atom_fqdn"].split("@", 1)[1]
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "publishable_candidate"
        assert row["developer_semantic_verdict"] == "source_aligned_with_evidence"
        assert row["trust_readiness"] == "ready_for_manifest_merge"
        assert row["limitations"] == []
        assert row["required_actions"] == []
        assert row["authoritative_sources"] == [
            "src/sciona/atoms/bio/alphafold/references.json",
            "src/sciona/atoms/bio/alphafold/cdg.json",
        ]
        for rel in row["authoritative_sources"]:
            assert (ROOT / rel).exists()
