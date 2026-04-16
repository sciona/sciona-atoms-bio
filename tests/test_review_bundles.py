from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "sciona" / "atoms" / "bio" / "molecular_docking"
BUNDLE = ROOT / "review_bundle.json"


def _expected_atom_keys() -> set[str]:
    keys: set[str] = set()
    for refs_path in sorted(ROOT.rglob("references.json")):
        data = json.loads(refs_path.read_text())
        keys.update(data.get("atoms", {}).keys())
    return keys


def test_molecular_docking_review_bundle_covers_all_rows() -> None:
    bundle = json.loads(BUNDLE.read_text())
    expected_keys = _expected_atom_keys()

    assert bundle["family"] == "bio.molecular_docking"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["review_status"] == "partial"
    assert bundle["trust_readiness"] == "blocked_on_uncertainty_backfill"
    assert bundle["authoritative_sources"] == [
        "src/sciona/atoms/bio/molecular_docking/**/references.json",
        "src/sciona/atoms/bio/molecular_docking/**/cdg.json",
        "src/sciona/atoms/bio/molecular_docking/**/matches.json",
    ]
    assert len(bundle["rows"]) == len(expected_keys)
    assert {row["atom_fqdn"] for row in bundle["rows"]} == expected_keys

    for idx, row in enumerate(bundle["rows"]):
        assert row["review_record_path"] == f"src/sciona/atoms/bio/molecular_docking/review_bundle.json#rows[{idx}]"
        assert row["source_path"].startswith("sciona/atoms/bio/molecular_docking/")
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "publishable_candidate"
        assert row["authoritative_sources"][0].startswith("src/sciona/atoms/bio/molecular_docking/")
        assert row["authoritative_sources"][1].endswith("cdg.json")
        assert row["authoritative_sources"][2].endswith("matches.json")
        assert row["trust_readiness"] == "blocked_on_uncertainty_backfill"
        assert row["required_actions"]

