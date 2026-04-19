from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "src" / "sciona" / "atoms" / "bio" / "mint" / "fasta_dataset"
BUNDLE = ROOT / "review_bundle.json"

SAFE_PUBREV037_FQDNS = {
    "sciona.atoms.bio.mint.fasta_dataset.dataset_item_retrieval",
    "sciona.atoms.bio.mint.fasta_dataset.dataset_length_query",
    "sciona.atoms.bio.mint.fasta_dataset.dataset_state_initialization",
    "sciona.atoms.bio.mint.fasta_dataset.token_budget_batch_planning",
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


def _reference_keys() -> set[str]:
    data = json.loads((ROOT / "references.json").read_text())
    return set(data.get("atoms", {}))


def test_mint_fasta_dataset_review_bundle_pubrev037_rows_are_ready() -> None:
    bundle = _bundle()
    rows = _rows_by_base_fqdn()

    assert bundle["schema_version"] == "1.0"
    assert bundle["provider_repo"] == "sciona-atoms-bio"
    assert bundle["family"] == "bio.mint.fasta_dataset"
    assert bundle["review_status"] == "reviewed"
    assert bundle["semantic_verdict"] == "publishable_candidate"
    assert bundle["developer_semantic_verdict"] == "source_aligned_with_direct_audit"
    assert bundle["trust_readiness"] == "ready_for_manifest_merge"
    assert bundle["limitations"] == []
    assert bundle["required_actions"] == []
    assert set(rows) == SAFE_PUBREV037_FQDNS
    assert {row["atom_fqdn"] for row in bundle["rows"]} == _reference_keys()

    for idx, fqdn in enumerate(
        [
            "sciona.atoms.bio.mint.fasta_dataset.dataset_item_retrieval",
            "sciona.atoms.bio.mint.fasta_dataset.dataset_length_query",
            "sciona.atoms.bio.mint.fasta_dataset.dataset_state_initialization",
            "sciona.atoms.bio.mint.fasta_dataset.token_budget_batch_planning",
        ]
    ):
        row = rows[fqdn]
        assert row["atom_fqdn"] == f"{fqdn}@{row['source_path']}"
        assert row["review_status"] == "reviewed"
        assert row["semantic_verdict"] == "publishable_candidate"
        assert row["developer_semantic_verdict"] == "source_aligned_with_direct_audit"
        assert row["trust_readiness"] == "ready_for_manifest_merge"
        assert row["limitations"] == []
        assert row["required_actions"] == []
        assert row["description"]
        assert row["parameters"]
        assert row["io_specs"]
        assert row["review_record_path"] == (
            f"src/sciona/atoms/bio/mint/fasta_dataset/review_bundle.json#rows[{idx}]"
        )
        assert row["audit_batch"] == "pubrev-037"
        assert row["audit_scope"] == "sciona.atoms.bio.mint.fasta_dataset"

        source_rel, _, line_text = row["source_path"].partition(":")
        assert line_text
        assert (REPO_ROOT / "src" / source_rel).exists()
        for rel in row["authoritative_sources"]:
            assert (REPO_ROOT / rel).exists()

        module_name, function_name = fqdn.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        source_text, _ = inspect.getsourcelines(function)
        assert f"def {function_name}(" in "".join(source_text)
        source_lines = (REPO_ROOT / "src" / source_rel).read_text().splitlines()
        assert source_lines[int(line_text) - 1].lstrip().startswith(
            f"def {function_name}("
        )


def test_mint_fasta_dataset_cdg_provides_io_specs_for_all_pubrev037_atoms() -> None:
    cdg = json.loads((ROOT / "cdg.json").read_text())
    atomic_nodes = {
        f"sciona.atoms.bio.mint.fasta_dataset.{node['name']}": node
        for node in cdg["nodes"]
        if node.get("status") == "atomic"
    }

    assert set(atomic_nodes) == SAFE_PUBREV037_FQDNS
    for fqdn in SAFE_PUBREV037_FQDNS:
        node = atomic_nodes[fqdn]
        assert node["description"]
        assert node["inputs"] or node["outputs"]
        assert node["outputs"]

    state_node = atomic_nodes[
        "sciona.atoms.bio.mint.fasta_dataset.dataset_state_initialization"
    ]
    assert [spec["name"] for spec in state_node["inputs"]] == [
        "sequence_labels",
        "sequence_strs",
    ]
    assert [spec["name"] for spec in state_node["outputs"]] == ["dataset_state"]
    assert "fasta_file" not in json.dumps(state_node)
