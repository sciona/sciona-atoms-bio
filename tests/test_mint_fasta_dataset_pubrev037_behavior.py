from __future__ import annotations

import pytest
from icontract.errors import ViolationError

from sciona.atoms.bio.mint.fasta_dataset.atoms import (
    dataset_item_retrieval,
    dataset_length_query,
    dataset_state_initialization,
    token_budget_batch_planning,
)


def test_dataset_state_initialization_copies_aligned_inputs() -> None:
    labels = ["seq_a", "seq_b"]
    seqs = ["ACD", "WXYZ"]

    state = dataset_state_initialization(labels, seqs)
    labels.append("seq_c")
    seqs[0] = "mutated"

    assert state == {
        "sequence_labels": ["seq_a", "seq_b"],
        "sequence_strs": ["ACD", "WXYZ"],
    }


def test_dataset_state_initialization_rejects_misaligned_inputs() -> None:
    with pytest.raises(ViolationError):
        dataset_state_initialization(["seq_a"], ["ACD", "EFG"])


def test_length_and_item_retrieval_preserve_index_alignment() -> None:
    state = dataset_state_initialization(["short", "long"], ["AC", "ACDEFG"])

    assert dataset_length_query(state) == 2
    assert dataset_item_retrieval(state, 1) == ("long", "ACDEFG")


def test_token_budget_batch_planning_matches_sorted_greedy_rule() -> None:
    state = dataset_state_initialization(
        ["medium", "long", "short", "tiny"],
        ["ABCDE", "ABCDEFGHIJ", "ABC", "A"],
    )

    assert token_budget_batch_planning(
        state,
        toks_per_batch=12,
        extra_toks_per_seq=1,
    ) == [[3, 2], [0], [1]]


def test_token_budget_batch_planning_keeps_oversized_singletons() -> None:
    state = dataset_state_initialization(["too_long"], ["ABCDEFGHIJ"])

    assert token_budget_batch_planning(state, toks_per_batch=4) == [[0]]
