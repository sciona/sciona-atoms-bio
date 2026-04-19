from __future__ import annotations

from typing import TypedDict

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dataset_item_retrieval,
    witness_dataset_length_query,
    witness_dataset_state_initialization,
    witness_token_budget_batch_planning,
)


class DatasetState(TypedDict):
    sequence_labels: list[str]
    sequence_strs: list[str]


@register_atom(witness_dataset_state_initialization)
@icontract.require(lambda sequence_labels, sequence_strs: len(sequence_labels) == len(sequence_strs), "sequence_labels and sequence_strs must have equal length")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def dataset_state_initialization(
    sequence_labels: list[str],
    sequence_strs: list[str],
) -> DatasetState:
    """Build a sequence dataset state snapshot from aligned in-memory inputs.

    Args:
        sequence_labels: Sequence identifiers.
        sequence_strs: Biological sequence strings.

    Returns:
        DatasetState with aligned label and sequence lists.
    """
    return {
        "sequence_labels": list(sequence_labels),
        "sequence_strs": list(sequence_strs),
    }


@register_atom(witness_dataset_length_query)
@icontract.require(lambda dataset_state: dataset_state is not None, "dataset_state cannot be None")
@icontract.ensure(lambda result: isinstance(result, int), "result must be int")
def dataset_length_query(dataset_state: DatasetState) -> int:
    """Return the number of labeled sequences in dataset_state.

    Args:
        dataset_state: DatasetState containing `sequence_labels`.

    Returns:
        Number of sequences, always non-negative.
    """
    return len(dataset_state["sequence_labels"])


@register_atom(witness_dataset_item_retrieval)
@icontract.require(lambda idx: isinstance(idx, int), "idx must be int")
@icontract.ensure(lambda result: result is not None, "result must not be None")
def dataset_item_retrieval(dataset_state: DatasetState, idx: int) -> tuple[str, str]:
    """Retrieve one labeled sequence by index.

    Args:
        dataset_state: DatasetState containing labels and sequences.
        idx: Valid integer index into the dataset.

    Returns:
        Pair of `(label, sequence_str)`.
    """
    return (dataset_state["sequence_labels"][idx], dataset_state["sequence_strs"][idx])


@register_atom(witness_token_budget_batch_planning)
@icontract.require(lambda toks_per_batch: isinstance(toks_per_batch, int), "toks_per_batch must be int")
@icontract.require(lambda extra_toks_per_seq: isinstance(extra_toks_per_seq, int), "extra_toks_per_seq must be int")
@icontract.ensure(lambda result: isinstance(result, list), "result must be list")
def token_budget_batch_planning(
    dataset_state: DatasetState,
    toks_per_batch: int,
    extra_toks_per_seq: int = 0,
) -> list[list[int]]:
    """Plan batches under a token budget.

    Args:
        dataset_state: DatasetState using `sequence_strs` for length-aware batching.
        toks_per_batch: Maximum token count per batch.
        extra_toks_per_seq: Per-sequence overhead tokens.

    Returns:
        List of batches, each represented as dataset indices.
    """
    seqs = dataset_state["sequence_strs"]
    sizes = [(len(seq), i) for i, seq in enumerate(seqs)]
    sizes.sort()
    batches: list[list[int]] = []
    current_batch: list[int] = []
    max_len = 0

    def _flush_current_batch() -> None:
        nonlocal current_batch, max_len
        if not current_batch:
            return
        batches.append(current_batch)
        current_batch = []
        max_len = 0

    for seq_len, idx in sizes:
        token_count = seq_len + extra_toks_per_seq
        if max(token_count, max_len) * (len(current_batch) + 1) > toks_per_batch:
            _flush_current_batch()
        max_len = max(max_len, token_count)
        current_batch.append(idx)

    _flush_current_batch()
    return batches
