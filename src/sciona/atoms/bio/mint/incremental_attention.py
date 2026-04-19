from __future__ import annotations

import uuid
from typing import Any

import icontract
from sciona.ghost.registry import register_atom
from .incremental_attention_witnesses import witness_enable_incremental_state_configuration


class FairseqIncrementalState:
    """Minimal Fairseq-style incremental state mixin used by MINT attention."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self.init_incremental_state()

    def init_incremental_state(self) -> None:
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return f"{self._incremental_state_id}.{key}"

    def get_incremental_state(self, incremental_state: dict | None, key: str) -> Any:
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: dict | None,
        key: str,
        value: Any,
    ) -> dict | None:
        if incremental_state is not None:
            incremental_state[self._get_full_incremental_state_key(key)] = value
        return incremental_state


@register_atom(witness_enable_incremental_state_configuration)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda cls: cls is not None, "cls cannot be None")
@icontract.ensure(lambda result: result is not None, "enable_incremental_state_configuration output must not be None")
def enable_incremental_state_configuration(cls: type) -> type:
    """Decorate a class with Fairseq/MINT incremental-state behavior.

    Args:
        cls: Class object to augment.

    Returns:
        The same class with ``FairseqIncrementalState`` prepended to its bases.
    """
    if FairseqIncrementalState not in cls.__bases__:
        cls.__bases__ = (FairseqIncrementalState,) + tuple(
            base for base in cls.__bases__ if base is not FairseqIncrementalState
        )
    return cls
