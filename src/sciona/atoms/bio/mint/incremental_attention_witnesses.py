from __future__ import annotations

from typing import TypeVar

T = TypeVar("T", bound=type)


def witness_enable_incremental_state_configuration(cls: T) -> T:
    """The decorator returns the same class object with augmented bases."""
    return cls
