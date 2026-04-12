from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .incremental_attention_witnesses import witness_enable_incremental_state_configuration

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_enable_incremental_state_configuration)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda cls: cls is not None, "cls cannot be None")
@icontract.ensure(lambda result: result is not None, "enable_incremental_state_configuration output must not be None")
def enable_incremental_state_configuration(cls: type) -> type:
    """Produces an incremental-state-enabled class/configuration as a pure class-level transformation.

    Args:
        cls: Base class object.

    Returns:
        Configured class/object representing incremental-state behavior without hidden mutation.
    """
    # Add incremental state tracking capability to a class
    if not hasattr(cls, '_incremental_state'):
        cls._incremental_state = {}
    original_init = cls.__init__ if hasattr(cls, '__init__') else lambda self: None

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._incremental_state = {}

    cls.__init__ = new_init
    cls.get_incremental_state = lambda self, key: self._incremental_state.get(key)
    cls.set_incremental_state = lambda self, key, value: self._incremental_state.update({key: value})
    return cls
