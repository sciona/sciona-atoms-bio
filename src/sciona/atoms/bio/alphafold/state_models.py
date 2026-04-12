from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field
from typing import Any

class AlphaFoldStructuralState(BaseModel):
    """Internal state for 3D equivariant structure module."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: jnp.ndarray | None = Field(default=None)
    frames: Any | None = Field(default=None) # Rigid objects
    pairs: jnp.ndarray | None = Field(default=None)
