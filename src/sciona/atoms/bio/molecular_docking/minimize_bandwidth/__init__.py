from sciona.ghost.registry import REGISTRY

from .atoms import *  # noqa: F401,F403

validate_symmetric_input_dense = REGISTRY["validate_symmetric_input_dense"]["impl"]
validate_symmetric_input_thresholded = REGISTRY["validate_symmetric_input_thresholded"]["impl"]
