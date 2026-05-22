"""Shared defaults for sheaf model components."""

from typing import Final


DEFAULT_SHEAF_ACTIVATION: Final[str] = "tanh"
DEFAULT_AUGMENTED_LAPLACIAN: Final[bool] = True
DEFAULT_DROPOUT: Final[float] = 0
DEFAULT_INPUT_DROPOUT: Final[float] = 0
DEFAULT_RIGHT_WEIGHTS: Final[bool] = True
DEFAULT_LEFT_WEIGHTS: Final[bool] = True
