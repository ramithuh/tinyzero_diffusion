"""Tiny Zero Diffusion - A minimal diffusion language model framework."""

from tzd.models import BaseModel, DiffusionModel
from tzd.data import TinyShakespeareDataModule

__version__ = "0.0.1"

__all__ = [
    "BaseModel",
    "DiffusionModel",
    "TinyShakespeareDataModule",
]
