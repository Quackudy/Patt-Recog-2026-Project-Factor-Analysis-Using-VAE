"""FactorVAE architecture (add sibling packages under ``models/`` for new architectures)."""

from factor_vae.models.vae.factory import build_factor_vae
from factor_vae.models.vae.network import (
    FactorVAE,
    FeatureExtractor,
    FactorEncoder,
    FactorDecoder,
    FactorPredictor,
)

__all__ = [
    "build_factor_vae",
    "FactorVAE",
    "FeatureExtractor",
    "FactorEncoder",
    "FactorDecoder",
    "FactorPredictor",
]
