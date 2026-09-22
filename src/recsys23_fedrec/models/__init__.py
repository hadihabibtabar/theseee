"""Model subsystem for the RecSys23 federated-installation project.

Stage 4: single-task Transformer baseline over per-feature tokens.
Stage 5: Transformer + MMoE multi-task model (Click + Install).
Stage 7: SSL-capable variant (projection head on the CLS path).
Federated components arrive in later stages.
"""

from .transformer import (
    CategoricalIdError,
    TransformerBaseline,
    TransformerConfig,
    install_loss,
)
from .mmoe import MMoEConfig, TransformerMMoE, mmoe_loss
from .ssl_transformer_mmoe import SSLConfig, SSLTransformerMMoE
from .ssl_transformer_baseline import SSLTransformerBaseline

__all__ = [
    "TransformerBaseline",
    "TransformerConfig",
    "CategoricalIdError",
    "install_loss",
    "TransformerMMoE",
    "MMoEConfig",
    "mmoe_loss",
    "SSLTransformerMMoE",
    "SSLConfig",
    "SSLTransformerBaseline",
]
