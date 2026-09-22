"""Stage 7 SSL subsystem: contrastive learning on the Transformer CLS path.

Public API::

    FeatureCorruptionAugmentation   # feature-corruption views (rate 0.15)
    nt_xent_loss / contrastive_loss # symmetric NT-Xent / InfoNCE
    ssl_joint_loss                  # alpha * supervised + (1-alpha) * contrastive
    SSLTransformerMMoE / SSLConfig  # projection head + dual-view model (models pkg)
"""

from .augmentations import (
    BINARY_NEUTRAL_VALUE,
    CATEGORICAL_UNK_INDEX,
    MISSING_NEUTRAL_VALUE,
    NUMERICAL_NEUTRAL_VALUE,
    CorruptionMasks,
    FeatureCorruptionAugmentation,
)
from .contrastive_loss import (
    DEFAULT_TEMPERATURE,
    ContrastiveLossError,
    contrastive_loss,
    l2_normalize,
    nt_xent_loss,
)
from .joint_loss import DEFAULT_ALPHA, JointLossError, ssl_joint_loss

__all__ = [
    "FeatureCorruptionAugmentation",
    "CorruptionMasks",
    "CATEGORICAL_UNK_INDEX",
    "NUMERICAL_NEUTRAL_VALUE",
    "BINARY_NEUTRAL_VALUE",
    "MISSING_NEUTRAL_VALUE",
    "nt_xent_loss",
    "contrastive_loss",
    "l2_normalize",
    "DEFAULT_TEMPERATURE",
    "ContrastiveLossError",
    "ssl_joint_loss",
    "DEFAULT_ALPHA",
    "JointLossError",
]
