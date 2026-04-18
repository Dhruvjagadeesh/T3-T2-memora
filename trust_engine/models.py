from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

class TrustTier(Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"

@dataclass
class UserTrustState:
    trust_score: float = 0.30
    behavior_centroid: Optional[np.ndarray] = None
    bc_ema: float = 0.50
    es_ema: float = 0.30
    session_verifications: list = field(default_factory=list)
    adversarial_flags: int = 0
    consecutive_ic_fails: int = 0
    last_tier: TrustTier = TrustTier.LOW
    tier_stable_count: int = 0