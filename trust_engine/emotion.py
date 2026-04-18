"""
trust_engine/emotion.py  [MODIFIED — v2]
-----------------------------------------
EmotionalSignal (ES) component.
Logic unchanged; print replaced with logger.
"""

from __future__ import annotations
import logging
import numpy as np
from .models import UserTrustState

logger = logging.getLogger(__name__)


def _compute_es(self, state: UserTrustState,
                valence: float, arousal: float) -> float:
    """
    ES(u,t) = sigmoid((valence+1)/2 * arousal), EMA-smoothed.
    Returns float in (0, 1).
    """
    raw = ((valence + 1) / 2) * arousal
    es  = float(1 / (1 + np.exp(-raw)))
    state.es_ema = (1 - self.alpha_es) * state.es_ema + self.alpha_es * es
    logger.debug("ES  valence=%.2f  arousal=%.2f  es_raw=%.3f  ema=%.3f",
                 valence, arousal, es, state.es_ema)
    return state.es_ema
