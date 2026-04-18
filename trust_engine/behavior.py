"""
trust_engine/behavior.py  [MODIFIED — v2]
-----------------------------------------
BehaviorConsistency (BC) component.

FIX (Issue #2 — Inconsistent Persona Failure):
    BC EMA with alpha=0.10 was too slow: 10 turns needed to shift
    the centroid meaningfully, so rapidly oscillating users appeared
    artificially stable.

    Changes:
    1. Adaptive EMA: when BC drops sharply (delta_bc < -0.15 for
       2+ consecutive turns), alpha_bc temporarily doubles (0.20) so
       the centroid tracks the shift faster. It reverts once BC
       stabilises.

    2. Volatility penalty: a rolling short-term variance of BC
       is computed over the last N_VOL=5 turns. When variance > 0.03,
       a penalty is subtracted from the returned BC score:
           BC_final = BC_ema - lambda_vol * sqrt(variance)
           # consistency penalty signal (for trust engine use)
           # consistency_penalty = np.sqrt(self.recent_variance)
       This directly encodes the intuition: unstable users should
       score lower, not the same as a stable user with the same mean.
"""

from __future__ import annotations
import logging
import numpy as np
from .models import UserTrustState

logger = logging.getLogger(__name__)

_N_VOL:       int   = 5      # rolling window for variance penalty
_VOL_THRESH:  float = 0.03   # variance above which penalty kicks in
_LAMBDA_VOL:  float = 0.25   # penalty scale
_ADAPT_DROP:  float = 0.15   # BC drop threshold for adaptive alpha
_ADAPT_MULT:  float = 2.0    # alpha multiplier when adapting


def _compute_bc(self, state: UserTrustState,
                behavior_vec: np.ndarray) -> float:
    """
    Compute BC(u) with adaptive EMA + volatility penalty.

    Returns float in [0, 1].
    """
    if state.behavior_centroid is None:
        state.behavior_centroid = behavior_vec.copy()
        # Initialise rolling BC history
        if not hasattr(state, '_bc_history'):
            state._bc_history = []
        if not hasattr(state, '_bc_drop_streak'):
            state._bc_drop_streak = 0
        return 0.50

    # Ensure history attrs exist (for loaded states)
    if not hasattr(state, '_bc_history'):
        state._bc_history = []
    if not hasattr(state, '_bc_drop_streak'):
        state._bc_drop_streak = 0

    cos_sim = float(np.dot(behavior_vec, state.behavior_centroid) /
                    (np.linalg.norm(behavior_vec) *
                     np.linalg.norm(state.behavior_centroid) + 1e-8))
    bc_raw = (cos_sim + 1) / 2.0

    # Adaptive EMA: detect sustained drop
    delta_bc = bc_raw - state.bc_ema
    if delta_bc < -_ADAPT_DROP:
        state._bc_drop_streak += 1
    else:
        state._bc_drop_streak = 0

    alpha_eff = self.alpha_bc * _ADAPT_MULT if state._bc_drop_streak >= 2 else self.alpha_bc

    # Update EMA
    state.bc_ema = (1 - alpha_eff) * state.bc_ema + alpha_eff * bc_raw

    # Update centroid (use base alpha, not adaptive, to keep centroid stable)
    state.behavior_centroid = (
        (1 - self.alpha_bc) * state.behavior_centroid
        + self.alpha_bc * behavior_vec
    )

    # Rolling BC history for volatility
    state._bc_history.append(state.bc_ema)
    if len(state._bc_history) > _N_VOL:
        state._bc_history.pop(0)

    # Volatility penalty
    vol_penalty = 0.0
    if len(state._bc_history) >= 3:
        variance = float(np.var(state._bc_history))
        if variance > _VOL_THRESH:
            vol_penalty = _LAMBDA_VOL * np.sqrt(variance)

    bc_final = float(np.clip(state.bc_ema - vol_penalty, 0.0, 1.0))

    logger.debug(
        "BC  cos=%.3f  raw=%.3f  ema=%.3f  alpha=%.2f  vol_pen=%.3f  final=%.3f",
        cos_sim, bc_raw, state.bc_ema, alpha_eff, vol_penalty, bc_final,
    )
    return bc_final
