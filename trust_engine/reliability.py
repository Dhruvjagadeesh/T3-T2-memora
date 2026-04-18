"""
trust_engine/reliability.py  [MODIFIED — v2]
--------------------------------------------
HistoricReliability (HR) component.

FIX (Issue #1 — HR Saturation):
    Raw HR was a weighted fraction in [0, 1] that quickly stabilised
    at ~0.60-1.00, causing overconfidence.  Two changes applied:

    1. Diminishing-returns via log scaling:
           HR_final = log(1 + k * raw_HR) / log(1 + k)
       with k = 3.0 (default).  Compresses high raw values while
       preserving ordering and keeping HR in [0, 1].

    2. Session-count confidence gating: HR is down-weighted when
       history is sparse (< min_sessions):
           HR_gated = HR_final * min(K / min_sessions, 1.0)

    Effect: HR contribution to trust stays below +0.08 until K >= 3
    sessions, growing smoothly thereafter. Prevents single-session
    overconfidence.
"""

from __future__ import annotations
import logging
import numpy as np
from .models import UserTrustState

logger = logging.getLogger(__name__)

MAX_SESSION_HISTORY: int = 20
_LOG_K:              float = 3.0
_MIN_SESSIONS:       int   = 3


def _compute_hr(self, state: UserTrustState) -> float:
    """
    Compute HR(u) with log-scale diminishing returns + confidence gating.

    Formula:
        raw_HR  = sum(lambda^(K-1-k) * V_k) / sum(lambda^(K-1-k))
        HR_log  = log(1 + k * raw_HR) / log(1 + k)
        HR_final = HR_log * min(K / min_sessions, 1.0)

    Returns float in [0, 1].
    """
    history = state.session_verifications
    K = len(history)

    if K == 0:
        logger.debug("HR=0.000 (no sessions)")
        return 0.0

    # Decay-weighted fraction (original formula)
    weights  = np.array([self.decay ** (K - 1 - k) for k in range(K)], dtype=float)
    verified = np.array(history, dtype=float)
    raw_hr   = float(np.dot(weights, verified) / weights.sum())

    # Diminishing returns: log transform
    hr_log = float(np.log1p(_LOG_K * raw_hr) / np.log1p(_LOG_K))

    # Confidence gating: sparse history gets down-weighted
    confidence = min(K / _MIN_SESSIONS, 1.0)
    hr_final   = hr_log * confidence

    logger.debug(
        "HR  K=%d  raw=%.3f  log=%.3f  conf=%.2f  final=%.3f",
        K, raw_hr, hr_log, confidence, hr_final,
    )
    return hr_final
