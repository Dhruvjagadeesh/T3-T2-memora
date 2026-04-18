# trust_engine/trust_engine.py  [v2 — fixed adversarial detection]

import numpy as np
from typing import Optional, Tuple, Dict
import logging

from .models import UserTrustState, TrustTier
from .behavior import _compute_bc
from .emotion  import _compute_es
from .reliability import _compute_hr

logger = logging.getLogger(__name__)


class TrustEngine:
    def __init__(self,
                 decay:       float = 0.95,
                 w1:          float = 0.55,
                 w2:          float = 0.20,
                 w3:          float = 0.25,
                 alpha_bc:    float = 0.10,
                 alpha_es:    float = 0.30,
                 theta_known: float = 0.65,
                 theta_new:   float = 0.35,
                 max_delta:   float = 0.15,
                 disable_bc:       bool = False,
                 disable_es:       bool = False,
                 disable_hr:       bool = False,
                 disable_ic_gate:  bool = False):

        self.decay       = decay
        self.w           = (w1, w2, w3)
        self.alpha_bc    = alpha_bc
        self.alpha_es    = alpha_es
        self.theta_known = theta_known
        self.theta_new   = theta_new
        self.max_delta   = max_delta

        self.disable_bc      = disable_bc
        self.disable_es      = disable_es
        self.disable_hr      = disable_hr
        self.disable_ic_gate = disable_ic_gate

        self._store: Dict[str, UserTrustState] = {}

    def _get_state(self, user_id: str) -> UserTrustState:
        if user_id not in self._store:
            self._store[user_id] = UserTrustState()
            logger.info("Cold start: new user %s, T=0.30", user_id)
        return self._store[user_id]

    def _score_to_tier(self, score: float) -> TrustTier:
        if score >= 0.80: return TrustTier.HIGH
        if score >= 0.50: return TrustTier.MEDIUM
        return TrustTier.LOW

    def update(self, user_id: str, ic_score: float,
               valence: float, arousal: float,
               behavior_vec: np.ndarray,
               session_boundary: bool = False) -> Tuple[float, TrustTier]:

        state  = self._get_state(user_id)
        t_prev = state.trust_score

        # ── Identity gate ────────────────────────────────────────────────
        if not self.disable_ic_gate:
            if ic_score < self.theta_new:
                state.consecutive_ic_fails += 1
                if state.consecutive_ic_fails >= 3:
                    self._reset_trust(state, user_id, reason="ic_fail_streak")
                else:
                    state.trust_score = np.clip(t_prev * 0.50, 0, 1)
                return state.trust_score, self._score_to_tier(state.trust_score)
            elif ic_score < self.theta_known:
                state.trust_score = np.clip(t_prev * 0.80, 0, 1)
                return state.trust_score, self._score_to_tier(state.trust_score)
            else:
                state.consecutive_ic_fails = 0

        # ── Compute components ───────────────────────────────────────────
        bc = 0.50 if self.disable_bc else _compute_bc(self, state, behavior_vec)
        es = 0.50 if self.disable_es else _compute_es(self, state, valence, arousal)
        hr = 0.00 if self.disable_hr else _compute_hr(self, state)

        logger.debug("Components  BC=%.3f  ES=%.3f  HR=%.3f", bc, es, hr)

        # ── Adversarial detection ────────────────────────────────────────
        # Use instantaneous (pre-EMA) signals so a single adversarial turn
        # is detectable before the slow EMA catches up.
        raw_es_instant = ((valence + 1) / 2) * arousal

        # Instantaneous cosine BC (pre-EMA, against current centroid)
        centroid = state.behavior_centroid
        if centroid is not None and not self.disable_bc:
            cos_inst = float(np.dot(behavior_vec, centroid) /
                             (np.linalg.norm(behavior_vec) *
                              np.linalg.norm(centroid) + 1e-8))
            bc_inst = (cos_inst + 1) / 2.0
        else:
            bc_inst = 0.50

        if bc_inst < 0.20 and raw_es_instant < 0.30:
            state.adversarial_flags += 1
            logger.warning("Adversarial signal for %s (flags=%d)",
                           user_id, state.adversarial_flags)
            if state.adversarial_flags >= 2:
                self._reset_trust(state, user_id, reason="adversarial")
                return state.trust_score, TrustTier.LOW
            state.trust_score = np.clip(t_prev * 0.30, 0, 1)
            return state.trust_score, TrustTier.LOW

        # ── Trust update ─────────────────────────────────────────────────
        w1, w2, w3 = self.w
        t_raw = t_prev * self.decay + w1 * bc + w2 * es + w3 * hr

        delta             = np.clip(t_raw - t_prev, -self.max_delta, self.max_delta)
        state.trust_score = np.clip(t_prev + delta, 0, 1)

        # ── Hysteretic tier assignment ───────────────────────────────────
        new_tier   = self._score_to_tier(state.trust_score)
        tier_order = {TrustTier.LOW: 0, TrustTier.MEDIUM: 1, TrustTier.HIGH: 2}

        if new_tier != state.last_tier:
            is_upgrade = tier_order[new_tier] > tier_order[state.last_tier]
            if is_upgrade:
                state.tier_stable_count += 1
                if state.tier_stable_count >= 2:
                    state.last_tier         = new_tier
                    state.tier_stable_count = 0
            else:
                state.last_tier         = new_tier
                state.tier_stable_count = 0
        else:
            state.tier_stable_count = 0

        # ── Session boundary ─────────────────────────────────────────────
        if session_boundary:
            verified = ic_score >= self.theta_known and bc >= 0.30
            state.session_verifications.append(int(verified))
            if len(state.session_verifications) > 20:
                state.session_verifications.pop(0)
            state.adversarial_flags = 0

        return state.trust_score, state.last_tier

    def get_trust(self, user_id: str) -> float:
        return self._get_state(user_id).trust_score

    def get_trust_tier(self, user_id: str) -> TrustTier:
        return self._get_state(user_id).last_tier

    def privacy_penalty(self, sensitivity: int, trust: float,
                        psi: float = 1.0) -> float:
        return sensitivity * (1 - trust) * psi

    def retrieval_score(self, cos_sim: float, recency: float,
                        emotion: float, trust: float,
                        sensitivity: int,
                        a=0.40, b=0.25, g=0.20, d=0.15) -> float:
        penalty = self.privacy_penalty(sensitivity, trust)
        return a * cos_sim + b * recency + g * emotion * trust - d * penalty

    def _reset_trust(self, state: UserTrustState, user_id: str, reason: str):
        logger.warning("Trust RESET for %s, reason: %s", user_id, reason)
        state.trust_score       = 0.10
        state.behavior_centroid = None
        state.bc_ema            = 0.50
        state.adversarial_flags = 0
        state.last_tier         = TrustTier.LOW
