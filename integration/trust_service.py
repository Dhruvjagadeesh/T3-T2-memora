"""
integration/trust_service.py
============================
Bridge between the T3 Trust Engine and the T2 Memory Pipeline.

Wraps TrustEngine so the memory pipeline can call a single function
to get a T3Trust dataclass, ready to pass directly into
MemoryPipelineInput.

Usage
-----
    from integration.trust_service import TrustService

    svc = TrustService()
    trust = svc.evaluate(
        user_id="user_1",
        ic_score=0.82,
        valence=0.4,
        arousal=0.6,
        behavior_vec=np.array([...]),
        sensitivity=2,
        session_boundary=True,
    )
    # trust.trust_score, trust.trust_tier, trust.privacy_penalty
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trust_engine import TrustEngine, TrustTier


@dataclass
class T3Trust:
    trust_score: float
    trust_tier: str          # "HIGH" | "MEDIUM" | "LOW"
    privacy_penalty: float


class TrustService:
    """
    Singleton-safe wrapper around TrustEngine.

    Parameters mirror TrustEngine.__init__; defaults match v2 paper config.
    """

    def __init__(self, **engine_kwargs):
        self._engine = TrustEngine(**engine_kwargs)

    def evaluate(
        self,
        user_id: str,
        ic_score: float,
        valence: float,
        arousal: float,
        behavior_vec: np.ndarray,
        sensitivity: int = 1,
        psi: float = 1.0,
        session_boundary: bool = False,
    ) -> T3Trust:
        """
        Run one trust update and return a T3Trust dataclass.

        Parameters
        ----------
        user_id         : unique user identifier
        ic_score        : identity confidence score in [0, 1]
        valence         : emotion valence in [-1, 1]
        arousal         : emotion arousal in [0, 1]
        behavior_vec    : behaviour embedding vector (any dimension)
        sensitivity     : memory sensitivity level (1–5), used for privacy penalty
        psi             : privacy scaling factor (default 1.0)
        session_boundary: True at the end of each session
        """
        trust_score, tier_enum = self._engine.update(
            user_id=user_id,
            ic_score=ic_score,
            valence=valence,
            arousal=arousal,
            behavior_vec=behavior_vec,
            session_boundary=session_boundary,
        )

        privacy_penalty = self._engine.privacy_penalty(
            sensitivity=sensitivity,
            trust=trust_score,
            psi=psi,
        )

        return T3Trust(
            trust_score=round(trust_score, 4),
            trust_tier=tier_enum.value,
            privacy_penalty=round(privacy_penalty, 4),
        )

    def get_trust(self, user_id: str) -> float:
        return self._engine.get_trust(user_id)

    def get_tier(self, user_id: str) -> str:
        return self._engine.get_trust_tier(user_id).value

    def retrieval_score(
        self,
        cos_sim: float,
        recency: float,
        emotion: float,
        trust: float,
        sensitivity: int,
    ) -> float:
        """Proxy to TrustEngine.retrieval_score for use in reranking."""
        return self._engine.retrieval_score(
            cos_sim=cos_sim,
            recency=recency,
            emotion=emotion,
            trust=trust,
            sensitivity=sensitivity,
        )
