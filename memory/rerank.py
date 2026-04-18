import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from sentence_transformers import CrossEncoder
from retrieval import RawCandidate

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_THRESHOLD       = 0.4
CRAG_MIN_SCORE      = 0.30
DEDUP_COS_THRESHOLD = 0.90
CONTEXT_TOKEN_BUDGET = 512
AVG_CHARS_PER_TOKEN  = 4

ALPHA_DEFAULT = 0.60
BETA_DEFAULT  = 0.25
GAMMA_DEFAULT = 0.10
DELTA_DEFAULT = 0.05

LAMBDA_PRESETS = {
    "fast":   0.05,
    "medium": 0.02,
    "slow":   0.005,
}
LAMBDA_DEFAULT = "medium"


@dataclass
class ScoringWeights:
    alpha: float = ALPHA_DEFAULT
    beta:  float = BETA_DEFAULT
    gamma: float = GAMMA_DEFAULT
    delta: float = DELTA_DEFAULT
    lambda_key: str = LAMBDA_DEFAULT

    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma + self.delta
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"alpha+beta+gamma+delta must sum to 1.0, got {total:.3f}")


@dataclass
class TrustContext:
    trust_score:     float = 1.0
    trust_tier:      str   = "HIGH"
    privacy_penalty: float = 0.0


@dataclass
class ScoredMemory:
    memory_id:     str
    text:          str
    partition:     str
    importance:    float
    emo_saliency:  float
    emo_valence:   float
    emo_arousal:   float
    created_at:    object
    rerank_score:  float
    recency_score: float
    final_score:   float
    rrf_score:     float


@dataclass
class RerankResult:
    memories:       list[ScoredMemory]
    context_string: str
    passed_crag:    bool
    top_score:      float


_cross_encoder = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _recency_score(created_at, lambda_val: float) -> float:
    if created_at is None:
        return 0.5
    now = datetime.now(timezone.utc)
    if hasattr(created_at, "tzinfo") and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta_days = (now - created_at).total_seconds() / 86400.0
    return math.exp(-lambda_val * delta_days)


def _normalize(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _dedup(memories: list[ScoredMemory]) -> list[ScoredMemory]:
    from embedder import get_embedder
    if len(memories) <= 1:
        return memories
    embedder = get_embedder()
    vecs = [embedder.encode(m.text).vector for m in memories]
    kept = [0]
    for i in range(1, len(memories)):
        duplicate = False
        for j in kept:
            if _cosine_sim(vecs[i], vecs[j]) > DEDUP_COS_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return [memories[i] for i in kept]


def score_and_rerank(
    query:       str,
    candidates:  list[RawCandidate],
    trust_ctx:   TrustContext,
    weights:     ScoringWeights = None,
    top_n:       int = 8,
) -> RerankResult:
    if weights is None:
        weights = ScoringWeights()

    if not candidates:
        return RerankResult(
            memories=[],
            context_string="",
            passed_crag=False,
            top_score=0.0,
        )

    ce = _get_cross_encoder()
    pairs = [(query, c.text) for c in candidates]
    raw_scores = ce.predict(pairs).tolist()

    lambda_val = LAMBDA_PRESETS.get(weights.lambda_key, LAMBDA_PRESETS["medium"])
    recency_raw = [_recency_score(c.created_at, lambda_val) for c in candidates]
    emo_raw     = [c.emo_saliency for c in candidates]
    imp_raw     = [c.importance   for c in candidates]

    rerank_norm  = _normalize(raw_scores)
    recency_norm = _normalize(recency_raw)
    emo_norm     = _normalize(emo_raw)
    imp_norm     = _normalize(imp_raw)

    T = max(0.0, min(1.0, trust_ctx.trust_score))
    P = max(0.0, min(1.0, trust_ctx.privacy_penalty))

    final_scores = []
    for i in range(len(candidates)):
        s = (
            weights.alpha * rerank_norm[i]
            + weights.beta  * recency_norm[i]
            + weights.gamma * emo_norm[i] * T
            - weights.delta * P
        )
        final_scores.append(s)

    scored = []
    for i, c in enumerate(candidates):
        scored.append(
            ScoredMemory(
                memory_id=c.memory_id,
                text=c.text,
                partition=c.partition,
                importance=c.importance,
                emo_saliency=c.emo_saliency,
                emo_valence=c.emo_valence,
                emo_arousal=c.emo_arousal,
                created_at=c.created_at,
                rerank_score=raw_scores[i],
                recency_score=recency_raw[i],
                final_score=final_scores[i],
                rrf_score=c.rrf_score,
            )
        )

    scored.sort(key=lambda x: -x.final_score)
    top_score = scored[0].final_score if scored else 0.0

    passed_crag = top_score >= CRAG_MIN_SCORE
    if not passed_crag:
        return RerankResult(
            memories=[],
            context_string="",
            passed_crag=False,
            top_score=top_score,
        )

    top_scored = scored[:top_n * 2]
    deduped    = _dedup(top_scored)
    final      = deduped[:top_n]

    context = _build_context(final)
    return RerankResult(
        memories=final,
        context_string=context,
        passed_crag=True,
        top_score=top_score,
    )


def _build_context(memories: list[ScoredMemory]) -> str:
    budget_chars = CONTEXT_TOKEN_BUDGET * AVG_CHARS_PER_TOKEN
    blocks = []
    used = 0
    for m in memories:
        block = f"[{m.partition.upper()}] {m.text}"
        if used + len(block) > budget_chars:
            break
        blocks.append(block)
        used += len(block) + 1
    return "\n".join(blocks)
