import time
from dataclasses import dataclass
from typing import Optional
from retrieval import retrieve, RetrievalInput
from rerank import score_and_rerank, ScoringWeights, TrustContext, RerankResult
from memory_writer import update_access_stats, log_session

FALLBACK_RESPONSE = "I am here with you. Please take your time."


@dataclass
class T1Identity:
    user_id:          str
    IC_score:         float
    face_confidence:  float
    voice_confidence: float


@dataclass
class T3Trust:
    trust_score:     float
    trust_tier:      str
    privacy_penalty: float


@dataclass
class MemoryPipelineInput:
    query:        str
    identity:     T1Identity
    trust:        T3Trust
    hyde_text:    Optional[str] = None
    weights:      Optional[ScoringWeights] = None
    top_n:        int = 8


@dataclass
class MemoryPipelineOutput:
    context_string:  str
    memories:        list
    top_score:       float
    passed_crag:     bool
    used_fallback:   bool
    latency_ms:      int
    route:           str


def run_memory_pipeline(inp: MemoryPipelineInput) -> MemoryPipelineOutput:
    t0 = time.monotonic()

    if inp.trust.trust_tier == "LOW" or inp.trust.trust_tier == "UNKNOWN":
        elapsed = int((time.monotonic() - t0) * 1000)
        log_session(
            user_id=inp.identity.user_id,
            query=inp.query,
            retrieved_ids=[],
            rerank_scores=[],
            final_score=0.0,
            trust_tier=inp.trust.trust_tier,
            trust_score=inp.trust.trust_score,
            route="blocked_low_trust",
            response_text=FALLBACK_RESPONSE,
            nli_passed=None,
            used_fallback=True,
            latency_ms=elapsed,
        )
        return MemoryPipelineOutput(
            context_string="",
            memories=[],
            top_score=0.0,
            passed_crag=False,
            used_fallback=True,
            latency_ms=elapsed,
            route="blocked_low_trust",
        )

    retrieval_inp = RetrievalInput(
        query=inp.query,
        user_id=inp.identity.user_id,
        trust_tier=inp.trust.trust_tier,
        hyde_text=inp.hyde_text,
        top_k_post_rrf=40,
    )
    candidates = retrieve(retrieval_inp)

    trust_ctx = TrustContext(
        trust_score=inp.trust.trust_score,
        trust_tier=inp.trust.trust_tier,
        privacy_penalty=inp.trust.privacy_penalty,
    )
    weights = inp.weights or ScoringWeights()

    result: RerankResult = score_and_rerank(
        query=inp.query,
        candidates=candidates,
        trust_ctx=trust_ctx,
        weights=weights,
        top_n=inp.top_n,
    )

    elapsed = int((time.monotonic() - t0) * 1000)
    route = "retrieval" if result.passed_crag else "crag_fallback"

    retrieved_ids   = [m.memory_id    for m in result.memories]
    rerank_scores   = [m.rerank_score for m in result.memories]

    update_access_stats(retrieved_ids)

    log_session(
        user_id=inp.identity.user_id,
        query=inp.query,
        retrieved_ids=retrieved_ids,
        rerank_scores=rerank_scores,
        final_score=result.top_score,
        trust_tier=inp.trust.trust_tier,
        trust_score=inp.trust.trust_score,
        route=route,
        response_text=None,
        nli_passed=None,
        used_fallback=not result.passed_crag,
        latency_ms=elapsed,
    )

    return MemoryPipelineOutput(
        context_string=result.context_string,
        memories=result.memories,
        top_score=result.top_score,
        passed_crag=result.passed_crag,
        used_fallback=not result.passed_crag,
        latency_ms=elapsed,
        route=route,
    )
