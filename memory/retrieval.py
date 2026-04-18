import re
import math
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional
from db_init import get_connection
from embedder import get_embedder

DENSE_TOP_K   = 30
SPARSE_TOP_K  = 20
GRAPH_MAX_HOP = 1
RRF_K         = 60
POST_RRF_TOP  = 40

TRUST_TIER_MIN = {
    "HIGH":    ("LOW", "MEDIUM", "HIGH"),
    "MEDIUM":  ("LOW", "MEDIUM"),
    "LOW":     ("LOW",),
    "UNKNOWN": ("LOW",),
}


@dataclass
class RawCandidate:
    memory_id:   str
    chunk_id:    Optional[str]
    text:        str
    partition:   str
    importance:  float
    emo_saliency: float
    emo_valence: float
    emo_arousal: float
    created_at:  object
    source:      str
    dense_rank:  Optional[int] = None
    sparse_rank: Optional[int] = None
    graph_rank:  Optional[int] = None
    rrf_score:   float = 0.0


@dataclass
class RetrievalInput:
    query:          str
    user_id:        str
    trust_tier:     str = "LOW"
    hyde_text:      Optional[str] = None
    top_k_post_rrf: int = POST_RRF_TOP


def _tier_filter_sql(trust_tier: str) -> str:
    tiers = TRUST_TIER_MIN.get(trust_tier, ("LOW",))
    quoted = ", ".join(f"'{t}'" for t in tiers)
    return f"min_trust_tier IN ({quoted})"


def _dense_retrieve(
    query_vec: list[float],
    user_id: str,
    trust_tier: str,
    top_k: int = DENSE_TOP_K,
) -> list[dict]:
    tier_filter = _tier_filter_sql(trust_tier)
    vec_str = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                mc.id          AS chunk_id,
                mc.parent_id   AS memory_id,
                mc.chunk_text  AS text,
                m.partition,
                m.importance,
                m.emo_saliency,
                m.emo_valence,
                m.emo_arousal,
                m.created_at,
                m.source,
                1 - (mc.chunk_embedding <=> %s::vector) AS cos_sim
            FROM memory_chunks mc
            JOIN memories m ON mc.parent_id = m.id
            WHERE mc.user_id = %s
              AND m.{tier_filter}
              AND m.staleness < 0.9
            ORDER BY mc.chunk_embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (vec_str, user_id, vec_str, top_k),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    cols = ["chunk_id", "memory_id", "text", "partition", "importance",
            "emo_saliency", "emo_valence", "emo_arousal", "created_at", "source", "cos_sim"]
    return [dict(zip(cols, r)) for r in rows]


def _sparse_retrieve(
    query: str,
    user_id: str,
    trust_tier: str,
    top_k: int = SPARSE_TOP_K,
) -> list[dict]:
    tier_filter = _tier_filter_sql(trust_tier)
    tokens = " | ".join(re.findall(r'\b\w{3,}\b', query.lower()))
    if not tokens:
        return []
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                NULL          AS chunk_id,
                m.id          AS memory_id,
                m.text,
                m.partition,
                m.importance,
                m.emo_saliency,
                m.emo_valence,
                m.emo_arousal,
                m.created_at,
                m.source,
                similarity(m.text, %s) AS trgm_sim
            FROM memories m
            WHERE m.user_id = %s
              AND m.{tier_filter}
              AND m.staleness < 0.9
              AND similarity(m.text, %s) > 0.08
            ORDER BY trgm_sim DESC
            LIMIT %s
            """,
            (query, user_id, query, top_k),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    cols = ["chunk_id", "memory_id", "text", "partition", "importance",
            "emo_saliency", "emo_valence", "emo_arousal", "created_at", "source", "trgm_sim"]
    return [dict(zip(cols, r)) for r in rows]


def _graph_retrieve(
    query: str,
    user_id: str,
    trust_tier: str,
) -> list[dict]:
    words = set(re.findall(r'\b[A-Z][a-z]+\b', query))
    words |= set(re.findall(r'\b(?:daughter|son|wife|husband|mother|father|doctor|nurse|friend)\b', query.lower()))
    if not words:
        return []

    tier_filter = _tier_filter_sql(trust_tier)
    placeholders = ", ".join(["%s"] * len(words))
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT DISTINCT
                NULL          AS chunk_id,
                m.id          AS memory_id,
                m.text,
                m.partition,
                m.importance,
                m.emo_saliency,
                m.emo_valence,
                m.emo_arousal,
                m.created_at,
                m.source,
                1.0           AS graph_score
            FROM memory_graph g
            JOIN memories m ON g.memory_id = m.id
            WHERE g.user_id = %s
              AND m.{tier_filter}
              AND m.staleness < 0.9
              AND (
                  g.subject ILIKE ANY(ARRAY[{placeholders}])
                  OR g.object ILIKE ANY(ARRAY[{placeholders}])
                  OR g.relation_type ILIKE ANY(ARRAY[{placeholders}])
              )
            LIMIT 15
            """,
            (user_id, *words, *words, *words),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    cols = ["chunk_id", "memory_id", "text", "partition", "importance",
            "emo_saliency", "emo_valence", "emo_arousal", "created_at", "source", "graph_score"]
    return [dict(zip(cols, r)) for r in rows]


def _reciprocal_rank_fusion(
    dense:  list[dict],
    sparse: list[dict],
    graph:  list[dict],
    k:      int = RRF_K,
) -> list[RawCandidate]:
    scores: dict[str, dict] = {}

    def add(results, rank_key):
        for rank, row in enumerate(results, start=1):
            mid = row["memory_id"]
            if mid not in scores:
                scores[mid] = {
                    "row": row,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "graph_rank": None,
                    "rrf": 0.0,
                }
            scores[mid][rank_key] = rank
            scores[mid]["rrf"] += 1.0 / (k + rank)

    add(dense,  "dense_rank")
    add(sparse, "sparse_rank")
    add(graph,  "graph_rank")

    seen_ids = set()
    candidates = []
    for mid, data in sorted(scores.items(), key=lambda x: -x[1]["rrf"]):
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        r = data["row"]
        candidates.append(
            RawCandidate(
                memory_id=mid,
                chunk_id=r.get("chunk_id"),
                text=r["text"],
                partition=r["partition"],
                importance=r["importance"],
                emo_saliency=r["emo_saliency"],
                emo_valence=r["emo_valence"],
                emo_arousal=r["emo_arousal"],
                created_at=r["created_at"],
                source=r["source"],
                dense_rank=data["dense_rank"],
                sparse_rank=data["sparse_rank"],
                graph_rank=data["graph_rank"],
                rrf_score=data["rrf"],
            )
        )
    return candidates


def _fetch_parent(memory_id: str, user_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, text, partition, importance, emo_saliency,
                   emo_valence, emo_arousal, created_at, source
            FROM memories
            WHERE id = %s AND user_id = %s
            """,
            (memory_id, user_id),
        )
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        cols = ["id", "text", "partition", "importance", "emo_saliency",
                "emo_valence", "emo_arousal", "created_at", "source"]
        return dict(zip(cols, row))
    finally:
        conn.close()


def retrieve(inp: RetrievalInput) -> list[RawCandidate]:
    embedder = get_embedder()

    query_text = inp.hyde_text if inp.hyde_text else inp.query
    query_vec  = embedder.encode(query_text).vector.tolist()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f_dense  = ex.submit(_dense_retrieve,  query_vec, inp.user_id, inp.trust_tier)
        f_sparse = ex.submit(_sparse_retrieve, inp.query, inp.user_id, inp.trust_tier)
        f_graph  = ex.submit(_graph_retrieve,  inp.query, inp.user_id, inp.trust_tier)
        dense  = f_dense.result()
        sparse = f_sparse.result()
        graph  = f_graph.result()

    candidates = _reciprocal_rank_fusion(dense, sparse, graph)
    candidates = candidates[: inp.top_k_post_rrf]

    upgraded = []
    for c in candidates:
        parent = _fetch_parent(c.memory_id, inp.user_id)
        if parent:
            c.text = parent["text"]
        upgraded.append(c)

    return upgraded
