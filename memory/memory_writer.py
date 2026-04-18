import math
import json
from datetime import datetime, timezone
from typing import Optional
from db_init import get_connection
from embedder import get_embedder
from memory_store import MemoryInput, insert_memory

LIFELONG_ADAPT_RATE = 0.01
ADAPT_CONFIDENCE_THRESHOLD = 0.80
STALENESS_TAU_DAYS = 90.0
STALENESS_MAX = 0.95


def update_embedding_lifelong(memory_id: str, new_text: str, confidence: float):
    if confidence < ADAPT_CONFIDENCE_THRESHOLD:
        return
    embedder = get_embedder()
    new_vec  = embedder.encode(new_text).vector

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT embedding FROM memories WHERE id = %s",
            (memory_id,),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            cur.close()
            return

        import numpy as np
        raw = row[0]
        if isinstance(raw, str):
            stored_vec = np.array([float(x) for x in raw.strip("[]").split(",")], dtype=np.float32)
        else:
            stored_vec = np.array(raw, dtype=np.float32)

        updated = (1.0 - LIFELONG_ADAPT_RATE) * stored_vec + LIFELONG_ADAPT_RATE * new_vec
        norm = float(np.linalg.norm(updated))
        if norm > 1e-9:
            updated = updated / norm

        vec_str = "[" + ",".join(f"{v:.6f}" for v in updated.tolist()) + "]"
        cur.execute(
            "UPDATE memories SET embedding = %s::vector WHERE id = %s",
            (vec_str, memory_id),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def decay_staleness(user_id: str):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE memories
            SET staleness = LEAST(
                %s,
                1.0 - EXP(
                    -EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 / %s
                )
            )
            WHERE user_id = %s
            """,
            (STALENESS_MAX, STALENESS_TAU_DAYS, user_id),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def mark_confirmed(memory_id: str):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE memories SET confirmed = TRUE, staleness = staleness * 0.5 WHERE id = %s",
            (memory_id,),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def update_access_stats(memory_ids: list[str]):
    if not memory_ids:
        return
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE memories
            SET last_accessed = NOW(),
                access_count  = access_count + 1
            WHERE id = ANY(%s)
            """,
            (memory_ids,),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def log_session(
    user_id:       str,
    query:         str,
    retrieved_ids: list[str],
    rerank_scores: list[float],
    final_score:   float,
    trust_tier:    str,
    trust_score:   float,
    route:         str,
    response_text: Optional[str],
    nli_passed:    Optional[bool],
    used_fallback: bool,
    latency_ms:    Optional[int],
):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO session_log
                (user_id, query, retrieved_ids, rerank_scores, final_score,
                 trust_tier, trust_score, route, response_text, nli_passed,
                 used_fallback, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                query,
                retrieved_ids,
                rerank_scores,
                final_score,
                trust_tier,
                trust_score,
                route,
                response_text,
                nli_passed,
                used_fallback,
                latency_ms,
            ),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def log_ablation(
    experiment_name: str,
    lambda_val:      Optional[float] = None,
    alpha_val:       Optional[float] = None,
    beta_val:        Optional[float] = None,
    gamma_val:       Optional[float] = None,
    delta_val:       Optional[float] = None,
    precision_at_5:  Optional[float] = None,
    precision_at_10: Optional[float] = None,
    faithfulness:    Optional[float] = None,
    emotional_lift:  Optional[float] = None,
    notes:           Optional[str]   = None,
):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ablation_log
                (experiment_name, lambda_val, alpha_val, beta_val, gamma_val, delta_val,
                 precision_at_5, precision_at_10, faithfulness, emotional_lift, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                experiment_name,
                lambda_val, alpha_val, beta_val, gamma_val, delta_val,
                precision_at_5, precision_at_10, faithfulness, emotional_lift,
                notes,
            ),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def write_back_new_memory(
    user_id:    str,
    text:       str,
    event_type: Optional[str] = None,
    importance: float = 0.5,
    source:     str = "interaction",
) -> str:
    result = insert_memory(
        MemoryInput(
            user_id=user_id,
            text=text,
            event_type=event_type,
            importance=importance,
            source=source,
        )
    )
    return result.memory_id
