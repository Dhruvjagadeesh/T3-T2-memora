import uuid
import re
from dataclasses import dataclass, field
from typing import Optional
from db_init import get_connection
from embedder import get_embedder, get_emotion_tagger, EmotionResult

EMOTIONAL_PARTITION_THRESHOLD_VALENCE = 0.6
EMOTIONAL_PARTITION_THRESHOLD_AROUSAL = 0.5
CHUNK_MAX_CHARS = 256
CHUNK_OVERLAP   = 40


@dataclass
class MemoryInput:
    user_id:        str
    text:           str
    partition:      Optional[str] = None
    importance:     float = 0.5
    relationship:   Optional[str] = None
    event_type:     Optional[str] = None
    source:         str = "system"
    min_trust_tier: str = "LOW"
    emo_valence:    Optional[float] = None
    emo_arousal:    Optional[float] = None


@dataclass
class MemoryInsertResult:
    memory_id:   str
    chunk_ids:   list[str]
    partition:   str
    emo_valence: float
    emo_arousal: float
    emo_saliency: float
    emo_label:   str


def _route_partition(
    requested: Optional[str],
    emo_valence: float,
    emo_arousal: float,
    event_type: Optional[str],
) -> str:
    if requested and requested in ("episodic", "semantic", "emotional", "identity"):
        return requested
    if event_type in ("identity_record",):
        return "identity"
    if (
        abs(emo_valence) > EMOTIONAL_PARTITION_THRESHOLD_VALENCE
        and emo_arousal > EMOTIONAL_PARTITION_THRESHOLD_AROUSAL
    ):
        return "emotional"
    if event_type in ("family_visit", "medical_appointment", "personal_event"):
        return "episodic"
    return "semantic"


def _split_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    if len(chunks) <= 1:
        return chunks if chunks else [text[:max_chars]]
    with_overlap = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else chunks[i - 1]
        with_overlap.append((tail + " " + chunks[i]).strip())
    return with_overlap


def insert_memory(inp: MemoryInput) -> MemoryInsertResult:
    embedder = get_embedder()
    tagger   = get_emotion_tagger()

    if inp.emo_valence is not None and inp.emo_arousal is not None:
        emo = EmotionResult(
            label="provided",
            score=1.0,
            valence=inp.emo_valence,
            arousal=inp.emo_arousal,
            saliency=tagger._compute_saliency(inp.emo_valence, inp.emo_arousal),
        )
    else:
        emo = tagger.tag(inp.text)

    partition = _route_partition(inp.partition, emo.valence, emo.arousal, inp.event_type)

    emb = embedder.encode(inp.text)
    vec_str = "[" + ",".join(f"{v:.6f}" for v in emb.vector) + "]"

    memory_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO memories
                (id, user_id, partition, text, embedding, importance,
                 emo_valence, emo_arousal, relationship, event_type,
                 source, min_trust_tier)
            VALUES
                (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                memory_id,
                inp.user_id,
                partition,
                inp.text,
                vec_str,
                inp.importance,
                emo.valence,
                emo.arousal,
                inp.relationship,
                inp.event_type,
                inp.source,
                inp.min_trust_tier,
            ),
        )

        chunks = _split_chunks(inp.text)
        chunk_embs = embedder.encode_batch(chunks)
        chunk_ids = []
        for idx, (chunk_text, chunk_emb) in enumerate(zip(chunks, chunk_embs)):
            c_vec = "[" + ",".join(f"{v:.6f}" for v in chunk_emb.vector) + "]"
            cid = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO memory_chunks
                    (id, parent_id, user_id, chunk_text, chunk_embedding, chunk_index)
                VALUES (%s, %s, %s, %s, %s::vector, %s)
                """,
                (cid, memory_id, inp.user_id, chunk_text, c_vec, idx),
            )
            chunk_ids.append(cid)

        conn.commit()
        cur.close()
    finally:
        conn.close()

    saliency = tagger._compute_saliency(emo.valence, emo.arousal)
    return MemoryInsertResult(
        memory_id=memory_id,
        chunk_ids=chunk_ids,
        partition=partition,
        emo_valence=emo.valence,
        emo_arousal=emo.arousal,
        emo_saliency=saliency,
        emo_label=emo.label,
    )


def insert_relation(user_id: str, subject: str, relation_type: str, obj: str, memory_id: Optional[str] = None):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO memory_graph (user_id, subject, relation_type, object, memory_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, subject, relation_type.upper(), obj, memory_id),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def batch_insert(memories: list[MemoryInput]) -> list[MemoryInsertResult]:
    return [insert_memory(m) for m in memories]


def get_memory_by_id(memory_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, user_id, partition, text, importance, emo_valence, emo_arousal, "
            "emo_saliency, relationship, event_type, min_trust_tier, staleness, "
            "confirmed, created_at FROM memories WHERE id = %s",
            (memory_id,),
        )
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        cols = [
            "id", "user_id", "partition", "text", "importance",
            "emo_valence", "emo_arousal", "emo_saliency", "relationship",
            "event_type", "min_trust_tier", "staleness", "confirmed", "created_at",
        ]
        return dict(zip(cols, row))
    finally:
        conn.close()
