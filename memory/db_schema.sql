CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TYPE memory_partition AS ENUM ('episodic', 'semantic', 'emotional', 'identity');
CREATE TYPE trust_tier_enum AS ENUM ('HIGH', 'MEDIUM', 'LOW', 'UNKNOWN');

CREATE TABLE IF NOT EXISTS memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    partition       memory_partition NOT NULL,
    text            TEXT NOT NULL,
    embedding       VECTOR(768),
    importance      FLOAT NOT NULL DEFAULT 0.5 CHECK (importance >= 0.0 AND importance <= 1.0),
    emo_valence     FLOAT NOT NULL DEFAULT 0.0 CHECK (emo_valence >= -1.0 AND emo_valence <= 1.0),
    emo_arousal     FLOAT NOT NULL DEFAULT 0.0 CHECK (emo_arousal >= 0.0 AND emo_arousal <= 1.0),
    emo_saliency    FLOAT GENERATED ALWAYS AS (
                        CASE WHEN ABS(emo_valence) > 0.6 AND emo_arousal > 0.5
                             THEN (ABS(emo_valence) + emo_arousal) / 2.0
                             ELSE ABS(emo_valence) * 0.3 + emo_arousal * 0.2
                        END
                    ) STORED,
    relationship    TEXT,
    event_type      TEXT,
    source          TEXT DEFAULT 'system',
    min_trust_tier  trust_tier_enum NOT NULL DEFAULT 'LOW',
    staleness       FLOAT NOT NULL DEFAULT 0.0 CHECK (staleness >= 0.0 AND staleness <= 1.0),
    confirmed       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed   TIMESTAMPTZ,
    access_count    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS memory_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id       UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    chunk_text      TEXT NOT NULL,
    chunk_embedding VECTOR(768),
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memory_graph (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    subject         TEXT NOT NULL,
    relation_type   TEXT NOT NULL,
    object          TEXT NOT NULL,
    memory_id       UUID REFERENCES memories(id) ON DELETE SET NULL,
    weight          FLOAT NOT NULL DEFAULT 1.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    query           TEXT NOT NULL,
    retrieved_ids   UUID[],
    rerank_scores   FLOAT[],
    final_score     FLOAT,
    trust_tier      trust_tier_enum,
    trust_score     FLOAT,
    route           TEXT,
    response_text   TEXT,
    nli_passed      BOOLEAN,
    used_fallback   BOOLEAN NOT NULL DEFAULT FALSE,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ablation_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name TEXT NOT NULL,
    lambda_val      FLOAT,
    alpha_val       FLOAT,
    beta_val        FLOAT,
    gamma_val       FLOAT,
    delta_val       FLOAT,
    precision_at_5  FLOAT,
    precision_at_10 FLOAT,
    faithfulness    FLOAT,
    emotional_lift  FLOAT,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_user     ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_memories_partition ON memories (partition);
CREATE INDEX IF NOT EXISTS idx_memories_user_part ON memories (user_id, partition);
CREATE INDEX IF NOT EXISTS idx_memories_trgm     ON memories USING gin (text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_memories_created  ON memories (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_parent     ON memory_chunks (parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user       ON memory_chunks (user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_trgm       ON memory_chunks USING gin (chunk_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_graph_user        ON memory_graph (user_id);
CREATE INDEX IF NOT EXISTS idx_graph_subject     ON memory_graph (user_id, subject);
CREATE INDEX IF NOT EXISTS idx_session_user      ON session_log (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_hnsw ON memories
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON memory_chunks
    USING hnsw (chunk_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
