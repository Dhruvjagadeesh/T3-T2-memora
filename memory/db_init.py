import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("MEMORA_DB_HOST", "localhost"),
    "port":     int(os.getenv("MEMORA_DB_PORT", "5432")),
    "dbname":   os.getenv("MEMORA_DB_NAME", "memora"),
    "user":     os.getenv("MEMORA_DB_USER", "postgres"),
    "password": os.getenv("MEMORA_DB_PASSWORD", ""),
}

SCHEMA_PATH = Path(__file__).parent / "sql" / "db_schema.sql"


def get_connection(autocommit=False):
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = autocommit
    return conn


def check_extensions(conn):
    cur = conn.cursor()
    cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');")
    found = {row[0] for row in cur.fetchall()}
    cur.close()
    missing = {"vector", "pg_trgm"} - found
    if missing:
        raise RuntimeError(
            f"Missing PostgreSQL extensions: {missing}. "
            "Run as superuser: CREATE EXTENSION vector; CREATE EXTENSION pg_trgm;"
        )


def run_schema(conn):
    sql = SCHEMA_PATH.read_text()
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()


def verify_tables(conn):
    required = {"memories", "memory_chunks", "memory_graph", "session_log", "ablation_log"}
    cur = conn.cursor()
    cur.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
    )
    found = {row[0] for row in cur.fetchall()}
    cur.close()
    missing = required - found
    if missing:
        raise RuntimeError(f"Tables not created: {missing}")
    return found & required


def verify_indexes(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname = 'public';"
    )
    indexes = {row[0] for row in cur.fetchall()}
    cur.close()
    required_indexes = {
        "idx_memories_hnsw",
        "idx_chunks_hnsw",
        "idx_memories_trgm",
        "idx_chunks_trgm",
    }
    missing = required_indexes - indexes
    if missing:
        print(f"Warning: indexes not found: {missing}. May need data before HNSW builds.")
    return indexes


def seed_test_data(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM memories WHERE user_id = 'test_patient_001';")
    if cur.fetchone()[0] > 0:
        cur.close()
        return

    test_memories = [
        {
            "user_id": "test_patient_001",
            "partition": "episodic",
            "text": "Your daughter Ananya visited you last Sunday and brought flowers.",
            "importance": 0.9,
            "emo_valence": 0.8,
            "emo_arousal": 0.6,
            "relationship": "daughter",
            "event_type": "family_visit",
            "min_trust_tier": "LOW",
        },
        {
            "user_id": "test_patient_001",
            "partition": "semantic",
            "text": "You used to work as a schoolteacher for thirty years in Bengaluru.",
            "importance": 0.7,
            "emo_valence": 0.3,
            "emo_arousal": 0.2,
            "relationship": None,
            "event_type": "personal_history",
            "min_trust_tier": "LOW",
        },
        {
            "user_id": "test_patient_001",
            "partition": "emotional",
            "text": "You became very distressed when you could not remember your son Arjun's phone number.",
            "importance": 0.8,
            "emo_valence": -0.7,
            "emo_arousal": 0.8,
            "relationship": "son",
            "event_type": "distress_event",
            "min_trust_tier": "MEDIUM",
        },
        {
            "user_id": "test_patient_001",
            "partition": "identity",
            "text": "Your name is Kamala. You live at 14 Jayanagar, Bengaluru. Your emergency contact is Ananya at 9845012345.",
            "importance": 1.0,
            "emo_valence": 0.0,
            "emo_arousal": 0.0,
            "relationship": None,
            "event_type": "identity_record",
            "min_trust_tier": "HIGH",
        },
        {
            "user_id": "test_patient_001",
            "partition": "episodic",
            "text": "You had a doctor appointment with Dr. Sharma on Monday for your blood pressure check.",
            "importance": 0.75,
            "emo_valence": -0.1,
            "emo_arousal": 0.3,
            "relationship": "doctor",
            "event_type": "medical_appointment",
            "min_trust_tier": "MEDIUM",
        },
    ]

    for m in test_memories:
        cur.execute(
            """
            INSERT INTO memories
                (user_id, partition, text, importance, emo_valence, emo_arousal,
                 relationship, event_type, min_trust_tier)
            VALUES
                (%(user_id)s, %(partition)s, %(text)s, %(importance)s, %(emo_valence)s,
                 %(emo_arousal)s, %(relationship)s, %(event_type)s, %(min_trust_tier)s)
            """,
            m
        )

    cur.execute(
        """
        INSERT INTO memory_graph (user_id, subject, relation_type, object)
        VALUES
            ('test_patient_001', 'Kamala', 'DAUGHTER', 'Ananya'),
            ('test_patient_001', 'Kamala', 'SON', 'Arjun'),
            ('test_patient_001', 'Kamala', 'DOCTOR', 'Dr. Sharma'),
            ('test_patient_001', 'Ananya', 'VISITED', 'last Sunday'),
            ('test_patient_001', 'Arjun', 'PHONE', '9845012345')
        ON CONFLICT DO NOTHING;
        """
    )

    conn.commit()
    cur.close()
    print("Test data seeded for test_patient_001.")


def init_db(seed=False):
    print(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    conn = get_connection()
    try:
        check_extensions(conn)
        print("Extensions: OK (vector, pg_trgm)")

        run_schema(conn)
        print("Schema: OK")

        tables = verify_tables(conn)
        print(f"Tables verified: {sorted(tables)}")

        indexes = verify_indexes(conn)
        print(f"Indexes found: {len(indexes)}")

        if seed:
            seed_test_data(conn)

    finally:
        conn.close()
    print("DB init complete.")


if __name__ == "__main__":
    seed_flag = "--seed" in sys.argv
    init_db(seed=seed_flag)
