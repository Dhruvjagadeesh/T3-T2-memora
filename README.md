# Memora

**Trust-gated episodic memory for AI companions** — a dementia-care assistant that remembers the right things for the right people, at the right trust level.

Memora combines three subsystems:

| Layer | What it does |
|-------|--------------|
| **T1 — Identity** | Face + voice biometrics → identity confidence score (IC) |
| **T2 — Memory** | Emotion-aware storage, hybrid retrieval (dense + sparse + graph), trust-gated reranking |
| **T3 — Trust Engine** | Real-time trust score from IC + behaviour + emotion + session history |

This repo contains **T2** and **T3** — the parts you can run without specialised biometric hardware.

---

## Repository structure

```
memora/
├── trust_engine/               # T3: Trust Engine (v2, canonical)
│   ├── trust_engine.py         # TrustEngine class — main update loop
│   ├── models.py               # UserTrustState, TrustTier dataclasses
│   ├── behavior.py             # BC: Behaviour Consistency (adaptive EMA + volatility penalty)
│   ├── emotion.py              # ES: Emotional Signal (sigmoid + EMA)
│   ├── reliability.py          # HR: Historic Reliability (log-scale + confidence gating)
│   └── __init__.py
│
├── memory/                     # T2: Memory Pipeline
│   ├── db_schema.sql           # PostgreSQL schema (pgvector + pg_trgm)
│   ├── db_init.py              # Schema runner + seed data helper
│   ├── embedder.py             # Singleton BGE embedder + DistilRoBERTa emotion tagger
│   ├── memory_store.py         # insert_memory, chunking, partition routing
│   ├── retrieval.py            # Hybrid retrieval: dense + sparse + graph → RRF fusion
│   ├── rerank.py               # Cross-encoder reranking, CRAG filter, dedup, context builder
│   ├── memory_pipeline.py      # Top-level orchestrator: retrieval → rerank → log
│   ├── memory_writer.py        # Lifelong embedding adaptation, staleness decay, session log
│   └── __init__.py
│
├── integration/                # T2 + T3 bridge
│   ├── trust_service.py        # TrustService wrapper → T3Trust dataclass for memory pipeline
│   └── __init__.py
│
├── experiments/                # T3 evaluation
│   ├── run_experiments_v2.py   # Paper-grade ablation study (200 personas, 4 configs)
│   ├── run_dataset_experiment.py # Real-dataset experiments (EmpatheticDialogues, PersonaChat, MSC)
│   ├── data/
│   │   ├── synthetic_trust_dataset.py
│   │   ├── empathetic_loader.py
│   │   └── persona_loader.py
│   ├── evaluation/
│   │   ├── metrics.py          # TCE, TierAccuracy, SeparationScore, Stability
│   │   └── plots.py            # Publication figures
│   └── results_v2/             # Pre-computed results (CSVs + 4 PNG figures)
│       └── figures/
│
├── tests/
│   ├── test_trust_engine.py    # 30 unit tests for T3
│   └── test_t2.py              # Unit tests for T2 reranking + memory writer
│
├── run_demo.py                 # Quick T3 demo (no DB needed)
├── run_t2_t3_integration.py    # End-to-end T2+T3 demo
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quick start — Trust Engine only (no DB)

```bash
git clone https://github.com/your-org/memora.git
cd memora
pip install -r requirements.txt

# Run the trust engine demo
python run_demo.py

# Run the T2+T3 integration demo (trust engine only, no DB)
python run_t2_t3_integration.py --demo-trust-only
```

You should see 15 steps of trust evolution, including an IC drop event and an adversarial signal.

---

## Full setup — Memory pipeline with PostgreSQL

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 15+ with extensions:
  ```sql
  CREATE EXTENSION vector;    -- pgvector
  CREATE EXTENSION pg_trgm;   -- trigram similarity
  ```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your DB credentials
```

```env
MEMORA_DB_HOST=localhost
MEMORA_DB_PORT=5432
MEMORA_DB_NAME=memora
MEMORA_DB_USER=postgres
MEMORA_DB_PASSWORD=your_password
```

### 3. Initialise the database

```bash
# Create schema + indexes
python memory/db_init.py

# Create schema + seed test patient data
python memory/db_init.py --seed
```

The seed creates a test patient (`test_patient_001` — Kamala) with memories across all four partitions: episodic, semantic, emotional, and identity.

### 4. Run the full integration demo

```bash
python run_t2_t3_integration.py
```

---

## Running experiments

### Ablation study (synthetic dataset, no DB)

```bash
# Full run: 200 personas × 4 ablation configs → results_v2/ CSVs + 4 figures
python experiments/run_experiments_v2.py

# Quick mode: 20 personas, no plots
python experiments/run_experiments_v2.py --quick
```

### Real dataset experiments

```bash
# All 5 experiments (EmpatheticDialogues, PersonaChat, MSC, adversarial, TCE)
python experiments/run_dataset_experiment.py

# Single experiment
python experiments/run_dataset_experiment.py --dataset empathetic --max_convs 200

# With ablation config
python experiments/run_dataset_experiment.py --ablation no_emotion_weight
```

Pre-computed results are in `experiments/results_v2/`.

---

## Running tests

```bash
# T3 Trust Engine (27 tests, no DB required)
python -m pytest tests/test_trust_engine.py -v

# T2 Memory Pipeline (unit tests, no DB required)
python -m pytest tests/test_t2.py -v

# All tests
python -m pytest tests/ -v
```

---

## How the Trust Engine works

The `TrustEngine` maintains a per-user `UserTrustState` and updates it each turn:

```
trust(t) = trust(t-1) × decay  +  w1 × BC  +  w2 × ES  +  w3 × HR
```

| Component | Formula | What it captures |
|-----------|---------|-----------------|
| **BC** — Behaviour Consistency | Cosine similarity of behaviour vector to rolling centroid, EMA-smoothed with adaptive alpha and volatility penalty | How consistently the user behaves |
| **ES** — Emotional Signal | `sigmoid((valence+1)/2 × arousal)`, EMA-smoothed | Emotional engagement and authenticity |
| **HR** — Historic Reliability | Decay-weighted verified session fraction, log-scaled, confidence-gated | Long-term track record across sessions |

**Identity gate**: IC score checked before every update. Three consecutive IC failures reset trust to 0.10.

**Adversarial detection**: If both BC < 0.20 and ES < 0.15 in the same turn, an adversarial flag is raised. Two flags → trust reset.

**Tier hysteresis**: Trust upgrades (LOW→MEDIUM, MEDIUM→HIGH) require two consecutive ticks above the threshold. Downgrades are immediate.

### Default parameters (v2)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `decay` | 0.95 | Per-turn trust decay |
| `w1, w2, w3` | 0.55, 0.20, 0.25 | BC, ES, HR weights |
| `theta_known` | 0.65 | IC threshold — recognised user |
| `theta_new` | 0.35 | IC threshold — new/unknown |
| `max_delta` | 0.15 | Max trust change per turn |
| `alpha_bc` | 0.10 | BC EMA rate |
| `alpha_es` | 0.30 | ES EMA rate |

### Basic usage

```python
import numpy as np
from trust_engine import TrustEngine, TrustTier

engine = TrustEngine()

trust_score, tier = engine.update(
    user_id="patient_001",
    ic_score=0.87,          # from T1 biometrics
    valence=0.4,            # from emotion tagger
    arousal=0.5,
    behavior_vec=np.random.randn(10),   # from T1 behaviour features
    session_boundary=True,              # end of session
)

print(f"Trust: {trust_score:.3f}  Tier: {tier.value}")
# Trust: 0.412  Tier: MEDIUM
```

Or via the integration bridge:

```python
from integration.trust_service import TrustService

svc = TrustService()
trust = svc.evaluate(
    user_id="patient_001",
    ic_score=0.87,
    valence=0.4,
    arousal=0.5,
    behavior_vec=np.random.randn(10),
    sensitivity=2,          # memory sensitivity level (1–5)
    session_boundary=True,
)
# trust.trust_score, trust.trust_tier, trust.privacy_penalty
```

---

## How the Memory Pipeline works

```
Query + Identity + Trust
        │
        ▼
┌──────────────────┐
│  Trust gate      │  LOW trust → blocked, return fallback response
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Hybrid Retrieval (parallel)                 │
│  ├── Dense: HNSW cosine on chunk embeddings  │
│  ├── Sparse: pg_trgm trigram similarity      │
│  └── Graph: relation-aware keyword hop       │
│              ↓ Reciprocal Rank Fusion         │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Reranking                                   │
│  score = α×CE + β×recency + γ×emotion×T - δ×P │
│  CRAG filter → dedup → context budget         │
└──────┬───────────────────────────────────────┘
       │
       ▼
  Context string → LLM response
```

Memory partitions: `episodic` | `semantic` | `emotional` | `identity`

Each partition has a minimum trust tier requirement. Identity memories (emergency contacts, full name) require `HIGH` trust.

---

## Memory partition routing

| Partition | Routed when |
|-----------|-------------|
| `identity` | `event_type = "identity_record"` |
| `emotional` | `abs(valence) > 0.6` AND `arousal > 0.5` |
| `episodic` | Family visits, medical appointments, personal events |
| `semantic` | Everything else |

---

## Results summary (v2)

Pre-computed on 200 synthetic personas × 25 turns each:

| Config | TCE ↓ | Tier Acc ↑ | Separation ↑ | Stability ↓ |
|--------|-------|-----------|-------------|------------|
| Full model | **best** | **best** | **best** | **best** |
| No trust gate | higher | lower | lower | — |
| No behaviour | higher | lower | — | higher |
| No emotion | higher | lower | — | — |

See `experiments/results_v2/ablation_table.csv` for exact numbers and `experiments/results_v2/figures/` for the four publication figures.

---

## Architecture notes

- **Embedder**: `BAAI/bge-base-en-v1.5` (768-dim, HNSW-indexed in pgvector)
- **Emotion tagger**: `j-hartmann/emotion-english-distilroberta-base` → 7-class → valence/arousal mapping
- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- **DB**: PostgreSQL 15 + pgvector 0.6 + pg_trgm — all retrieval happens in SQL, no external vector DB needed

---

## Contributing

1. Fork the repo and create a feature branch
2. Run `python -m pytest tests/ -v` — all tests must pass
3. For trust engine changes, add/update tests in `tests/test_trust_engine.py`
4. Open a PR with a clear description of what changed and why

---

## License

MIT — see [LICENSE](LICENSE).
