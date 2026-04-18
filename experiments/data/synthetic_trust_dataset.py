"""
data/synthetic_trust_dataset.py
================================
Generates synthetic user personas with scripted trust trajectories and
ground-truth T_human labels for Trust Calibration Error (TCE) evaluation.

Persona archetypes (5 total)
-----------------------------
  COOPERATIVE_STABLE  — consistent BC, positive valence/arousal, stable IC
  COOPERATIVE_GROWING — starts low, ramps up over sessions
  ADVERSARIAL         — low IC, hostile affect, incoherent behavior vecs
  INCONSISTENT        — oscillates between cooperative and hostile signals
  NEW_THEN_DROPS      — starts legitimate, IC collapses mid-session

Usage
-----
    ds       = SyntheticTrustDataset(n_variants_per_archetype=40, seed=42)
    personas = ds.generate()

    for p in personas:
        for turn in p.turns:
            engine.update(p.user_id, turn.ic_score, turn.valence,
                          turn.arousal, turn.behavior_vec,
                          turn.session_boundary)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class TurnData:
    """One processed turn — matches run_experiments_v2.py field access."""
    ic_score:         float
    valence:          float
    arousal:          float
    behavior_vec:     np.ndarray   # shape (12,), values ∈ [0, 1]
    session_boundary: bool


@dataclass
class SyntheticPersona:
    user_id:   str
    archetype: str
    t_human:   float               # ground-truth trust label ∈ [0, 1]
    turns:     List[TurnData] = field(default_factory=list)


# ============================================================================
# Dataset class
# ============================================================================

class SyntheticTrustDataset:
    """
    Generates 5 × n_variants_per_archetype synthetic personas.
    All randomness is seeded; same seed → identical personas.
    """

    ARCHETYPES: Dict[str, dict] = {
        "COOPERATIVE_STABLE": {
            "t_human_range":    (0.78, 0.92),
            "n_sessions":       4,
            "turns_per_session": 8,
        },
        "COOPERATIVE_GROWING": {
            "t_human_range":    (0.55, 0.75),
            "n_sessions":       5,
            "turns_per_session": 6,
        },
        "ADVERSARIAL": {
            "t_human_range":    (0.05, 0.22),
            "n_sessions":       2,
            "turns_per_session": 6,
        },
        "INCONSISTENT": {
            "t_human_range":    (0.30, 0.55),
            "n_sessions":       3,
            "turns_per_session": 8,
        },
        "NEW_THEN_DROPS": {
            "t_human_range":    (0.25, 0.45),
            "n_sessions":       3,
            "turns_per_session": 6,
        },
    }

    # Dimensionality of behavior_vec — must match whatever _compute_bc expects
    _BVEC_DIM: int = 12

    def __init__(self, n_variants_per_archetype: int = 40, seed: int = 42):
        self.n_variants = n_variants_per_archetype
        self._master_rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------
    def generate(self) -> List[SyntheticPersona]:
        personas: List[SyntheticPersona] = []
        for archetype, cfg in self.ARCHETYPES.items():
            for v in range(self.n_variants):
                # Derive a deterministic per-persona RNG from master
                variant_seed = int(self._master_rng.integers(0, 2**31))
                rng = np.random.default_rng(variant_seed)

                uid = f"syn_{archetype[:4].lower()}_{v:03d}"
                t_lo, t_hi = cfg["t_human_range"]
                t_human = float(np.clip(rng.uniform(t_lo, t_hi), 0.0, 1.0))

                turns = self._generate_turns(archetype, cfg, rng)
                personas.append(SyntheticPersona(
                    user_id=uid,
                    archetype=archetype,
                    t_human=t_human,
                    turns=turns,
                ))
        return personas

    def train_test_split(self, personas: List[SyntheticPersona],
                         test_ratio: float = 0.20,
                         seed: int = 0):
        rng     = np.random.default_rng(seed)
        indices = rng.permutation(len(personas))
        n_test  = max(1, int(len(personas) * test_ratio))
        test_set  = set(indices[:n_test].tolist())
        train = [p for i, p in enumerate(personas) if i not in test_set]
        test  = [p for i, p in enumerate(personas) if i     in test_set]
        return train, test

    # -----------------------------------------------------------------------
    # Archetype-specific signal generators
    # -----------------------------------------------------------------------

    def _generate_turns(self, archetype: str, cfg: dict,
                        rng: np.random.Generator) -> List[TurnData]:
        n_sess  = cfg["n_sessions"]
        n_turns = cfg["turns_per_session"]

        if archetype == "COOPERATIVE_STABLE":
            return self._coop_stable(n_sess, n_turns, rng)
        elif archetype == "COOPERATIVE_GROWING":
            return self._coop_growing(n_sess, n_turns, rng)
        elif archetype == "ADVERSARIAL":
            return self._adversarial(n_sess, n_turns, rng)
        elif archetype == "INCONSISTENT":
            return self._inconsistent(n_sess, n_turns, rng)
        elif archetype == "NEW_THEN_DROPS":
            return self._new_then_drops(n_sess, n_turns, rng)
        else:
            raise ValueError(f"Unknown archetype: {archetype}")

    # ── COOPERATIVE_STABLE ─────────────────────────────────────────────────
    # High, stable IC. Positive valence, moderate arousal.
    # Behavior vec close to a fixed persona centroid throughout.

    def _coop_stable(self, n_sess: int, n_turns: int,
                     rng: np.random.Generator) -> List[TurnData]:
        # Sample a persona centroid once; all turns stay near it
        centroid = rng.uniform(0.4, 0.8, size=self._BVEC_DIM).astype(np.float32)
        turns: List[TurnData] = []
        for s in range(n_sess):
            for t in range(n_turns):
                noise = rng.normal(0, 0.04, size=self._BVEC_DIM)
                bvec  = np.clip(centroid + noise, 0, 1).astype(np.float32)
                turns.append(TurnData(
                    ic_score = float(np.clip(rng.normal(0.88, 0.02), 0.75, 0.97)),
                    valence          = float(np.clip(rng.normal(0.65, 0.10), 0.30, 1.00)),
                    arousal          = float(np.clip(rng.normal(0.55, 0.08), 0.20, 0.90)),
                    behavior_vec     = bvec,
                    session_boundary = (t == 0),
                ))
        return turns

    # ── COOPERATIVE_GROWING ────────────────────────────────────────────────
    # IC starts at new-user level, gradually increases.
    # Valence ramps from neutral → positive.
    # Behavior vec starts scattered, converges to a centroid over sessions.

    def _coop_growing(self, n_sess: int, n_turns: int,
                      rng: np.random.Generator) -> List[TurnData]:
        target = rng.uniform(0.5, 0.8, size=self._BVEC_DIM).astype(np.float32)
        total  = n_sess * n_turns
        turns: List[TurnData] = []
        g = 0
        for s in range(n_sess):
            for t in range(n_turns):
                progress = g / max(total - 1, 1)       # 0 → 1 over all turns
                # Behavior starts random, shifts toward target
                scatter  = rng.uniform(0, 1, size=self._BVEC_DIM).astype(np.float32)
                bvec     = np.clip(
                    (1 - progress) * scatter + progress * target
                    + rng.normal(0, 0.05, size=self._BVEC_DIM),
                    0, 1
                ).astype(np.float32)
                turns.append(TurnData(
                    ic_score         = float(np.clip(
                        rng.normal(0.40 + 0.45 * progress, 0.035), 0.20, 0.95)),
                    valence          = float(np.clip(
                        rng.normal(-0.10 + 0.75 * progress, 0.10), -0.50, 1.00)),
                    arousal          = float(np.clip(
                        rng.normal(0.30 + 0.30 * progress, 0.08), 0.10, 0.90)),
                    behavior_vec     = bvec,
                    session_boundary = (t == 0),
                ))
                g += 1
        return turns

    # ── ADVERSARIAL ────────────────────────────────────────────────────────
    # IC drops frequently below theta_new (0.40).
    # Strongly negative valence, high arousal.
    # Behavior vec jumps randomly — no centroid.

    def _adversarial(self, n_sess: int, n_turns: int,
                     rng: np.random.Generator) -> List[TurnData]:
        turns: List[TurnData] = []
        for s in range(n_sess):
            for t in range(n_turns):
                # IC below theta_new on ~70% of turns
                if rng.random() < 0.70:
                    ic = float(np.clip(rng.normal(0.22, 0.07), 0.05, 0.38))
                else:
                    ic = float(np.clip(rng.normal(0.50, 0.08), 0.35, 0.65))
                # Random incoherent behavior vec (no stable centroid)
                bvec = rng.uniform(0, 1, size=self._BVEC_DIM).astype(np.float32)
                turns.append(TurnData(
                    ic_score         = ic,
                    valence          = float(np.clip(rng.normal(-0.70, 0.15), -1.00, -0.30)),
                    arousal          = float(np.clip(rng.normal(0.80, 0.10),  0.50,  1.00)),
                    behavior_vec     = bvec,
                    session_boundary = (t == 0),
                ))
        return turns

    # ── INCONSISTENT ───────────────────────────────────────────────────────
    # IC is adequate (~0.68) but behavior vec and valence oscillate.
    # Alternates between cooperative-looking and hostile turns.

    def _inconsistent(self, n_sess: int, n_turns: int,
                      rng: np.random.Generator) -> List[TurnData]:
        coop_vec  = rng.uniform(0.5, 0.9, size=self._BVEC_DIM).astype(np.float32)
        hostile_vec = np.clip(1.0 - coop_vec + rng.normal(0, 0.15, size=self._BVEC_DIM),
                              0, 1).astype(np.float32)
        turns: List[TurnData] = []
        cooperative = True   # toggle each session
        for s in range(n_sess):
            cooperative = not cooperative   # flip every session for clear oscillation
            for t in range(n_turns):
                # Also flip mid-session on a few turns to add within-session variance
                flipped = cooperative ^ (rng.random() < 0.35)
                if flipped:
                    base    = coop_vec
                    valence = float(np.clip(rng.normal(0.45, 0.15), -0.20, 0.90))
                    arousal = float(np.clip(rng.normal(0.45, 0.12),  0.15, 0.80))
                else:
                    base    = hostile_vec
                    valence = float(np.clip(rng.normal(-0.50, 0.15), -0.90, 0.10))
                    arousal = float(np.clip(rng.normal(0.70, 0.12),   0.40, 1.00))

                bvec = np.clip(
                    base + rng.normal(0, 0.08, size=self._BVEC_DIM),
                    0, 1
                ).astype(np.float32)
                turns.append(TurnData(
                    ic_score = float(np.clip(rng.normal(0.70, 0.04), 0.55, 0.82)),
                    valence          = valence,
                    arousal          = arousal,
                    behavior_vec     = bvec,
                    session_boundary = (t == 0),
                ))
        return turns

    # ── NEW_THEN_DROPS ─────────────────────────────────────────────────────
    # Session 0: new user, IC builds from ~0.38 → ~0.70.
    # Session 1: established, IC is comfortable (~0.75).
    # Session 2: sudden IC collapse (lighting / spoof / device change).

    def _new_then_drops(self, n_sess: int, n_turns: int,
                        rng: np.random.Generator) -> List[TurnData]:
        centroid = rng.uniform(0.45, 0.75, size=self._BVEC_DIM).astype(np.float32)
        turns: List[TurnData] = []
        for s in range(n_sess):
            for t in range(n_turns):
                progress_in_sess = t / max(n_turns - 1, 1)

                if s == 0:
                    # Ramps from new-user IC up to known-user threshold
                    ic = float(np.clip(
                        rng.normal(0.40 + 0.32 * progress_in_sess, 0.04),
                        0.20, 0.72))
                    valence = float(np.clip(rng.normal(0.30, 0.15), -0.20, 0.70))
                elif s == 1:
                    # Comfortable, established
                    ic = float(np.clip(rng.normal(0.78, 0.04), 0.67, 0.90))
                    valence = float(np.clip(rng.normal(0.55, 0.12), 0.20, 0.85))
                else:
                    # Session 2: IC drops sharply — falls below theta_new on most turns
                    ic = float(np.clip(rng.normal(0.28, 0.06), 0.05, 0.45))
                    valence = float(np.clip(rng.normal(0.00, 0.20), -0.50, 0.50))

                noise = rng.normal(0, 0.06 if s < 2 else 0.15, size=self._BVEC_DIM)
                bvec  = np.clip(centroid + noise, 0, 1).astype(np.float32)
                arousal = float(np.clip(rng.normal(0.45, 0.12), 0.15, 0.85))

                turns.append(TurnData(
                    ic_score         = ic,
                    valence          = valence,
                    arousal          = arousal,
                    behavior_vec     = bvec,
                    session_boundary = (t == 0),
                ))
        return turns


# ============================================================================
# Quick smoke-test
# ============================================================================

if __name__ == "__main__":
    ds = SyntheticTrustDataset(n_variants_per_archetype=4, seed=0)
    personas = ds.generate()
    print(f"Generated {len(personas)} personas")
    for p in personas:
        t0 = p.turns[0]
        print(
            f"  {p.user_id:30s}  archetype={p.archetype:25s}  "
            f"t_human={p.t_human:.3f}  turns={len(p.turns):3d}  "
            f"bvec_dim={t0.behavior_vec.shape[0]}  "
            f"ic0={t0.ic_score:.3f}"
        )
    train, test = ds.train_test_split(personas, test_ratio=0.20)
    print(f"\ntrain={len(train)}  test={len(test)}")
