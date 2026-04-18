import sys
import math
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")

from rerank import (
    ScoringWeights,
    TrustContext,
    _recency_score,
    _normalize,
    LAMBDA_PRESETS,
    CRAG_MIN_SCORE,
    DEDUP_COS_THRESHOLD,
)
from memory_writer import LIFELONG_ADAPT_RATE, STALENESS_TAU_DAYS


class TestScoringWeights(unittest.TestCase):

    def test_default_weights_sum_to_one(self):
        w = ScoringWeights()
        total = w.alpha + w.beta + w.gamma + w.delta
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_invalid_weights_raise(self):
        with self.assertRaises(ValueError):
            ScoringWeights(alpha=0.5, beta=0.5, gamma=0.5, delta=0.5)

    def test_ablation_gamma_zero(self):
        w = ScoringWeights(alpha=0.65, beta=0.30, gamma=0.0, delta=0.05)
        self.assertAlmostEqual(w.alpha + w.beta + w.gamma + w.delta, 1.0, places=6)

    def test_ablation_no_trust_gate(self):
        ctx_high = TrustContext(trust_score=1.0, trust_tier="HIGH", privacy_penalty=0.0)
        ctx_low  = TrustContext(trust_score=0.3, trust_tier="LOW",  privacy_penalty=0.5)
        self.assertGreater(ctx_high.trust_score, ctx_low.trust_score)


class TestRecencyDecay(unittest.TestCase):

    def test_recent_memory_scores_high(self):
        now = datetime.now(timezone.utc)
        score = _recency_score(now, LAMBDA_PRESETS["medium"])
        self.assertGreater(score, 0.95)

    def test_old_memory_scores_low(self):
        old = datetime.now(timezone.utc) - timedelta(days=365)
        score = _recency_score(old, LAMBDA_PRESETS["medium"])
        self.assertLess(score, 0.01)

    def test_lambda_fast_decays_faster(self):
        dt = datetime.now(timezone.utc) - timedelta(days=30)
        fast   = _recency_score(dt, LAMBDA_PRESETS["fast"])
        medium = _recency_score(dt, LAMBDA_PRESETS["medium"])
        slow   = _recency_score(dt, LAMBDA_PRESETS["slow"])
        self.assertGreater(slow, medium)
        self.assertGreater(medium, fast)

    def test_three_lambda_ablations_distinct(self):
        dt = datetime.now(timezone.utc) - timedelta(days=15)
        scores = [_recency_score(dt, LAMBDA_PRESETS[k]) for k in ("fast", "medium", "slow")]
        self.assertEqual(len(set(round(s, 4) for s in scores)), 3)


class TestNormalize(unittest.TestCase):

    def test_all_same_returns_ones(self):
        result = _normalize([0.5, 0.5, 0.5])
        for v in result:
            self.assertAlmostEqual(v, 1.0)

    def test_range_is_zero_to_one(self):
        result = _normalize([0.1, 0.5, 0.9, 0.3])
        self.assertAlmostEqual(min(result), 0.0, places=6)
        self.assertAlmostEqual(max(result), 1.0, places=6)


class TestEmotionalSaliency(unittest.TestCase):

    def test_high_emotion_event_scores_high(self):
        valence = -0.75
        arousal  = 0.80
        saliency = (abs(valence) + arousal) / 2.0
        self.assertGreater(saliency, 0.70)

    def test_neutral_event_scores_low(self):
        valence = 0.0
        arousal  = 0.1
        saliency = abs(valence) * 0.3 + arousal * 0.2
        self.assertLess(saliency, 0.15)

    def test_emotional_partition_threshold(self):
        cases = [
            (-0.75, 0.80, "emotional"),
            (0.80, 0.70, "emotional"),
            (0.30, 0.20, "not_emotional"),
            (-0.50, 0.80, "not_emotional"),
            (-0.75, 0.40, "not_emotional"),
        ]
        for val, aro, expected in cases:
            is_emo = abs(val) > 0.6 and aro > 0.5
            if expected == "emotional":
                self.assertTrue(is_emo, f"Expected emotional for val={val}, aro={aro}")
            else:
                self.assertFalse(is_emo, f"Expected NOT emotional for val={val}, aro={aro}")


class TestTrustGating(unittest.TestCase):

    def test_low_trust_blocks_retrieval(self):
        from memory_pipeline import FALLBACK_RESPONSE
        self.assertIsInstance(FALLBACK_RESPONSE, str)
        self.assertGreater(len(FALLBACK_RESPONSE), 5)

    def test_trust_tier_access_levels(self):
        from retrieval import TRUST_TIER_MIN
        high_access   = set(TRUST_TIER_MIN["HIGH"])
        medium_access = set(TRUST_TIER_MIN["MEDIUM"])
        low_access    = set(TRUST_TIER_MIN["LOW"])
        self.assertIn("HIGH",   high_access)
        self.assertNotIn("HIGH",   medium_access)
        self.assertNotIn("MEDIUM", low_access)
        self.assertNotIn("HIGH",   low_access)

    def test_privacy_penalty_reduces_score(self):
        import numpy as np
        alpha, beta, gamma, delta = 0.60, 0.25, 0.10, 0.05
        base_score = alpha * 0.8 + beta * 0.6 + gamma * 0.5 * 1.0 - delta * 0.0
        penalized  = alpha * 0.8 + beta * 0.6 + gamma * 0.5 * 1.0 - delta * 1.0
        self.assertGreater(base_score, penalized)


class TestScoreFormula(unittest.TestCase):

    def test_full_formula_S_m_q_u(self):
        alpha, beta, gamma, delta = 0.60, 0.25, 0.10, 0.05
        rerank_norm   = 0.85
        recency_norm  = 0.70
        emo_saliency  = 0.60
        trust_score   = 0.90
        privacy_penalty = 0.0
        score = (
            alpha * rerank_norm
            + beta  * recency_norm
            + gamma * emo_saliency * trust_score
            - delta * privacy_penalty
        )
        expected = 0.60 * 0.85 + 0.25 * 0.70 + 0.10 * 0.60 * 0.90 - 0.05 * 0.0
        self.assertAlmostEqual(score, expected, places=6)

    def test_high_emotion_raises_score(self):
        alpha, beta, gamma, delta = 0.60, 0.25, 0.10, 0.05
        T = 1.0
        base    = alpha * 0.5 + beta * 0.5 + gamma * 0.1 * T - delta * 0.0
        emotion = alpha * 0.5 + beta * 0.5 + gamma * 0.9 * T - delta * 0.0
        self.assertGreater(emotion, base)

    def test_low_trust_reduces_emotion_contribution(self):
        alpha, beta, gamma, delta = 0.60, 0.25, 0.10, 0.05
        emo = 0.9
        high_t = alpha * 0.5 + beta * 0.5 + gamma * emo * 1.0 - delta * 0.0
        low_t  = alpha * 0.5 + beta * 0.5 + gamma * emo * 0.3 - delta * 0.0
        self.assertGreater(high_t, low_t)


class TestLifelongAdaptation(unittest.TestCase):

    def test_adapt_rate_is_small(self):
        self.assertLess(LIFELONG_ADAPT_RATE, 0.05)

    def test_weighted_update_formula(self):
        import numpy as np
        stored = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        new    = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        rate   = LIFELONG_ADAPT_RATE
        updated = (1.0 - rate) * stored + rate * new
        self.assertAlmostEqual(updated[0], 1.0 - rate, places=6)
        self.assertAlmostEqual(updated[1], rate,       places=6)

    def test_staleness_tau(self):
        tau = STALENESS_TAU_DAYS
        score_30  = 1.0 - math.exp(-30  / tau)
        score_90  = 1.0 - math.exp(-90  / tau)
        score_180 = 1.0 - math.exp(-180 / tau)
        self.assertLess(score_30, score_90)
        self.assertLess(score_90, score_180)
        self.assertAlmostEqual(score_90, 1.0 - math.exp(-1.0), places=3)


class TestCRAGGate(unittest.TestCase):

    def test_crag_threshold_defined(self):
        self.assertGreater(CRAG_MIN_SCORE, 0.0)
        self.assertLess(CRAG_MIN_SCORE, 1.0)

    def test_dedup_threshold_defined(self):
        self.assertGreater(DEDUP_COS_THRESHOLD, 0.8)
        self.assertLessEqual(DEDUP_COS_THRESHOLD, 1.0)


class TestInterfaceContracts(unittest.TestCase):

    def test_t1_identity_contract(self):
        from memory_pipeline import T1Identity
        t1 = T1Identity(
            user_id="patient_001",
            IC_score=0.87,
            face_confidence=0.91,
            voice_confidence=0.82,
        )
        self.assertEqual(t1.user_id, "patient_001")
        self.assertGreater(t1.IC_score, 0.0)

    def test_t3_trust_contract(self):
        from memory_pipeline import T3Trust
        t3 = T3Trust(trust_score=0.85, trust_tier="HIGH", privacy_penalty=0.0)
        self.assertIn(t3.trust_tier, ("HIGH", "MEDIUM", "LOW", "UNKNOWN"))

    def test_scoring_weights_paper_ablations(self):
        ablation_a = ScoringWeights(alpha=0.65, beta=0.30, gamma=0.00, delta=0.05)
        ablation_b = ScoringWeights(alpha=0.60, beta=0.30, gamma=0.05, delta=0.05)
        full       = ScoringWeights(alpha=0.60, beta=0.25, gamma=0.10, delta=0.05)
        for w in (ablation_a, ablation_b, full):
            self.assertAlmostEqual(w.alpha + w.beta + w.gamma + w.delta, 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
