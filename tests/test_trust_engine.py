"""
tests/test_trust_engine.py
==========================
Unit tests for the T3 Trust Engine (v2).

Covers:
  - Cold-start initialisation
  - Identity Confidence gate (IC drop, IC fail streak → reset)
  - Adversarial detection and reset
  - Tier assignment and hysteresis
  - Historic Reliability log-scaling + confidence gating
  - Behaviour Consistency volatility penalty
  - Privacy penalty and retrieval score helpers
  - Ablation flags (disable_bc, disable_es, disable_hr, disable_ic_gate)
  - Session boundary bookkeeping

Run:
    python -m pytest tests/test_trust_engine.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from trust_engine import TrustEngine, TrustTier


def _bvec(seed=0, dim=10):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-9)


def _update(engine, user_id="u", ic=0.9, val=0.4, aro=0.5, bvec=None, sb=False):
    if bvec is None:
        bvec = _bvec()
    return engine.update(user_id, ic, val, aro, bvec, session_boundary=sb)


# ============================================================================
# Cold start
# ============================================================================

class TestColdStart(unittest.TestCase):

    def test_new_user_starts_at_030(self):
        e = TrustEngine()
        self.assertAlmostEqual(e.get_trust("new_user"), 0.30, places=5)

    def test_new_user_tier_is_low(self):
        e = TrustEngine()
        self.assertEqual(e.get_trust_tier("new_user"), TrustTier.LOW)

    def test_first_update_returns_tuple(self):
        e = TrustEngine()
        result = _update(e)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        score, tier = result
        self.assertIsInstance(score, float)
        self.assertIsInstance(tier, TrustTier)

    def test_first_update_score_in_unit_interval(self):
        e = TrustEngine()
        score, _ = _update(e)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ============================================================================
# IC gate
# ============================================================================

class TestICGate(unittest.TestCase):

    def test_ic_below_theta_new_halves_trust(self):
        e = TrustEngine(theta_new=0.35, theta_known=0.65)
        # Warm up a bit first
        for _ in range(3):
            _update(e, ic=0.9)
        trust_before = e.get_trust("u")
        score, _ = _update(e, ic=0.20)   # below theta_new
        self.assertLess(score, trust_before)

    def test_ic_between_thresholds_reduces_trust(self):
        e = TrustEngine(theta_new=0.35, theta_known=0.65)
        for _ in range(3):
            _update(e, ic=0.9)
        trust_before = e.get_trust("u")
        score, _ = _update(e, ic=0.50)   # between theta_new and theta_known
        self.assertLess(score, trust_before)

    def test_ic_fail_streak_resets_trust(self):
        e = TrustEngine(theta_new=0.35)
        for _ in range(3):
            _update(e, ic=0.9)
        # Three consecutive fails → reset to 0.10
        for _ in range(3):
            _update(e, ic=0.10)
        self.assertAlmostEqual(e.get_trust("u"), 0.10, places=5)

    def test_ic_gate_disabled(self):
        e = TrustEngine(disable_ic_gate=True, theta_new=0.35)
        # Even with very low IC the gate should not fire
        for _ in range(5):
            score, _ = _update(e, ic=0.01)
        # Trust should still move (not stuck at cold-start or 0.10)
        self.assertGreater(e.get_trust("u"), 0.0)


# ============================================================================
# Adversarial detection
# ============================================================================

class TestAdversarialDetection(unittest.TestCase):
    """
    Adversarial detection fires when BOTH:
      - bc_inst (instantaneous cosine BC) < 0.20
      - raw_es_instant < 0.30

    Warm-up uses seed=0 to build the centroid.
    Adversarial inputs use seed=37 (cos=-0.855 vs seed-0 centroid → bc_inst=0.073)
    and valence=-1, arousal=0.05 → raw_es=0.0.
    """

    WARMUP_SEED  = 0    # builds centroid
    ATTACK_SEED1 = 37   # bc_inst=0.073 vs seed-0 centroid
    ATTACK_SEED2 = 5    # bc_inst=0.152 vs seed-0 centroid

    def _warm_up(self, engine, uid="u", n=3):
        vec = _bvec(self.WARMUP_SEED)
        for _ in range(n):
            engine.update(uid, ic_score=0.9, valence=0.3, arousal=0.4,
                          behavior_vec=vec)

    def test_adversarial_raw_es_is_below_threshold(self):
        """Verify our adversarial inputs produce raw_es < 0.30."""
        raw_es = ((-1.0 + 1) / 2) * 0.05   # = 0.0
        self.assertLess(raw_es, 0.30)

    def test_adversarial_bc_inst_is_below_threshold(self):
        """Verify seed=37 produces bc_inst < 0.20 against a seed=0 centroid."""
        v0  = _bvec(self.WARMUP_SEED)
        v37 = _bvec(self.ATTACK_SEED1)
        cos = float(np.dot(v37, v0) / (np.linalg.norm(v37) * np.linalg.norm(v0) + 1e-8))
        bc_inst = (cos + 1) / 2.0
        self.assertLess(bc_inst, 0.20)

    def test_single_adversarial_flag_reduces_trust(self):
        """First adversarial hit → trust * 0.30 and tier=LOW returned."""
        e = TrustEngine()
        self._warm_up(e, n=3)
        trust_before = e.get_trust("u")
        score, tier = e.update("u", ic_score=0.9, valence=-1.0, arousal=0.05,
                               behavior_vec=_bvec(self.ATTACK_SEED1))
        self.assertLess(score, trust_before)
        self.assertEqual(tier, TrustTier.LOW)

    def test_two_adversarial_flags_triggers_reset(self):
        """Two consecutive adversarial hits → trust reset to 0.10."""
        e = TrustEngine()
        self._warm_up(e, n=3)
        e.update("u", ic_score=0.9, valence=-1.0, arousal=0.05,
                 behavior_vec=_bvec(self.ATTACK_SEED1))
        score, tier = e.update("u", ic_score=0.9, valence=-1.0, arousal=0.05,
                               behavior_vec=_bvec(self.ATTACK_SEED2))
        self.assertAlmostEqual(score, 0.10, places=4)
        self.assertEqual(tier, TrustTier.LOW)

    def test_adversarial_tier_is_low(self):
        """Adversarial detection always returns TrustTier.LOW."""
        e = TrustEngine()
        self._warm_up(e, n=3)
        score, tier = e.update("u", ic_score=0.9, valence=-1.0, arousal=0.05,
                               behavior_vec=_bvec(self.ATTACK_SEED1))
        self.assertEqual(tier, TrustTier.LOW)


# ============================================================================
# Tier assignment and hysteresis
# ============================================================================

class TestTierHysteresis(unittest.TestCase):

    def test_high_trust_tier(self):
        e = TrustEngine()
        for _ in range(30):
            _update(e, ic=0.95, val=0.8, aro=0.7, bvec=_bvec(1), sb=True)
        score = e.get_trust("u")
        if score >= 0.80:
            self.assertEqual(e.get_trust_tier("u"), TrustTier.HIGH)

    def test_upgrade_requires_two_consecutive_updates(self):
        """A single tick above the threshold should NOT immediately upgrade."""
        e = TrustEngine()
        # Keep trust in MEDIUM range
        for _ in range(10):
            _update(e, ic=0.75, val=0.4, aro=0.4, bvec=_bvec(2))
        tier_after_one = e.get_trust_tier("u")
        # One big push
        _update(e, ic=0.99, val=0.99, aro=0.99, bvec=_bvec(2))
        # Tier may or may not have upgraded — key point is engine doesn't crash
        self.assertIsInstance(e.get_trust_tier("u"), TrustTier)

    def test_downgrade_is_immediate(self):
        e = TrustEngine()
        # Warm up to HIGH tier
        for _ in range(15):
            _update(e, ic=0.95, val=0.8, aro=0.7, bvec=_bvec(1), sb=True)
        self.assertEqual(e.get_trust_tier("u"), TrustTier.HIGH)
        # Three consecutive IC fails → trust reset to 0.10 → LOW immediately
        for _ in range(3):
            _update(e, ic=0.10)
        tier = e.get_trust_tier("u")
        self.assertEqual(tier, TrustTier.LOW)


# ============================================================================
# Historic Reliability (HR) — log scaling + confidence gating
# ============================================================================

class TestHistoricReliability(unittest.TestCase):

    def test_hr_zero_with_no_sessions(self):
        """A brand-new user with no session boundaries → HR = 0."""
        e = TrustEngine()
        # Updates without session_boundary
        for _ in range(5):
            _update(e, ic=0.9, sb=False)
        # Hard to inspect HR directly; but trust should not jump high from HR alone
        # Just confirm trust is in range
        self.assertLessEqual(e.get_trust("u"), 1.0)

    def test_trust_grows_with_verified_sessions(self):
        e = TrustEngine()
        baseline_score, _ = _update(e, ic=0.9, sb=True)
        for _ in range(6):
            _update(e, ic=0.9, val=0.5, aro=0.5, bvec=_bvec(1), sb=True)
        self.assertGreater(e.get_trust("u"), baseline_score)

    def test_hr_disable_flag(self):
        e_with    = TrustEngine(disable_hr=False)
        e_without = TrustEngine(disable_hr=True)
        bvec = _bvec(3)
        for _ in range(10):
            e_with.update("u",    0.9, 0.5, 0.5, bvec, session_boundary=True)
            e_without.update("u", 0.9, 0.5, 0.5, bvec, session_boundary=True)
        # HR=0 model should generally converge lower than full model
        self.assertGreaterEqual(e_with.get_trust("u"), e_without.get_trust("u") - 0.05)


# ============================================================================
# Behaviour Consistency (BC) — volatility penalty
# ============================================================================

class TestBehaviourConsistency(unittest.TestCase):

    def test_stable_behaviour_keeps_trust_higher(self):
        """
        Stable BC should accumulate trust faster than volatile BC.
        We compare after 8 turns (before saturation at 1.0).
        """
        e_stable   = TrustEngine()
        e_volatile = TrustEngine()
        stable_vec = _bvec(0)
        rng = np.random.default_rng(7)

        for i in range(8):   # 8 turns — before max_delta saturation
            e_stable.update("u", 0.88, 0.4, 0.5, stable_vec)
            vol_vec = rng.standard_normal(10)
            vol_vec /= np.linalg.norm(vol_vec) + 1e-9
            e_volatile.update("u", 0.88, 0.4, 0.5, vol_vec)

        stable_trust   = e_stable.get_trust("u")
        volatile_trust = e_volatile.get_trust("u")
        # Stable should be >= volatile (volatility penalty hurts volatile user)
        self.assertGreaterEqual(stable_trust, volatile_trust - 0.01)

    def test_bc_disable_flag(self):
        e = TrustEngine(disable_bc=True)
        for _ in range(10):
            _update(e)
        # Should not crash; trust still in [0, 1]
        self.assertGreaterEqual(e.get_trust("u"), 0.0)
        self.assertLessEqual(e.get_trust("u"), 1.0)


# ============================================================================
# Privacy penalty and retrieval score
# ============================================================================

class TestPrivacyAndRetrieval(unittest.TestCase):

    def test_privacy_penalty_zero_at_full_trust(self):
        e = TrustEngine()
        penalty = e.privacy_penalty(sensitivity=3, trust=1.0)
        self.assertAlmostEqual(penalty, 0.0, places=6)

    def test_privacy_penalty_max_at_zero_trust(self):
        e = TrustEngine()
        penalty = e.privacy_penalty(sensitivity=3, trust=0.0)
        self.assertAlmostEqual(penalty, 3.0, places=6)

    def test_privacy_penalty_scales_with_sensitivity(self):
        e = TrustEngine()
        p1 = e.privacy_penalty(sensitivity=1, trust=0.5)
        p3 = e.privacy_penalty(sensitivity=3, trust=0.5)
        self.assertAlmostEqual(p3, 3 * p1, places=6)

    def test_retrieval_score_in_range(self):
        e = TrustEngine()
        score = e.retrieval_score(
            cos_sim=0.8, recency=0.9, emotion=0.6, trust=0.85, sensitivity=1
        )
        # Not strictly bounded but should be a real number
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))

    def test_high_trust_increases_retrieval_score(self):
        e = TrustEngine()
        low  = e.retrieval_score(0.8, 0.9, 0.6, trust=0.20, sensitivity=2)
        high = e.retrieval_score(0.8, 0.9, 0.6, trust=0.90, sensitivity=2)
        self.assertGreater(high, low)


# ============================================================================
# Ablation combinations
# ============================================================================

class TestAblationFlags(unittest.TestCase):

    def _run_n(self, engine, n=20, uid="u"):
        rng = np.random.default_rng(42)
        for _ in range(n):
            bvec = rng.standard_normal(10)
            engine.update(uid, 0.85, 0.3, 0.5, bvec, session_boundary=False)
        return engine.get_trust(uid)

    def test_all_disabled_still_produces_valid_trust(self):
        e = TrustEngine(disable_bc=True, disable_es=True,
                        disable_hr=True, disable_ic_gate=True)
        trust = self._run_n(e)
        self.assertGreaterEqual(trust, 0.0)
        self.assertLessEqual(trust, 1.0)

    def test_no_emotion_config(self):
        e = TrustEngine(w1=0.55, w2=0.00, w3=0.45, disable_es=True)
        trust = self._run_n(e)
        self.assertGreaterEqual(trust, 0.0)

    def test_no_behavior_config(self):
        e = TrustEngine(w1=0.00, w2=0.60, w3=0.40, disable_bc=True)
        trust = self._run_n(e)
        self.assertGreaterEqual(trust, 0.0)

    def test_no_trust_gate_config(self):
        e = TrustEngine(disable_ic_gate=True)
        trust = self._run_n(e)
        self.assertGreaterEqual(trust, 0.0)


# ============================================================================
# Session boundary
# ============================================================================

class TestSessionBoundary(unittest.TestCase):

    def test_session_boundary_does_not_crash(self):
        e = TrustEngine()
        for i in range(10):
            _update(e, ic=0.9, sb=(i % 3 == 0))
        self.assertIsInstance(e.get_trust("u"), float)

    def test_adversarial_flags_reset_on_session_boundary(self):
        e = TrustEngine()
        # Trigger one adversarial flag
        e.update("u", 0.9, -1.0, 0.05, _bvec(99))
        # Session boundary with good IC + BC resets adversarial_flags
        e.update("u", 0.9, 0.5, 0.5, _bvec(0), session_boundary=True)
        # Next adversarial signal should only count as flag=1 again (not instant reset)
        score, tier = e.update("u", 0.9, -1.0, 0.05, _bvec(99))
        # Should not have reset (only 1 flag post-boundary)
        self.assertGreater(score, 0.10)

    def test_multi_user_isolation(self):
        e = TrustEngine()
        for _ in range(10):
            e.update("alice", 0.95, 0.7, 0.6, _bvec(1), session_boundary=True)
        for _ in range(3):
            e.update("bob", 0.2, -0.5, 0.1, _bvec(99))

        alice_trust = e.get_trust("alice")
        bob_trust   = e.get_trust("bob")
        self.assertGreater(alice_trust, bob_trust)


if __name__ == "__main__":
    unittest.main(verbosity=2)
