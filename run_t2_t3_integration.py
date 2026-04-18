"""
run_t2_t3_integration.py
========================
End-to-end demonstration of T2 (Memory) + T3 (Trust) working together.

Shows the full flow:
  1. TrustService evaluates user identity + behaviour → trust score + tier
  2. MemoryPipelineInput is built with that trust context
  3. Memory pipeline retrieves & reranks memories gated by trust

Run (no DB required for the trust portion):
    python run_t2_t3_integration.py --demo-trust-only

Run (with PostgreSQL configured in .env):
    python run_t2_t3_integration.py
"""

from __future__ import annotations

import argparse
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integration.trust_service import TrustService


def demo_trust_only():
    """Runs only the T3 trust engine — no DB required."""
    print("\n" + "=" * 60)
    print("  MEMORA  —  T3 Trust Engine Demo")
    print("=" * 60)

    svc = TrustService()
    user_id = "demo_patient_001"

    scenarios = [
        dict(step=0,  label="Normal session start",
             ic=0.88, val=0.3,  aro=0.4, bvec=np.random.randn(10), sb=True),
        dict(step=1,  label="Positive interaction",
             ic=0.91, val=0.6,  aro=0.5, bvec=np.random.randn(10), sb=False),
        dict(step=2,  label="Positive interaction",
             ic=0.85, val=0.5,  aro=0.4, bvec=np.random.randn(10), sb=False),
        dict(step=3,  label="⚠️  IC drop",
             ic=0.30, val=0.1,  aro=0.2, bvec=np.random.randn(10), sb=False),
        dict(step=4,  label="Recovery attempt",
             ic=0.78, val=0.4,  aro=0.5, bvec=np.random.randn(10), sb=False),
        dict(step=5,  label="Session boundary",
             ic=0.90, val=0.5,  aro=0.4, bvec=np.random.randn(10), sb=True),
        dict(step=6,  label="🚨 Adversarial signal",
             ic=0.92, val=-1.0, aro=0.05, bvec=np.random.randn(10), sb=False),
        dict(step=7,  label="🚨 Adversarial signal (repeat)",
             ic=0.88, val=-0.9, aro=0.05, bvec=np.random.randn(10), sb=False),
        dict(step=8,  label="Post-adversarial probe",
             ic=0.85, val=0.3,  aro=0.4, bvec=np.random.randn(10), sb=False),
    ]

    np.random.seed(42)

    for s in scenarios:
        trust = svc.evaluate(
            user_id=user_id,
            ic_score=s["ic"],
            valence=s["val"],
            arousal=s["aro"],
            behavior_vec=s["bvec"],
            sensitivity=2,
            session_boundary=s["sb"],
        )
        print(f"\nStep {s['step']:2d}  [{s['label']}]")
        print(f"  IC={s['ic']:.2f}  val={s['val']:+.1f}  aro={s['aro']:.1f}")
        print(f"  → trust={trust.trust_score:.3f}  tier={trust.trust_tier:<7}  "
              f"privacy_penalty={trust.privacy_penalty:.3f}")
        print("  " + "-" * 50)

    print("\nDemo complete.\n")


def demo_full_pipeline():
    """Full T2+T3 demo — requires PostgreSQL and .env configured."""
    try:
        from memory.memory_pipeline import run_memory_pipeline, MemoryPipelineInput
        from memory.memory_pipeline import T1Identity, T3Trust as PipelineT3Trust
    except ImportError as e:
        print(f"Memory pipeline import failed: {e}")
        print("Make sure PostgreSQL is running and .env is configured.")
        return

    svc = TrustService()
    np.random.seed(42)

    # 1. Get trust evaluation
    trust = svc.evaluate(
        user_id="test_patient_001",
        ic_score=0.87,
        valence=0.4,
        arousal=0.5,
        behavior_vec=np.random.randn(10),
        sensitivity=2,
        session_boundary=True,
    )

    print(f"Trust evaluated: score={trust.trust_score}  tier={trust.trust_tier}")

    # 2. Build pipeline input
    identity = T1Identity(
        user_id="test_patient_001",
        IC_score=0.87,
        face_confidence=0.85,
        voice_confidence=0.89,
    )

    pipeline_trust = PipelineT3Trust(
        trust_score=trust.trust_score,
        trust_tier=trust.trust_tier,
        privacy_penalty=trust.privacy_penalty,
    )

    inp = MemoryPipelineInput(
        query="Did Ananya visit recently?",
        identity=identity,
        trust=pipeline_trust,
    )

    # 3. Run pipeline
    result = run_memory_pipeline(inp)

    print(f"Pipeline result: route={result.route}  passed_crag={result.passed_crag}")
    print(f"Top memories retrieved: {len(result.memories)}")
    if result.context_string:
        print("\nContext:\n" + result.context_string[:400])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-trust-only", action="store_true",
                        help="Run only the trust engine demo (no DB required)")
    args = parser.parse_args()

    if args.demo_trust_only or True:   # default to trust-only
        demo_trust_only()
    else:
        demo_full_pipeline()
