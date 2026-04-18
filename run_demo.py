import numpy as np
from trust_engine import TrustEngine

engine = TrustEngine()

user_id = "user_1"

print("\n--- TRUST ENGINE DEMO ---\n")

for step in range(15):
    
    # Simulate inputs
    ic_score = np.random.uniform(0.5, 1.0)      # identity confidence
    valence = np.random.uniform(-1, 1)          # emotion
    arousal = np.random.uniform(0, 1)
    behavior_vec = np.random.randn(10)          # random vector

    # Introduce some events
    if step == 5:
        print("\n⚠️ IC DROP EVENT\n")
        ic_score = 0.3

    if step == 10:
        print("\n🚨 ADVERSARIAL EVENT\n")
        valence = -1
        arousal = 0.1

    trust, tier = engine.update(
        user_id,
        ic_score,
        valence,
        arousal,
        behavior_vec,
        session_boundary=(step % 5 == 0)
    )

    print(f"Step {step}")
    print(f"IC: {ic_score:.2f}, Valence: {valence:.2f}, Arousal: {arousal:.2f}")
    print(f"Trust: {trust:.3f}, Tier: {tier.value}")
    print("-" * 40)