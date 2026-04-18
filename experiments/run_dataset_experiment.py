"""
run_dataset_experiment.py
=========================
Feeds real + synthetic datasets through the TrustEngine and logs:
  - per-turn trust score, tier, BC, ES, HR
  - per-conversation trust trajectory
  - tier transition events
  - adversarial response behavior

Produces structured CSV + JSON logs ready for evaluation metrics.

Usage
-----
    # Full run (all datasets)
    python run_dataset_experiment.py

    # Single dataset
    python run_dataset_experiment.py --dataset empathetic --max_convs 200

    # Ablation mode
    python run_dataset_experiment.py --ablation no_emotion_weight

Experiments run by this script
--------------------------------
EXP-1  Trust evolution across EmpatheticDialogues conversations
EXP-2  Behavioral consistency across PersonaChat
EXP-3  Multi-session HR accumulation on MSC
EXP-4  Adversarial vs cooperative comparison (synthetic)
EXP-5  TCE computation on synthetic trust dataset
"""

from __future__ import annotations
import sys, os, argparse, json, csv, time
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from trust_engine import TrustEngine, TrustTier
from data.empathetic_loader import EmpatheticLoader, TurnSample
from data.persona_loader import PersonaLoader, MSCLoader
from data.synthetic_trust_dataset import SyntheticTrustDataset, SyntheticPersona


# ============================================================================
# Result dataclasses
# ============================================================================

@dataclass
class TurnResult:
    conv_id:       str
    turn_index:    int
    speaker_id:    str
    emotion_label: str
    valence:       float
    arousal:       float
    ic_score:      float
    trust_score:   float
    trust_tier:    str
    text_snippet:  str   # first 60 chars

@dataclass
class ConvResult:
    conv_id:          str
    emotion_label:    str
    n_turns:          int
    trust_start:      float
    trust_end:        float
    trust_min:        float
    trust_max:        float
    trust_mean:       float
    tier_transitions: int
    final_tier:       str
    turns:            list[TurnResult] = field(default_factory=list)

@dataclass
class TCEResult:
    user_id:       str
    archetype:     str
    t_human:       float
    t_computed:    float
    tce:           float   # |t_human - t_computed|
    final_tier:    str
    notes:         str


# ============================================================================
# TrustEngine Ablation Configs
# ============================================================================

ABLATION_CONFIGS = {
    "full": dict(
        decay=0.95, w1=0.55, w2=0.20, w3=0.25,
        alpha_bc=0.10, alpha_es=0.30,
        theta_known=0.65, theta_new=0.40, max_delta=0.15
    ),
    "no_emotion_weight": dict(
        decay=0.95, w1=0.65, w2=0.00, w3=0.35,   # w2=0, redistribute to w1
        alpha_bc=0.10, alpha_es=0.30,
        theta_known=0.65, theta_new=0.40, max_delta=0.15
    ),
    "no_trust_gate": dict(
        # theta_known very low → effectively no gating (A1 ablation)
        decay=0.95, w1=0.55, w2=0.20, w3=0.25,
        alpha_bc=0.10, alpha_es=0.30,
        theta_known=0.00, theta_new=0.00, max_delta=0.15
    ),
    "high_decay": dict(
        decay=0.99, w1=0.55, w2=0.20, w3=0.25,   # slower decay
        alpha_bc=0.10, alpha_es=0.30,
        theta_known=0.65, theta_new=0.40, max_delta=0.15
    ),
    "low_decay": dict(
        decay=0.85, w1=0.55, w2=0.20, w3=0.25,   # faster decay
        alpha_bc=0.10, alpha_es=0.30,
        theta_known=0.65, theta_new=0.40, max_delta=0.15
    ),
}


# ============================================================================
# Core experiment functions
# ============================================================================

def run_conversation(engine: TrustEngine,
                     turns: list[TurnSample],
                     conv_id: str,
                     emotion_label: str) -> ConvResult:
    """Feed one conversation through TrustEngine; return ConvResult."""
    trust_scores: list[float] = []
    turn_results: list[TurnResult] = []
    tier_sequence: list[str] = []

    for turn in turns:
        score, tier = engine.update(
            user_id=turn.speaker_id,
            ic_score=turn.ic_score,
            valence=turn.valence,
            arousal=turn.arousal,
            behavior_vec=turn.behavior_vec,
            session_boundary=turn.session_boundary,
        )
        trust_scores.append(score)
        tier_str = tier.value if hasattr(tier, "value") else str(tier)
        tier_sequence.append(tier_str)

        turn_results.append(TurnResult(
            conv_id=conv_id,
            turn_index=turn.turn_index,
            speaker_id=turn.speaker_id,
            emotion_label=turn.emotion_label,
            valence=round(turn.valence, 4),
            arousal=round(turn.arousal, 4),
            ic_score=round(turn.ic_score, 4),
            trust_score=round(score, 4),
            trust_tier=tier_str,
            text_snippet=turn.text[:60],
        ))

    # Count tier transitions
    transitions = sum(
        1 for i in range(1, len(tier_sequence))
        if tier_sequence[i] != tier_sequence[i - 1]
    )

    return ConvResult(
        conv_id=conv_id,
        emotion_label=emotion_label,
        n_turns=len(turns),
        trust_start=trust_scores[0] if trust_scores else 0.0,
        trust_end=trust_scores[-1] if trust_scores else 0.0,
        trust_min=float(np.min(trust_scores)) if trust_scores else 0.0,
        trust_max=float(np.max(trust_scores)) if trust_scores else 0.0,
        trust_mean=float(np.mean(trust_scores)) if trust_scores else 0.0,
        tier_transitions=transitions,
        final_tier=tier_sequence[-1] if tier_sequence else "LOW",
        turns=turn_results,
    )


# ============================================================================
# EXP-1: EmpatheticDialogues
# ============================================================================

def exp1_empathetic(engine: TrustEngine, max_convs: int = 300,
                    results_dir: str = "results") -> list[ConvResult]:
    print("\n[EXP-1] EmpatheticDialogues — Trust evolution by emotion")
    loader = EmpatheticLoader(split="train", max_convs=max_convs)
    convs  = loader.load()
    print(f"  Loaded {len(convs)} conversations")

    conv_results: list[ConvResult] = []
    for conv in convs:
        if not conv.turns:
            continue
        result = run_conversation(
            engine=engine,
            turns=conv.turns,
            conv_id=conv.conv_id,
            emotion_label=conv.emotion_label,
        )
        conv_results.append(result)

    _save_conv_results(conv_results, os.path.join(results_dir, "exp1_empathetic.csv"))
    _print_summary("EXP-1", conv_results)
    return conv_results


# ============================================================================
# EXP-2: PersonaChat
# ============================================================================

def exp2_persona(engine: TrustEngine, max_convs: int = 200,
                 results_dir: str = "results") -> list[ConvResult]:
    print("\n[EXP-2] PersonaChat — Behavioral consistency trust dynamics")
    loader = PersonaLoader(max_convs=max_convs)
    convs  = loader.load()
    print(f"  Loaded {len(convs)} conversations")

    conv_results: list[ConvResult] = []
    for conv in convs:
        if not conv.turns:
            continue
        result = run_conversation(
            engine=engine,
            turns=conv.turns,
            conv_id=conv.conv_id,
            emotion_label=conv.emotion_label,
        )
        conv_results.append(result)

    _save_conv_results(conv_results, os.path.join(results_dir, "exp2_persona.csv"))
    _print_summary("EXP-2", conv_results)
    return conv_results


# ============================================================================
# EXP-3: MSC — multi-session HR accumulation
# ============================================================================

def exp3_msc(engine: TrustEngine, max_speakers: int = 50,
             results_dir: str = "results") -> list[ConvResult]:
    print("\n[EXP-3] MSC — Multi-session HR accumulation")
    loader = MSCLoader(max_speakers=max_speakers, sessions_per_speaker=3)
    users  = loader.load()
    print(f"  Loaded {len(users)} multi-session users")

    conv_results: list[ConvResult] = []
    for user in users:
        all_turns = user.all_turns()
        result = run_conversation(
            engine=engine,
            turns=all_turns,
            conv_id=user.user_id,
            emotion_label="multi_session",
        )
        conv_results.append(result)

    _save_conv_results(conv_results, os.path.join(results_dir, "exp3_msc.csv"))
    _print_summary("EXP-3", conv_results)
    return conv_results


# ============================================================================
# EXP-4: Adversarial vs Cooperative (Synthetic)
# ============================================================================

def exp4_adversarial(engine: TrustEngine, n_variants: int = 8,
                     results_dir: str = "results") -> dict:
    print("\n[EXP-4] Adversarial vs Cooperative — synthetic personas")
    ds = SyntheticTrustDataset(n_variants_per_archetype=n_variants, seed=42)
    personas = ds.generate()

    coop_scores: list[float] = []
    adv_scores:  list[float] = []
    conv_results: list[ConvResult] = []

    for p in personas:
        result = run_conversation(
            engine=engine,
            turns=p.turns,
            conv_id=p.user_id,
            emotion_label=p.archetype,
        )
        conv_results.append(result)
        if p.archetype == "ADVERSARIAL":
            adv_scores.append(result.trust_end)
        elif p.archetype in ("COOPERATIVE_STABLE", "COOPERATIVE_GROWING"):
            coop_scores.append(result.trust_end)

    _save_conv_results(conv_results, os.path.join(results_dir, "exp4_adversarial.csv"))

    summary = {
        "cooperative_mean_trust": float(np.mean(coop_scores)) if coop_scores else 0.0,
        "cooperative_std_trust":  float(np.std(coop_scores))  if coop_scores else 0.0,
        "adversarial_mean_trust": float(np.mean(adv_scores))  if adv_scores else 0.0,
        "adversarial_std_trust":  float(np.std(adv_scores))   if adv_scores else 0.0,
        "separation":             float(np.mean(coop_scores) - np.mean(adv_scores))
                                  if (coop_scores and adv_scores) else 0.0,
    }
    print(f"  Cooperative final trust: {summary['cooperative_mean_trust']:.3f} "
          f"± {summary['cooperative_std_trust']:.3f}")
    print(f"  Adversarial  final trust: {summary['adversarial_mean_trust']:.3f} "
          f"± {summary['adversarial_std_trust']:.3f}")
    print(f"  Separation gap: {summary['separation']:.3f}")
    return summary


# ============================================================================
# EXP-5: TCE Computation
# ============================================================================

def exp5_tce(engine: TrustEngine, n_variants: int = 10,
             results_dir: str = "results") -> dict:
    print("\n[EXP-5] TCE — Trust Calibration Error")
    ds = SyntheticTrustDataset(n_variants_per_archetype=n_variants, seed=99)
    personas = ds.generate()
    _, test_personas = ds.train_test_split(personas, test_ratio=0.20)
    print(f"  Test set: {len(test_personas)} personas")

    tce_results: list[TCEResult] = []
    tce_values:  list[float]     = []

    for p in test_personas:
        # Run turns through engine
        for turn in p.turns:
            score, tier = engine.update(
                user_id=p.user_id,
                ic_score=turn.ic_score,
                valence=turn.valence,
                arousal=turn.arousal,
                behavior_vec=turn.behavior_vec,
                session_boundary=turn.session_boundary,
            )

        t_computed = engine.get_trust(p.user_id)
        tier_str   = engine.get_trust_tier(p.user_id).value
        tce_val    = abs(p.t_human - t_computed)
        tce_values.append(tce_val)

        tce_results.append(TCEResult(
            user_id=p.user_id,
            archetype=p.archetype,
            t_human=round(p.t_human, 4),
            t_computed=round(t_computed, 4),
            tce=round(tce_val, 4),
            final_tier=tier_str,
            notes=p.notes,
        ))

    # Save
    tce_path = os.path.join(results_dir, "exp5_tce.csv")
    with open(tce_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user_id","archetype","t_human","t_computed","tce","final_tier","notes"
        ])
        writer.writeheader()
        for r in tce_results:
            writer.writerow(asdict(r))
    print(f"  Saved {len(tce_results)} TCE records → {tce_path}")

    summary = {
        "n_personas":   len(tce_results),
        "tce_mean":     float(np.mean(tce_values)),
        "tce_std":      float(np.std(tce_values)),
        "tce_median":   float(np.median(tce_values)),
        "tce_max":      float(np.max(tce_values)),
        "pct_within_0.10": float(np.mean(np.array(tce_values) < 0.10)),
        "pct_within_0.20": float(np.mean(np.array(tce_values) < 0.20)),
        "by_archetype": {}
    }
    for arch in SyntheticTrustDataset.ARCHETYPES:
        arch_vals = [r.tce for r in tce_results if r.archetype == arch]
        if arch_vals:
            summary["by_archetype"][arch] = {
                "mean": round(float(np.mean(arch_vals)), 4),
                "std":  round(float(np.std(arch_vals)), 4),
                "n":    len(arch_vals),
            }

    print(f"\n  TCE mean  : {summary['tce_mean']:.4f}")
    print(f"  TCE std   : {summary['tce_std']:.4f}")
    print(f"  TCE median: {summary['tce_median']:.4f}")
    print(f"  Within 0.10: {summary['pct_within_0.10']*100:.1f}%")
    print(f"  Within 0.20: {summary['pct_within_0.20']*100:.1f}%")
    for arch, stats in summary["by_archetype"].items():
        print(f"  [{arch:25s}] TCE={stats['mean']:.4f}±{stats['std']:.4f} "
              f"(n={stats['n']})")

    return summary


# ============================================================================
# Helpers
# ============================================================================

def _save_conv_results(results: list[ConvResult], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save summary CSV (one row per conversation)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "conv_id","emotion_label","n_turns","trust_start","trust_end",
            "trust_min","trust_max","trust_mean","tier_transitions","final_tier"
        ])
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row.pop("turns")
            writer.writerow(row)

    # Save turn-level detail as JSON
    json_path = path.replace(".csv", "_turns.json")
    with open(json_path, "w") as f:
        json.dump(
            [{**asdict(r), "turns": [asdict(t) for t in r.turns]}
             for r in results],
            f, indent=2
        )
    print(f"  Saved → {path} + {json_path}")


def _print_summary(exp_name: str, results: list[ConvResult]):
    if not results:
        print(f"  [{exp_name}] No results.")
        return
    ends  = [r.trust_end for r in results]
    means = [r.trust_mean for r in results]
    transitions = [r.tier_transitions for r in results]
    tier_counts = {}
    for r in results:
        tier_counts[r.final_tier] = tier_counts.get(r.final_tier, 0) + 1
    print(f"  Conversations  : {len(results)}")
    print(f"  Trust end mean : {np.mean(ends):.3f} ± {np.std(ends):.3f}")
    print(f"  Trust mean avg : {np.mean(means):.3f}")
    print(f"  Tier transitions/conv: {np.mean(transitions):.2f}")
    print(f"  Final tier distribution: {tier_counts}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="all",
                        choices=["all","empathetic","persona","msc","adversarial","tce"])
    parser.add_argument("--ablation",  default="full",
                        choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--max_convs", type=int, default=200)
    parser.add_argument("--n_variants",type=int, default=10)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Build engine with ablation config
    cfg = ABLATION_CONFIGS[args.ablation]
    engine = TrustEngine(**cfg)
    print(f"\n{'='*60}")
    print(f"  TrustEngine Experiment Runner")
    print(f"  Ablation : {args.ablation}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Results  → {args.results_dir}/")
    print(f"{'='*60}")

    all_summaries = {"ablation": args.ablation, "experiments": {}}
    t0 = time.time()

    if args.dataset in ("all", "empathetic"):
        r = exp1_empathetic(engine, args.max_convs, args.results_dir)
        all_summaries["experiments"]["exp1_empathetic"] = {
            "n_convs": len(r),
            "trust_end_mean": float(np.mean([x.trust_end for x in r]))
        }

    if args.dataset in ("all", "persona"):
        engine2 = TrustEngine(**cfg)   # fresh engine per experiment
        r = exp2_persona(engine2, args.max_convs, args.results_dir)
        all_summaries["experiments"]["exp2_persona"] = {
            "n_convs": len(r),
            "trust_end_mean": float(np.mean([x.trust_end for x in r]))
        }

    if args.dataset in ("all", "msc"):
        engine3 = TrustEngine(**cfg)
        r = exp3_msc(engine3, max_speakers=min(args.max_convs, 50), results_dir=args.results_dir)
        all_summaries["experiments"]["exp3_msc"] = {
            "n_users": len(r),
            "trust_end_mean": float(np.mean([x.trust_end for x in r]))
        }

    if args.dataset in ("all", "adversarial"):
        engine4 = TrustEngine(**cfg)
        summary = exp4_adversarial(engine4, args.n_variants, args.results_dir)
        all_summaries["experiments"]["exp4_adversarial"] = summary

    if args.dataset in ("all", "tce"):
        engine5 = TrustEngine(**cfg)
        summary = exp5_tce(engine5, args.n_variants, args.results_dir)
        all_summaries["experiments"]["exp5_tce"] = summary

    elapsed = time.time() - t0
    all_summaries["runtime_seconds"] = round(elapsed, 2)

    summary_path = os.path.join(args.results_dir,
                                f"summary_{args.ablation}.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[DONE] Total runtime: {elapsed:.1f}s")
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
