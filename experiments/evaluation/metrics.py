"""
evaluation/metrics.py
=====================
Computes all evaluation metrics for the T3 Trust Engine paper:

  1. TCE  — Trust Calibration Error  (primary T3 metric)
  2. TierAccuracy  — % of personas where final_tier matches T_human tier
  3. SeparationScore  — mean(coop_trust) - mean(adv_trust)
  4. Stability  — std of trust within single emotion-type conversations
  5. AblationTable  — formatted comparison across ablation configs

Usage
-----
    from evaluation.metrics import TrustMetrics

    m = TrustMetrics(results_dir="results")
    m.compute_all()
    m.print_paper_table()
"""

from __future__ import annotations
import os, sys, json, csv
import numpy as np
from dataclasses import dataclass
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@dataclass
class MetricSummary:
    ablation:         str
    tce_mean:         float
    tce_std:          float
    tier_accuracy:    float
    separation_score: float
    stability:        float
    n_tce_personas:   int


class TrustMetrics:

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir

    # -----------------------------------------------------------------------
    # 1. TCE
    # -----------------------------------------------------------------------
    def compute_tce(self, tce_csv: str) -> dict:
        """
        TCE = (1/N) * Σ|T_computed - T_human|
        Returns mean, std, median, per-archetype breakdown.
        """
        path = os.path.join(self.results_dir, tce_csv)
        if not os.path.exists(path):
            return {}

        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        tce_vals = [float(r["tce"]) for r in rows]
        archetypes = {}
        for r in rows:
            arch = r["archetype"]
            archetypes.setdefault(arch, []).append(float(r["tce"]))

        result = {
            "n": len(tce_vals),
            "tce_mean":   round(float(np.mean(tce_vals)), 4),
            "tce_std":    round(float(np.std(tce_vals)),  4),
            "tce_median": round(float(np.median(tce_vals)), 4),
            "tce_max":    round(float(np.max(tce_vals)), 4),
            "within_0.10": round(float(np.mean(np.array(tce_vals) < 0.10)), 3),
            "within_0.20": round(float(np.mean(np.array(tce_vals) < 0.20)), 3),
            "by_archetype": {
                arch: {
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)), 4),
                    "n":    len(vals),
                }
                for arch, vals in archetypes.items()
            }
        }
        return result

    # -----------------------------------------------------------------------
    # 2. Tier Accuracy
    # -----------------------------------------------------------------------
    def compute_tier_accuracy(self, tce_csv: str) -> float:
        """
        % of personas where model's final_tier matches
        the tier implied by T_human.
        """
        path = os.path.join(self.results_dir, tce_csv)
        if not os.path.exists(path):
            return 0.0

        def score_to_tier(t: float) -> str:
            if t >= 0.80: return "HIGH"
            if t >= 0.50: return "MEDIUM"
            return "LOW"

        correct = 0
        total   = 0
        with open(path) as f:
            for row in csv.DictReader(f):
                human_tier = score_to_tier(float(row["t_human"]))
                model_tier = row["final_tier"]
                correct += int(human_tier == model_tier)
                total   += 1

        return round(correct / max(total, 1), 4)

    # -----------------------------------------------------------------------
    # 3. Separation Score (EXP-4)
    # -----------------------------------------------------------------------
    def compute_separation(self, adv_csv: str) -> dict:
        path = os.path.join(self.results_dir, adv_csv)
        if not os.path.exists(path):
            return {}

        coop, adv = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                arch = row["emotion_label"]
                end  = float(row["trust_end"])
                if arch in ("COOPERATIVE_STABLE", "COOPERATIVE_GROWING"):
                    coop.append(end)
                elif arch == "ADVERSARIAL":
                    adv.append(end)

        return {
            "cooperative_mean": round(float(np.mean(coop)), 4) if coop else 0.0,
            "cooperative_std":  round(float(np.std(coop)),  4) if coop else 0.0,
            "adversarial_mean": round(float(np.mean(adv)),  4) if adv else 0.0,
            "adversarial_std":  round(float(np.std(adv)),   4) if adv else 0.0,
            "separation":       round(
                float(np.mean(coop)) - float(np.mean(adv)), 4
            ) if (coop and adv) else 0.0,
        }

    # -----------------------------------------------------------------------
    # 4. Stability — within-emotion trust variance
    # -----------------------------------------------------------------------
    def compute_stability(self, exp1_json: str) -> dict:
        path = os.path.join(self.results_dir, exp1_json)
        if not os.path.exists(path):
            return {}

        with open(path) as f:
            data = json.load(f)

        by_emotion: dict[str, list[float]] = {}
        for conv in data:
            emotion = conv["emotion_label"]
            scores = [t["trust_score"] for t in conv["turns"]]
            if scores:
                by_emotion.setdefault(emotion, []).extend(scores)

        result = {}
        for emotion, scores in by_emotion.items():
            result[emotion] = {
                "mean": round(float(np.mean(scores)), 4),
                "std":  round(float(np.std(scores)),  4),
                "n":    len(scores),
            }

        overall_stds = [v["std"] for v in result.values() if v["n"] >= 3]
        result["_overall_mean_std"] = round(
            float(np.mean(overall_stds)), 4
        ) if overall_stds else 0.0

        return result

    # -----------------------------------------------------------------------
    # 5. Full ablation comparison table
    # -----------------------------------------------------------------------
    def compute_ablation_table(self,
                               ablation_names: list[str]) -> list[dict]:
        rows = []
        for abl in ablation_names:
            summary_path = os.path.join(
                self.results_dir, f"summary_{abl}.json"
            )
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as f:
                s = json.load(f)

            tce_data = s.get("experiments", {}).get("exp5_tce", {})
            adv_data = s.get("experiments", {}).get("exp4_adversarial", {})
            exp1_data = s.get("experiments", {}).get("exp1_empathetic", {})

            rows.append({
                "ablation":       abl,
                "tce_mean":       tce_data.get("tce_mean", "—"),
                "tce_std":        tce_data.get("tce_std",  "—"),
                "within_0.10":    tce_data.get("pct_within_0.10", "—"),
                "separation":     adv_data.get("separation", "—"),
                "emp_trust_mean": exp1_data.get("trust_end_mean", "—"),
            })
        return rows

    # -----------------------------------------------------------------------
    # Print paper-ready tables
    # -----------------------------------------------------------------------
    def print_paper_table(self, ablation_names: Optional[list[str]] = None):
        print("\n" + "="*70)
        print("  T3 TRUST ENGINE — PAPER RESULTS TABLE")
        print("="*70)

        # TCE Table
        tce = self.compute_tce("exp5_tce.csv")
        if tce:
            print("\n── Table 1: Trust Calibration Error (TCE) ──")
            print(f"  N personas  : {tce['n']}")
            print(f"  TCE mean    : {tce['tce_mean']:.4f} ± {tce['tce_std']:.4f}")
            print(f"  TCE median  : {tce['tce_median']:.4f}")
            print(f"  Within 0.10 : {tce['within_0.10']*100:.1f}%")
            print(f"  Within 0.20 : {tce['within_0.20']*100:.1f}%")
            print(f"\n  Per-archetype breakdown:")
            print(f"  {'Archetype':28s}  {'TCE mean':>10}  {'TCE std':>9}  n")
            print(f"  {'-'*60}")
            for arch, stats in tce.get("by_archetype", {}).items():
                print(f"  {arch:28s}  {stats['mean']:>10.4f}  "
                      f"{stats['std']:>9.4f}  {stats['n']}")

        # Tier Accuracy
        tier_acc = self.compute_tier_accuracy("exp5_tce.csv")
        print(f"\n── Tier Accuracy: {tier_acc*100:.1f}%")

        # Separation
        sep = self.compute_separation("exp4_adversarial.csv")
        if sep:
            print(f"\n── Table 2: Cooperative vs Adversarial Separation ──")
            print(f"  Cooperative trust: {sep['cooperative_mean']:.3f} "
                  f"± {sep['cooperative_std']:.3f}")
            print(f"  Adversarial trust: {sep['adversarial_mean']:.3f} "
                  f"± {sep['adversarial_std']:.3f}")
            print(f"  Separation gap   : {sep['separation']:.3f}")

        # Ablation table
        if ablation_names:
            table = self.compute_ablation_table(ablation_names)
            if table:
                print(f"\n── Table 3: Ablation Comparison ──")
                hdr = f"  {'Ablation':22s}  {'TCE↓':>7}  {'±':>6}  "
                hdr += f"{'<0.10':>7}  {'Sep↑':>7}  {'EmpTrust':>10}"
                print(hdr)
                print(f"  {'-'*70}")
                for row in table:
                    def fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
                    print(f"  {row['ablation']:22s}  "
                          f"{fmt(row['tce_mean']):>7}  "
                          f"{fmt(row['tce_std']):>6}  "
                          f"{fmt(row['within_0.10']):>7}  "
                          f"{fmt(row['separation']):>7}  "
                          f"{fmt(row['emp_trust_mean']):>10}")

        print("\n" + "="*70)

    def compute_all(self):
        tce      = self.compute_tce("exp5_tce.csv")
        tier_acc = self.compute_tier_accuracy("exp5_tce.csv")
        sep      = self.compute_separation("exp4_adversarial.csv")

        result = {
            "tce":          tce,
            "tier_accuracy": tier_acc,
            "separation":   sep,
        }
        out = os.path.join(self.results_dir, "metrics_summary.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Metrics saved → {out}")
        return result


if __name__ == "__main__":
    m = TrustMetrics(results_dir=os.path.join(ROOT, "results"))
    m.compute_all()
    m.print_paper_table(ablation_names=["full", "no_emotion_weight",
                                        "no_trust_gate", "high_decay"])
