"""
run_experiments_v2.py
=====================
Upgraded experiment runner for T3 Trust Engine — paper-grade evaluation.

Changes from v1:
  - 200 personas (40 per archetype)
  - Full ablation study (4 configs)
  - Seed control for reproducibility
  - Clean CSV + JSON outputs
  - Generates all 4 publication figures
  - Suppresses per-turn print spam (logging only)

Usage:
    python run_experiments_v2.py                  # full run
    python run_experiments_v2.py --quick          # 20 personas, no plots
"""

from __future__ import annotations

import sys, os, json, csv, argparse, time, logging
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Silence sub-module debug noise ──────────────────────────────────────────
logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s  %(name)s — %(message)s")

from trust_engine import TrustEngine, TrustTier
from data.synthetic_trust_dataset import SyntheticTrustDataset

SEED = 42
np.random.seed(SEED)

# ============================================================================
# Ablation configurations
# ============================================================================

ABLATION_CONFIGS: Dict[str, dict] = {
    "full_model": dict(
        decay=0.95, w1=0.55, w2=0.20, w3=0.25,
        alpha_bc=0.10, alpha_es=0.30,
        disable_bc=False, disable_es=False,
        disable_hr=False, disable_ic_gate=False,
    ),
    "no_trust_gate": dict(
        decay=0.95, w1=0.55, w2=0.20, w3=0.25,
        alpha_bc=0.10, alpha_es=0.30,
        disable_bc=False, disable_es=False,
        disable_hr=False, disable_ic_gate=True,
    ),
    "no_behavior": dict(
        decay=0.95, w1=0.00, w2=0.60, w3=0.40,
        alpha_bc=0.10, alpha_es=0.30,
        disable_bc=True, disable_es=False,
        disable_hr=False, disable_ic_gate=False,
    ),
    "no_emotion": dict(
        decay=0.95, w1=0.70, w2=0.00, w3=0.30,
        alpha_bc=0.10, alpha_es=0.30,
        disable_bc=False, disable_es=True,
        disable_hr=False, disable_ic_gate=False,
    ),
}

# ============================================================================
# Data containers
# ============================================================================

@dataclass
class TCERecord:
    user_id:    str
    archetype:  str
    t_human:    float
    t_computed: float
    tce:        float
    final_tier: str

@dataclass
class AblationRow:
    config:           str
    tce_mean:         float
    tce_std:          float
    tce_median:       float
    separation:       float
    coop_trust_mean:  float
    adv_trust_mean:   float
    pct_within_010:   float
    pct_within_020:   float

# ============================================================================
# Core experiment: TCE + separation on one engine config
# ============================================================================

def run_evaluation(
    config_name: str,
    engine_cfg:  dict,
    personas,
    results_dir: str,
) -> Tuple[AblationRow, List[TCERecord], Dict[str, List[float]]]:
    """
    Run all personas through the engine config.
    Returns (AblationRow summary, TCE records, trust trajectories).
    """
    engine = TrustEngine(**engine_cfg)

    tce_records: List[TCERecord] = []
    trajectories: Dict[str, List[float]] = {}   # user_id -> [trust per turn]
    coop_finals, adv_finals = [], []

    for p in personas:
        traj = []
        for turn in p.turns:
            score, _ = engine.update(
                user_id          = p.user_id,
                ic_score         = turn.ic_score,
                valence          = turn.valence,
                arousal          = turn.arousal,
                behavior_vec     = turn.behavior_vec,
                session_boundary = turn.session_boundary,
            )
            traj.append(score)

        t_computed = engine.get_trust(p.user_id)
        tce_val    = abs(p.t_human - t_computed)
        tier_str   = engine.get_trust_tier(p.user_id).value

        tce_records.append(TCERecord(
            user_id    = p.user_id,
            archetype  = p.archetype,
            t_human    = round(p.t_human, 4),
            t_computed = round(t_computed, 4),
            tce        = round(tce_val, 4),
            final_tier = tier_str,
        ))
        trajectories[p.user_id] = traj

        if p.archetype in ("COOPERATIVE_STABLE", "COOPERATIVE_GROWING"):
            coop_finals.append(t_computed)
        elif p.archetype == "ADVERSARIAL":
            adv_finals.append(t_computed)

    # Save TCE CSV
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"tce_{config_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id","archetype","t_human",
                                          "t_computed","tce","final_tier"])
        w.writeheader()
        for r in tce_records:
            w.writerow(asdict(r))

    # Compute summary metrics
    tce_vals   = [r.tce for r in tce_records]
    separation = (np.mean(coop_finals) - np.mean(adv_finals)
                  if coop_finals and adv_finals else 0.0)

    row = AblationRow(
        config           = config_name,
        tce_mean         = float(np.mean(tce_vals)),
        tce_std          = float(np.std(tce_vals)),
        tce_median       = float(np.median(tce_vals)),
        separation       = float(separation),
        coop_trust_mean  = float(np.mean(coop_finals)) if coop_finals else 0.0,
        adv_trust_mean   = float(np.mean(adv_finals))  if adv_finals  else 0.0,
        pct_within_010   = float(np.mean(np.array(tce_vals) < 0.10)),
        pct_within_020   = float(np.mean(np.array(tce_vals) < 0.20)),
    )

    return row, tce_records, trajectories


# ============================================================================
# Figures
# ============================================================================

def make_figures(
    all_rows:        List[AblationRow],
    tce_records_full: List[TCERecord],
    trajectories:    Dict[str, List[float]],
    personas,
    results_dir:     str,
) -> None:
    """Generate all 4 publication-quality figures."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Paper style
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.linewidth":   0.8,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "grid.alpha":       0.3,
        "grid.linewidth":   0.5,
        "lines.linewidth":  1.6,
        "legend.framealpha":0.9,
        "legend.fontsize":  9,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
    })

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Fig 1: Trust vs Time (coop vs adversarial) ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    coop_trajs = [trajectories[p.user_id] for p in personas
                  if p.archetype == "COOPERATIVE_STABLE" and
                     p.user_id in trajectories][:8]
    adv_trajs  = [trajectories[p.user_id] for p in personas
                  if p.archetype == "ADVERSARIAL" and
                     p.user_id in trajectories][:8]
    incon_trajs = [trajectories[p.user_id] for p in personas
                   if p.archetype == "INCONSISTENT" and
                      p.user_id in trajectories][:8]

    def plot_band(ax, trajs, color, label):
        if not trajs:
            return
        max_len = max(len(t) for t in trajs)
        padded  = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        arr     = np.array(padded)
        mean    = arr.mean(axis=0)
        std     = arr.std(axis=0)
        x       = np.arange(max_len)
        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, np.clip(mean - std, 0, 1),
                        np.clip(mean + std, 0, 1),
                        color=color, alpha=0.15)

    plot_band(ax, coop_trajs,  "#2E75B6", "Cooperative Stable")
    plot_band(ax, incon_trajs, "#ED7D31", "Inconsistent")
    plot_band(ax, adv_trajs,   "#C00000", "Adversarial")

    ax.axhline(0.80, color="#375623", lw=0.9, ls="--", label="HIGH tier threshold")
    ax.axhline(0.50, color="#7F6000", lw=0.9, ls=":",  label="MEDIUM tier threshold")
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Trust score T(u, t)")
    ax.set_title("Fig 1 — Trust Score Evolution by User Archetype",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, axis="y")
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "fig1_trust_vs_time.pdf")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.savefig(p1.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 1 → {p1}")

    # ── Fig 2: Ablation comparison bar chart ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    labels  = [r.config.replace("_", "\n") for r in all_rows]
    tce_m   = [r.tce_mean       for r in all_rows]
    tce_e   = [r.tce_std        for r in all_rows]
    sep     = [r.separation     for r in all_rows]
    colors  = ["#2E75B6","#ED7D31","#A5A5A5","#FFC000"]

    x = np.arange(len(labels))
    axes[0].bar(x, tce_m, yerr=tce_e, capsize=4, color=colors, width=0.55,
                error_kw={"linewidth": 1.0})
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("TCE (↓ better)"); axes[0].set_title("TCE by Ablation Config",
                                                              fontweight="bold")
    axes[0].grid(True, axis="y")

    axes[1].bar(x, sep, color=colors, width=0.55)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Separation (↑ better)")
    axes[1].set_title("Cooperative–Adversarial Separation", fontweight="bold")
    axes[1].grid(True, axis="y")

    fig.suptitle("Fig 2 — Ablation Study: TCE and Separation Score",
                 fontweight="bold", fontsize=11, y=1.01)
    plt.tight_layout()
    p2 = os.path.join(fig_dir, "fig2_ablation.pdf")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.savefig(p2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 2 → {p2}")

    # ── Fig 3: TCE distribution histogram ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    archetypes    = ["COOPERATIVE_STABLE","COOPERATIVE_GROWING",
                     "ADVERSARIAL","INCONSISTENT","NEW_THEN_DROPS"]
    arch_colors   = ["#2E75B6","#70AD47","#C00000","#ED7D31","#7030A0"]
    arch_labels   = ["Coop. Stable","Coop. Growing",
                     "Adversarial","Inconsistent","New→Drop"]

    for arch, color, label in zip(archetypes, arch_colors, arch_labels):
        vals = [r.tce for r in tce_records_full if r.archetype == arch]
        if vals:
            ax.hist(vals, bins=15, alpha=0.55, color=color,
                    label=f"{label} (n={len(vals)})", density=True)

    ax.axvline(0.10, color="black", lw=1.2, ls="--", label="TCE=0.10 target")
    ax.axvline(0.20, color="gray",  lw=1.0, ls=":",  label="TCE=0.20 threshold")
    ax.set_xlabel("Trust Calibration Error (TCE = |T_computed − T_human|)")
    ax.set_ylabel("Density")
    ax.set_title("Fig 3 — TCE Distribution by Persona Archetype",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    plt.tight_layout()
    p3 = os.path.join(fig_dir, "fig3_tce_distribution.pdf")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.savefig(p3.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 3 → {p3}")

    # ── Fig 4: Coop vs Adv trust means across ablations ──────────────────
    fig, ax = plt.subplots(figsize=(8, 4))

    x       = np.arange(len(all_rows))
    width   = 0.35
    coops   = [r.coop_trust_mean for r in all_rows]
    advs    = [r.adv_trust_mean  for r in all_rows]
    labels  = [r.config.replace("_", "\n") for r in all_rows]

    b1 = ax.bar(x - width/2, coops, width, color="#2E75B6", label="Cooperative mean trust")
    b2 = ax.bar(x + width/2, advs,  width, color="#C00000", label="Adversarial mean trust")

    for bar, val in [(b1, coops), (b2, advs)]:
        for rect, v in zip(bar, val):
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean final trust score")
    ax.set_title("Fig 4 — Cooperative vs Adversarial Trust by Ablation",
                 fontweight="bold", fontsize=11)
    ax.legend(); ax.grid(True, axis="y"); ax.set_ylim(0, 1.15)
    plt.tight_layout()
    p4 = os.path.join(fig_dir, "fig4_coop_vs_adv.pdf")
    plt.savefig(p4, dpi=150, bbox_inches="tight")
    plt.savefig(p4.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 4 → {p4}")


# ============================================================================
# Print paper table
# ============================================================================

def print_paper_table(rows: List[AblationRow], by_arch: Dict) -> None:
    print("\n" + "=" * 78)
    print("  ABLATION STUDY — RESULTS TABLE")
    print("=" * 78)
    print(f"  {'Config':<18} {'TCE↓':>8} {'±':>6} {'Median':>8} "
          f"{'Sep↑':>8} {'Coop':>7} {'Adv':>7} {'<0.10':>7} {'<0.20':>7}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r.config:<18} {r.tce_mean:>8.4f} {r.tce_std:>6.4f} "
              f"{r.tce_median:>8.4f} {r.separation:>8.4f} "
              f"{r.coop_trust_mean:>7.3f} {r.adv_trust_mean:>7.3f} "
              f"{r.pct_within_010:>7.1%} {r.pct_within_020:>7.1%}")
    print("=" * 78)

    print("\n  TCE BREAKDOWN BY ARCHETYPE (full_model)")
    print("  " + "-" * 50)
    for arch, stats in by_arch.items():
        bar = "█" * int(stats["mean"] * 20)
        print(f"  {arch:<25}  {stats['mean']:.4f} ± {stats['std']:.4f}  "
              f"n={stats['n']:3d}  {bar}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",    action="store_true",
                        help="Fast mode: 20 personas, skip figures")
    parser.add_argument("--results_dir", default="results_v2")
    args = parser.parse_args()

    n_variants = 4 if args.quick else 40   # 40 × 5 archetypes = 200 personas
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  T3 Trust Engine — v2 Experiment Runner")
    print(f"  Personas : {n_variants * 5}  (n_variants={n_variants})")
    print(f"  Ablations: {len(ABLATION_CONFIGS)}")
    print(f"  Results  → {results_dir}/")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Generate personas ONCE (same seed → reproducible across ablations)
    print("[DATA] Generating synthetic personas...")
    ds       = SyntheticTrustDataset(n_variants_per_archetype=n_variants, seed=SEED)
    personas = ds.generate()
    print(f"  Generated {len(personas)} personas across {len(ds.ARCHETYPES)} archetypes")

    all_rows:   List[AblationRow]   = []
    tce_full:   List[TCERecord]     = []
    trajs_full: Dict[str, List[float]] = {}

    for cfg_name, cfg in ABLATION_CONFIGS.items():
        print(f"\n[ABLATION] {cfg_name} ...")
        row, tce_records, trajectories = run_evaluation(
            config_name = cfg_name,
            engine_cfg  = cfg,
            personas    = personas,
            results_dir = results_dir,
        )
        all_rows.append(row)
        print(f"  TCE={row.tce_mean:.4f}±{row.tce_std:.4f}  "
              f"Separation={row.separation:.4f}  "
              f"Coop={row.coop_trust_mean:.3f}  Adv={row.adv_trust_mean:.3f}")

        if cfg_name == "full_model":
            tce_full   = tce_records
            trajs_full = trajectories

    # Per-archetype breakdown for full_model
    archetypes   = list(ds.ARCHETYPES.keys())
    by_arch_full = {}
    for arch in archetypes:
        vals = [r.tce for r in tce_full if r.archetype == arch]
        if vals:
            by_arch_full[arch] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "n":    len(vals),
            }

    # ── Console table ────────────────────────────────────────────────────
    print_paper_table(all_rows, by_arch_full)

    # ── Ablation CSV ─────────────────────────────────────────────────────
    abl_path = os.path.join(results_dir, "ablation_table.csv")
    with open(abl_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(AblationRow.__dataclass_fields__))
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))
    print(f"[SAVED] Ablation table → {abl_path}")

    # ── Master JSON ──────────────────────────────────────────────────────
    summary = {
        "experiment_version": "v2",
        "n_personas":         len(personas),
        "seed":               SEED,
        "ablation_results":   [asdict(r) for r in all_rows],
        "full_model": {
            "tce_by_archetype": by_arch_full,
            "pct_within_010":   float(np.mean([r.tce < 0.10 for r in tce_full])),
            "pct_within_020":   float(np.mean([r.tce < 0.20 for r in tce_full])),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    json_path = os.path.join(results_dir, "summary_v2.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Master summary → {json_path}")

    # ── Figures ──────────────────────────────────────────────────────────
    if not args.quick:
        print("\n[FIGURES] Generating publication plots...")
        make_figures(all_rows, tce_full, trajs_full, personas, results_dir)

    print(f"\n[DONE]  Runtime: {time.time() - t0:.1f}s")
    print(f"        Results → {results_dir}/")
    print(f"        Figures → {results_dir}/figures/  (4 PDFs + PNGs)")


if __name__ == "__main__":
    main()
