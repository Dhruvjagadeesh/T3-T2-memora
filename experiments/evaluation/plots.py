"""
evaluation/plots.py
===================
Generates paper-ready visualizations for T3 Trust Engine experiments.

Figures produced
----------------
Fig 1 — Trust trajectory by emotion type (EXP-1)
Fig 2 — Cooperative vs Adversarial trust evolution (EXP-4)
Fig 3 — Multi-session HR accumulation (EXP-3)
Fig 4 — TCE distribution by archetype (EXP-5)
Fig 5 — Ablation comparison bar chart (Table 3 visual)

All outputs saved to results/figures/ as PDF + PNG (300 DPI).
"""

from __future__ import annotations
import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Paper style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

TIER_COLORS = {"HIGH": "#2ECC71", "MEDIUM": "#F39C12", "LOW": "#E74C3C"}
ARCHETYPE_COLORS = {
    "COOPERATIVE_STABLE":  "#2980B9",
    "COOPERATIVE_GROWING": "#27AE60",
    "ADVERSARIAL":         "#E74C3C",
    "INCONSISTENT":        "#E67E22",
    "NEW_THEN_DROPS":      "#8E44AD",
}


# ============================================================================
# Fig 1 — Trust trajectory by emotion (EXP-1)
# ============================================================================

def plot_emotion_trajectories(json_path: str, top_n_emotions: int = 6):
    if not os.path.exists(json_path):
        print(f"[plots] File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    # Group by emotion, compute mean trust trajectory
    emotion_data: dict[str, list[list[float]]] = {}
    for conv in data:
        em = conv["emotion_label"]
        scores = [t["trust_score"] for t in conv["turns"]]
        emotion_data.setdefault(em, []).append(scores)

    # Pick top_n emotions by frequency
    top_emotions = sorted(emotion_data, key=lambda e: len(emotion_data[e]),
                          reverse=True)[:top_n_emotions]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    axes = axes.flatten()

    for ax, emotion in zip(axes, top_emotions):
        trajectories = emotion_data[emotion]
        max_len = max(len(t) for t in trajectories)
        # Pad shorter trajectories
        padded = [t + [t[-1]] * (max_len - len(t)) for t in trajectories]
        arr = np.array(padded)   # shape (n_convs, max_len)

        xs = np.arange(max_len)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)

        ax.plot(xs, mean, color="#2C3E50", linewidth=2, label="Mean")
        ax.fill_between(xs, mean - std, mean + std,
                        alpha=0.2, color="#2C3E50", label="±1 std")
        ax.axhline(0.80, color=TIER_COLORS["HIGH"],   linestyle="--",
                   linewidth=0.8, alpha=0.7, label="HIGH tier")
        ax.axhline(0.50, color=TIER_COLORS["MEDIUM"], linestyle="--",
                   linewidth=0.8, alpha=0.7, label="MEDIUM tier")
        ax.set_title(f"{emotion} (n={len(trajectories)})")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Trust Score T(u,t)")
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fig 1 — Trust Score Trajectory by Emotion Type (EmpatheticDialogues)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "fig1_emotion_trajectories")


# ============================================================================
# Fig 2 — Cooperative vs Adversarial (EXP-4)
# ============================================================================

def plot_adversarial_comparison(json_path: str):
    if not os.path.exists(json_path):
        print(f"[plots] File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    archetype_trajectories: dict[str, list[list[float]]] = {}
    for conv in data:
        arch = conv["emotion_label"]   # archetype stored here for synthetic
        scores = [t["trust_score"] for t in conv["turns"]]
        archetype_trajectories.setdefault(arch, []).append(scores)

    fig, ax = plt.subplots(figsize=(9, 5))

    for arch, trajs in archetype_trajectories.items():
        if not trajs:
            continue
        color = ARCHETYPE_COLORS.get(arch, "#95A5A6")
        max_len = max(len(t) for t in trajs)
        padded  = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        arr  = np.array(padded)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        xs   = np.arange(max_len)
        ax.plot(xs, mean, color=color, linewidth=2,
                label=f"{arch} (n={len(trajs)})")
        ax.fill_between(xs, mean - std, mean + std, alpha=0.12, color=color)

    ax.axhline(0.80, color="#2ECC71", linestyle="--", linewidth=1, alpha=0.6,
               label="HIGH tier boundary")
    ax.axhline(0.50, color="#F39C12", linestyle="--", linewidth=1, alpha=0.6,
               label="MEDIUM tier boundary")
    ax.set_xlabel("Turn (across all sessions)")
    ax.set_ylabel("Trust Score T(u,t)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig 2 — Cooperative vs Adversarial Trust Evolution",
                 fontweight="bold")
    ax.legend(loc="right", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig2_adversarial_comparison")


# ============================================================================
# Fig 3 — Multi-session HR accumulation (EXP-3)
# ============================================================================

def plot_multisession_trust(json_path: str, n_users: int = 8):
    if not os.path.exists(json_path):
        print(f"[plots] File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.cm.Blues
    for i, conv in enumerate(data[:n_users]):
        scores = [t["trust_score"] for t in conv["turns"]]
        color  = cmap(0.4 + 0.6 * i / max(n_users - 1, 1))
        ax.plot(scores, color=color, linewidth=1.5, alpha=0.8,
                label=f"User {i+1}" if i < 5 else None)

        # Mark session boundaries
        for j, turn in enumerate(conv["turns"]):
            if turn.get("session_boundary") and j > 0:
                ax.axvline(x=j, color=color, linestyle=":", alpha=0.4,
                           linewidth=0.8)

    ax.axhline(0.80, color="#2ECC71", linestyle="--", linewidth=1,
               alpha=0.6, label="HIGH boundary")
    ax.axhline(0.50, color="#F39C12", linestyle="--", linewidth=1,
               alpha=0.6, label="MEDIUM boundary")
    ax.set_xlabel("Turn (all sessions concatenated)")
    ax.set_ylabel("Trust Score T(u,t)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig 3 — Multi-Session Trust & HR Accumulation (MSC)",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig3_multisession_trust")


# ============================================================================
# Fig 4 — TCE Distribution by Archetype (EXP-5)
# ============================================================================

def plot_tce_distribution(tce_csv: str):
    results_dir = os.path.dirname(tce_csv)
    if not os.path.exists(tce_csv):
        print(f"[plots] File not found: {tce_csv}")
        return

    by_arch: dict[str, list[float]] = {}
    with open(tce_csv) as f:
        for row in csv.DictReader(f):
            arch = row["archetype"]
            by_arch.setdefault(arch, []).append(float(row["tce"]))

    archetypes = list(by_arch.keys())
    fig, axes = plt.subplots(1, len(archetypes),
                             figsize=(3 * len(archetypes), 4),
                             sharey=True)
    if len(archetypes) == 1:
        axes = [axes]

    for ax, arch in zip(axes, archetypes):
        vals = by_arch[arch]
        color = ARCHETYPE_COLORS.get(arch, "#95A5A6")
        ax.hist(vals, bins=min(10, len(vals)), color=color, alpha=0.8,
                edgecolor="white")
        ax.axvline(np.mean(vals), color="#2C3E50", linestyle="--",
                   linewidth=1.5, label=f"μ={np.mean(vals):.3f}")
        ax.set_title(arch.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("TCE")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle("Fig 4 — TCE Distribution by Persona Archetype",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig4_tce_distribution")


# ============================================================================
# Fig 5 — Ablation comparison (bar chart)
# ============================================================================

def plot_ablation_bars(ablation_summaries: list[dict]):
    """
    ablation_summaries: list of dicts with keys
      ablation, tce_mean, tce_std, separation, tier_accuracy
    """
    if not ablation_summaries:
        return

    labels    = [d["ablation"] for d in ablation_summaries]
    tce_means = [d.get("tce_mean", 0) for d in ablation_summaries]
    tce_stds  = [d.get("tce_std",  0) for d in ablation_summaries]
    seps      = [d.get("separation", 0) for d in ablation_summaries]

    x = np.arange(len(labels))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # TCE (lower is better)
    bars1 = ax1.bar(x, tce_means, w, yerr=tce_stds, capsize=4,
                    color="#2E86C1", alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("TCE (↓ lower is better)")
    ax1.set_title("Trust Calibration Error", fontweight="bold")
    ax1.set_ylim(0, max(tce_means) * 1.4 if tce_means else 1.0)
    for bar, val in zip(bars1, tce_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Separation (higher is better)
    bars2 = ax2.bar(x, seps, w, color="#27AE60", alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Separation Score (↑ higher is better)")
    ax2.set_title("Cooperative vs Adversarial Separation", fontweight="bold")
    ax2.set_ylim(0, max(seps) * 1.4 if seps else 1.0)
    for bar, val in zip(bars2, seps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Fig 5 — Ablation Study: T3 Component Contributions",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig5_ablation_bars")


# ============================================================================
# Helper
# ============================================================================

def _save(fig, name: str):
    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ============================================================================
# Generate all figures at once
# ============================================================================

def generate_all_figures(results_dir: str = None):
    if results_dir is None:
        results_dir = os.path.join(ROOT, "results")

    print("[plots] Generating all paper figures...")

    plot_emotion_trajectories(
        os.path.join(results_dir, "exp1_empathetic_turns.json"))

    plot_adversarial_comparison(
        os.path.join(results_dir, "exp4_adversarial_turns.json"))

    plot_multisession_trust(
        os.path.join(results_dir, "exp3_msc_turns.json"))

    plot_tce_distribution(
        os.path.join(results_dir, "exp5_tce.csv"))

    # Try to load ablation summaries
    ablation_summaries = []
    for abl in ["full", "no_emotion_weight", "no_trust_gate",
                "high_decay", "low_decay"]:
        path = os.path.join(results_dir, f"summary_{abl}.json")
        if os.path.exists(path):
            with open(path) as f:
                s = json.load(f)
            tce_data = s.get("experiments", {}).get("exp5_tce", {})
            adv_data = s.get("experiments", {}).get("exp4_adversarial", {})
            ablation_summaries.append({
                "ablation":   abl,
                "tce_mean":   tce_data.get("tce_mean", 0),
                "tce_std":    tce_data.get("tce_std", 0),
                "separation": adv_data.get("separation", 0),
            })

    if ablation_summaries:
        plot_ablation_bars(ablation_summaries)

    print(f"[plots] All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
