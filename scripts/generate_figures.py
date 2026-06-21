#!/usr/bin/env python3
"""
Generate all paper figures from experiment results.
Outputs: figures/fig1_outcomes.pdf ... fig5_track_comparison.pdf
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Data ──
PATTERNS_ORDER = [
    "bishop_bishop_knight",
    "bishop_knight_3pawns",
    "rook_bishop_pawn",
    "rook_knight_pawn",
    "bishop_6pawns",
    "knight_6pawns",
    "rook_4pawns",
]

LABELS = {
    "rook_bishop_pawn":     "R+B+P",
    "rook_knight_pawn":     "R+N+P",
    "bishop_bishop_knight": "B+B+N",
    "rook_4pawns":          "R+4P",
    "bishop_knight_3pawns": "B+N+3P",
    "bishop_6pawns":        "B+6P",
    "knight_6pawns":        "N+6P",
}

CONC = {
    "rook_bishop_pawn": "High", "rook_knight_pawn": "High",
    "bishop_bishop_knight": "High",
    "rook_4pawns": "Medium", "bishop_knight_3pawns": "Medium",
    "bishop_6pawns": "Low", "knight_6pawns": "Low",
}

CONC_COLOR = {"High": "#4A90D9", "Medium": "#F5A623", "Low": "#D0021B"}

RESULTS_DIR = Path("runs/results")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

def load_jsonl(fp):
    games = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games

def wdl(games):
    w = sum(1 for g in games if g["result"] == "win")
    d = sum(1 for g in games if g["result"] == "draw")
    l = sum(1 for g in games if g["result"] == "loss")
    return w, d, l

def score(w, d, l):
    t = w + d + l
    return (w + 0.5 * d) / t if t > 0 else 0

def ci95(w, d, l, n=50000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.array([1.0+w, 1.0+d, 1.0+l])
    s = rng.dirichlet(a, size=n)
    sc = s[:,0] + 0.5*s[:,1]
    return float(np.percentile(sc, 2.5)), float(np.percentile(sc, 97.5))

# Load data
track_a, track_b, track_bs = {}, {}, {}
for pat in PATTERNS_ORDER:
    fa = RESULTS_DIR / f"track_a_{pat}.jsonl"
    fb = RESULTS_DIR / f"track_b_{pat}.jsonl"
    fbs = RESULTS_DIR / f"track_b_stochastic_{pat}.jsonl"
    if fa.exists(): track_a[pat] = load_jsonl(fa)
    if fb.exists(): track_b[pat] = load_jsonl(fb)
    if fbs.exists(): track_bs[pat] = load_jsonl(fbs)


# ═════════════════════════════════════════════════════════
#  FIGURE 1: Stacked Horizontal Bar — Game Outcomes
# ═════════════════════════════════════════════════════════
def fig1():
    # Only Stochastic
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    
    colors_w = "#3FA055"   # Green
    colors_d = "#FFFACD"   # Light yellow
    colors_l = "#DB4C37"   # Red-orange
    
    data = track_bs
    labels = []
    w_counts, d_counts, l_counts = [], [], []
    
    for pat in PATTERNS_ORDER:
        if pat not in data: continue
        w, d, l = wdl(data[pat])
        labels.append(LABELS[pat])
        w_counts.append(w)
        d_counts.append(d)
        l_counts.append(l)
    
    y = np.arange(len(labels))
    
    ax.barh(y, w_counts, color=colors_w, height=0.5, label="Q-side wins")
    ax.barh(y, d_counts, left=w_counts, color=colors_d, height=0.5, label="Draw")
    left2 = [w+d for w, d in zip(w_counts, d_counts)]
    ax.barh(y, l_counts, left=left2, color=colors_l, height=0.5, label="Q-side loss")
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    
    # Remove x axis ticks
    ax.set_xticks([])
    ax.invert_yaxis()
    
    # Add text inside bars
    for idx in range(len(labels)):
        w, d, l = w_counts[idx], d_counts[idx], l_counts[idx]
        
        # White win text
        if w > 0:
            ax.text(w/2, y[idx], str(w), color="white", ha="center", va="center", fontsize=11)
        
        # Draw text
        if d > 0:
            ax.text(w + d/2, y[idx], str(d), color="gray", ha="center", va="center", fontsize=11)
            
        # Black win text
        if l > 0:
            ax.text(w + d + l/2, y[idx], str(l), color="white", ha="center", va="center", fontsize=11)

    # Legend at the top center
    handles, labels_leg = ax.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=3, fontsize=13, bbox_to_anchor=(0.5, 1.1), frameon=True)
    
    fig.text(0.5, -0.05, "Figure 1: Game Outcomes by Handicap Pattern (Lc0)", ha="center", va="top", fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_outcomes.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_outcomes.png", bbox_inches="tight")
    print("  ✓ Figure 1 saved")
    plt.close(fig)


# ═════════════════════════════════════════════════════════
#  FIGURE 2: Q-Score by Concentration Level
# ═════════════════════════════════════════════════════════
def fig2():
    # Only Stochastic
    fig, ax = plt.subplots(figsize=(6, 5))
    
    conc_levels = ["High\n(3 pieces)", "Medium\n(5 pieces)", "Low\n(7 pieces)"]
    conc_keys = ["High", "Medium", "Low"]
    
    sto_scores, sto_lo, sto_hi = [], [], []
    
    for ck in conc_keys:
        sg = [g for p in PATTERNS_ORDER if p in track_bs and CONC[p]==ck for g in track_bs[p]]
        w, d, l = wdl(sg)
        s = score(w, d, l)
        lo, hi = ci95(w, d, l)
        sto_scores.append(s); sto_lo.append(s-lo); sto_hi.append(hi-s)
    
    x = np.arange(len(conc_levels))
    width = 0.4
    
    bars2 = ax.bar(x, sto_scores, width,
                   yerr=[sto_lo, sto_hi], capsize=5,
                   color="#ED7D31", edgecolor="black", linewidth=0.8, zorder=3)
    
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(conc_levels)
    ax.set_ylabel("Q-Score")
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    
    # Value labels
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f"{h:.3f}",
               ha="center", va="bottom", fontsize=12)
    
    ax.set_title("Figure 2: Q-Score by Concentration Level (Lc0)", ha="center", va="top", fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_concentration.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_concentration.png", bbox_inches="tight")
    print("  ✓ Figure 2 saved")
    plt.close(fig)


# ═════════════════════════════════════════════════════════
#  FIGURE 3: Pairwise Heatmap
# ═════════════════════════════════════════════════════════
def fig3():
    avail = [p for p in PATTERNS_ORDER if p in track_bs]
    n = len(avail)
    
    rng = np.random.default_rng(42)
    samples = {}
    for p in avail:
        w, d, l = wdl(track_bs[p])
        s = rng.dirichlet([1+w, 1+d, 1+l], 50000)
        samples[p] = s[:,0] + 0.5*s[:,1]
    
    matrix = np.full((n, n), np.nan)
    for i, p1 in enumerate(avail):
        for j, p2 in enumerate(avail):
            if i != j:
                matrix[i, j] = float((samples[p1] > samples[p2]).mean())
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    cmap = "viridis"
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    
    tick_labels = [LABELS[p] for p in avail]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels)
    
    ax.tick_params(axis="both", which="both", length=2)

    for i in range(n):
        for j in range(n):
            if i == j:
                pass
            else:
                val = matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, color="white")
    
    fig.text(0.5, -0.05, "Figure 4: Pairwise Bayesian Comparison (Lc0)", ha="center", va="top", fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_pairwise_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_pairwise_heatmap.png", bbox_inches="tight")
    print("  ✓ Figure 4 saved")
    plt.close(fig)


# ═════════════════════════════════════════════════════════
#  FIGURE 4: Color Asymmetry
# ═════════════════════════════════════════════════════════
def fig4():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    patterns = [p for p in PATTERNS_ORDER if p in track_bs]
    labels = [LABELS[p] for p in patterns]
    
    w_scores, b_scores = [], []
    for pat in patterns:
        # Queen is White when NoQ is Black
        wg = [g for g in track_bs[pat] if g["noq_side"] == "black"]
        # Queen is Black when NoQ is White
        bg = [g for g in track_bs[pat] if g["noq_side"] == "white"]
        
        # wdl() returns (Queen Wins, Draws, Queen Losses)
        qw, qd, ql = wdl(wg)
        bw, bd, bl = wdl(bg)
        
        w_scores.append(score(qw, qd, ql)) # Queen score when White
        b_scores.append(score(bw, bd, bl)) # Queen score when Black
    
    x = np.arange(len(patterns))
    width = 0.35
    
    ax.bar(x - width/2, w_scores, width, color="#5B9BD5", edgecolor="black", linewidth=0.8,
           label="Queen = White (has first-move)", zorder=3)
    ax.bar(x + width/2, b_scores, width, color="#333333", edgecolor="black", linewidth=0.8,
           label="Queen = Black", zorder=3)
    
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    
    for i in range(len(patterns)):
        delta = w_scores[i] - b_scores[i]
        mid = max(w_scores[i], b_scores[i]) + 0.03
        ax.annotate(f"Δ={delta:+.2f}", xy=(x[i], mid), ha="center", fontsize=10, color="black")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Q-Score (Queen's Win Rate)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    
    fig.text(0.5, -0.05, "Figure 5: Color Asymmetry (Lc0)", ha="center", va="top", fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_color_asymmetry.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_color_asymmetry.png", bbox_inches="tight")
    print("  ✓ Figure 5 saved")
    plt.close(fig)


# ═════════════════════════════════════════════════════════
#  FIGURE 5: Track A vs Track B (Negative Control)
# ═════════════════════════════════════════════════════════
def fig5():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    patterns = [p for p in PATTERNS_ORDER if p in track_a and p in track_bs]
    labels = [LABELS[p] for p in patterns]
    
    a_scores, bs_scores = [], []
    a_ci_lo, a_ci_hi = [], []
    bs_ci_lo, bs_ci_hi = [], []
    
    for pat in patterns:
        w, d, l = wdl(track_a[pat])
        s = score(w, d, l)
        lo, hi = ci95(w, d, l)
        a_scores.append(s); a_ci_lo.append(s-lo); a_ci_hi.append(hi-s)
        
        w, d, l = wdl(track_bs[pat])
        s = score(w, d, l)
        lo, hi = ci95(w, d, l)
        bs_scores.append(s); bs_ci_lo.append(s-lo); bs_ci_hi.append(hi-s)
    
    x = np.arange(len(patterns))
    width = 0.35
    
    ax.bar(x - width/2, a_scores, width,
           yerr=[a_ci_lo, a_ci_hi], capsize=4,
           color="#AAAAAA", edgecolor="black", linewidth=0.8,
           label="ScratchZero (~500 Elo)", zorder=3)
    ax.bar(x + width/2, bs_scores, width,
           yerr=[bs_ci_lo, bs_ci_hi], capsize=4,
           color="#2E86AB", edgecolor="black", linewidth=0.8,
           label="Lc0 (~2200+ Elo)", zorder=3)
    
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Q-Score")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    
    fig.text(0.5, -0.05, "Figure 6: ScratchZero vs Lc0 Comparison", ha="center", va="top", fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_track_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_track_comparison.png", bbox_inches="tight")
    print("  ✓ Figure 6 saved")
    plt.close(fig)


# ═════════════════════════════════════════════════════════
#  RUN ALL
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating paper figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print(f"\nAll figures saved to {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"  {f.name} ({f.stat().st_size/1024:.0f} KB)")
