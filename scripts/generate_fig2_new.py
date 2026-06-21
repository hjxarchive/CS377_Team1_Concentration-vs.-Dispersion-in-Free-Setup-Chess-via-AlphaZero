#!/usr/bin/env python3
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from scripts.generate_figures import load_jsonl, wdl, score, ci95

RESULTS_DIR = Path("runs/results")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# Define the new grouping based on Number of Minor Pieces Lost
MINOR_GROUPS = {
    "0 Minors": ["rook_4pawns"],
    "1 Minor": ["rook_knight_pawn", "rook_bishop_pawn", "bishop_6pawns", "knight_6pawns"],
    "2 Minors": ["bishop_knight_3pawns"],
    "3 Minors": ["bishop_bishop_knight"]
}

LABELS = {
    "rook_bishop_pawn":     "R+B+P",
    "rook_knight_pawn":     "R+N+P",
    "bishop_bishop_knight": "B+B+N",
    "rook_4pawns":          "R+4P",
    "bishop_knight_3pawns": "B+N+3P",
    "bishop_6pawns":        "B+6P",
    "knight_6pawns":        "N+6P",
}

# Load Stochastic data only
track_bs = {}
for pat in LABELS.keys():
    fp = RESULTS_DIR / f"track_b_stochastic_{pat}.jsonl"
    if fp.exists():
        track_bs[pat] = load_jsonl(fp)

def fig_new():
    fig, ax = plt.subplots(figsize=(7, 5))
    
    group_names = list(MINOR_GROUPS.keys())
    scores = []
    lo_errs = []
    hi_errs = []
    
    # Compute pooled average for each group and proper error bars
    for gn in group_names:
        pats = MINOR_GROUPS[gn]
        
        if len(pats) > 1:
            # Use Min and Max range for the error bar
            pattern_scores = []
            for p in pats:
                if p in track_bs:
                    w, d, l = wdl(track_bs[p])
                    pattern_scores.append(score(w, d, l))
            
            s = np.mean(pattern_scores)
            lo_err = s - np.min(pattern_scores)
            hi_err = np.max(pattern_scores) - s
            
        else:
            # For groups with 1 pattern, error bar = 95% Binomial CI of that pattern
            p = pats[0]
            w, d, l = wdl(track_bs[p])
            s = score(w, d, l)
            lo, hi = ci95(w, d, l)
            lo_err = s - lo
            hi_err = hi - s

        scores.append(s)
        lo_errs.append(lo_err)
        hi_errs.append(hi_err)

    x = np.arange(len(group_names))
    width = 0.5
    
    # Bar plot
    bars = ax.bar(x, scores, width, 
                  yerr=[lo_errs, hi_errs], 
                  capsize=5,
                  color="#ED7D31", 
                  edgecolor="black", 
                  linewidth=0.8, 
                  alpha=0.85,
                  zorder=2,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'zorder': 3})
                
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=12)
    ax.set_ylabel("Q-Score", fontsize=13)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    
    # Value labels on top of bars
    for i, bar in enumerate(bars):
        h = bar.get_height()
        # Place the text slightly above the top error bar cap
        ax.text(bar.get_x() + bar.get_width()/2, h + hi_errs[i] + 0.02, f"{h:.3f}",
               ha="center", va="bottom", fontsize=18, fontweight="bold")
    
    ax.set_title("Figure 2: Q-Score by Number of Minor Pieces Lost", 
                 ha="center", va="bottom", fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    out_path = OUT_DIR / "fig2_minor_pieces.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated {out_path} without scatter points.")

if __name__ == "__main__":
    fig_new()
