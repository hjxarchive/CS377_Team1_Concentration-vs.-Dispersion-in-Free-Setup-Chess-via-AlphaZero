#!/usr/bin/env python3
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from scripts.generate_figures import load_jsonl, wdl, score

RESULTS_DIR = Path("runs/results")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# Patterns with (Rooks, Minors)
# (1,0): R+4P
# (0,1): B+6P, N+6P
# (1,1): R+B+P, R+N+P
# (0,2): B+N+3P
# (0,3): B+B+N

DATA_MAP = {
    (1, 0): ["rook_4pawns"],
    (0, 1): ["bishop_6pawns", "knight_6pawns"],
    (1, 1): ["rook_bishop_pawn", "rook_knight_pawn"],
    (0, 2): ["bishop_knight_3pawns"],
    (0, 3): ["bishop_bishop_knight"]
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

track_bs = {}
for pat in LABELS.keys():
    fp = RESULTS_DIR / f"track_b_stochastic_{pat}.jsonl"
    if fp.exists():
        track_bs[pat] = load_jsonl(fp)

def fig_heatmap():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Grid: Rooks (0, 1) x Minors (0, 1, 2, 3)
    # y-axis: Rooks Lost (1 at top, 0 at bottom) -> rows 0 and 1
    # x-axis: Minors Lost (0, 1, 2, 3) -> cols 0, 1, 2, 3
    
    matrix = np.full((2, 4), np.nan)
    text_matrix = np.empty((2, 4), dtype=object)
    
    for r in [0, 1]:
        for m in [0, 1, 2, 3]:
            if (r, m) in DATA_MAP:
                pats = DATA_MAP[(r, m)]
                sg = []
                for p in pats:
                    if p in track_bs:
                        sg.extend(track_bs[p])
                w, d, l = wdl(sg)
                s = score(w, d, l)
                # Map to matrix: row 0 is 1 Rook, row 1 is 0 Rooks
                row = 1 - r
                col = m
                matrix[row, col] = s
                text_matrix[row, col] = f"{s:.3f}"
    
    cmap = plt.get_cmap("OrRd")
    # We want low scores to be light, high scores to be dark orange/red
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    
    # Ticks
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["1 Rook", "0 Rooks"], fontsize=12)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0 Minors", "1 Minor", "2 Minors", "3 Minors"], fontsize=12)
    
    ax.set_ylabel("Major Pieces Lost", fontsize=13)
    ax.set_xlabel("Minor Pieces Lost", fontsize=13)
    
    # Add text
    for r in range(2):
        for c in range(4):
            if not np.isnan(matrix[r, c]):
                val = matrix[r, c]
                color = "white" if val > 0.6 else "black"
                ax.text(c, r, text_matrix[r, c], ha="center", va="center", color=color, fontsize=17, fontweight="bold")
            else:
                ax.text(c, r, "N/A", ha="center", va="center", color="gray", fontsize=14, fontstyle="italic")
    
    # Add gridlines to separate cells clearly
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Add Delta value where both rows have data
    for c in range(4):
        if not np.isnan(matrix[0, c]) and not np.isnan(matrix[1, c]):
            delta = matrix[0, c] - matrix[1, c]
            ax.text(c, 0.5, f"Δ {delta:+.3f}", ha="center", va="center", 
                    fontsize=16, color='blue', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9), zorder=10)

    ax.set_title("Figure 3: Q-Score by Major vs Minor Pieces Lost", ha="center", va="bottom", fontsize=14, fontweight="bold", y=1.03)
    
    fig.tight_layout()
    out_path = OUT_DIR / "fig_heatmap_pieces.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated {out_path}")

if __name__ == "__main__":
    fig_heatmap()
