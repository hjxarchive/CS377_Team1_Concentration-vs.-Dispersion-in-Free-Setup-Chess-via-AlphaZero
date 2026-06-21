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

def fig_heatmap_asymmetry():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    matrix = np.full((2, 4), np.nan)
    text_matrix = np.empty((2, 4), dtype=object)
    
    for r in [0, 1]:
        for m in [0, 1, 2, 3]:
            if (r, m) in DATA_MAP:
                pats = DATA_MAP[(r, m)]
                
                # Calculate combined Q_White and Q_Black for the cell
                w_q_games = [] # Games where Queen is White (NoQ is Black)
                b_q_games = [] # Games where Queen is Black (NoQ is White)
                
                for p in pats:
                    if p in track_bs:
                        w_q_games.extend([g for g in track_bs[p] if g["noq_side"] == "black"])
                        b_q_games.extend([g for g in track_bs[p] if g["noq_side"] == "white"])
                
                qw, qd, ql = wdl(w_q_games)
                bw, bd, bl = wdl(b_q_games)
                
                # wdl() ALWAYS returns (Queen Wins, Draws, Queen Losses) because 'result' in JSON is from Queen's perspective.
                q_white_score = score(qw, qd, ql)
                q_black_score = score(bw, bd, bl)
                
                delta = q_white_score - q_black_score
                
                row = 1 - r
                col = m
                matrix[row, col] = delta
                text_matrix[row, col] = f"Δ {delta:+.3f}"
    
    cmap = plt.get_cmap("Reds")
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=0.2, aspect="auto")
    
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
                # Dark colors (high values) get white text, light colors get black text
                color = "white" if val > 0.1 else "black"
                ax.text(c, r, text_matrix[r, c], ha="center", va="center", color=color, fontsize=18, fontweight="bold")
            else:
                ax.text(c, r, "N/A", ha="center", va="center", color="gray", fontsize=14, fontstyle="italic")
    
    # Add gridlines to separate cells clearly
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    ax.set_title("Color Asymmetry Delta (Queen White - Queen Black)", ha="center", va="bottom", fontsize=14, fontweight="bold", y=1.03)
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("First-Mover Advantage (Delta)", rotation=270, labelpad=15, fontsize=11)
    
    fig.tight_layout()
    out_path = OUT_DIR / "fig_heatmap_asymmetry.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated {out_path}")

if __name__ == "__main__":
    fig_heatmap_asymmetry()
