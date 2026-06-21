import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.generate_figures import *

def fig2_fixed():
    fig, ax = plt.subplots(figsize=(6, 5))
    
    conc_levels = ["High\n(3 pieces)", "Medium\n(5 pieces)", "Low\n(7 pieces)"]
    conc_keys = ["High", "Medium", "Low"]
    
    sto_scores = []
    
    # We will compute the pooled average for the bars
    for ck in conc_keys:
        sg = [g for p in PATTERNS_ORDER if p in track_bs and CONC[p]==ck for g in track_bs[p]]
        w, d, l = wdl(sg)
        s = score(w, d, l)
        sto_scores.append(s)
    
    x = np.arange(len(conc_levels))
    width = 0.5
    
    # Plot the bars (averages)
    bars2 = ax.bar(x, sto_scores, width,
                   color="#ED7D31", edgecolor="black", linewidth=0.8, alpha=0.7, zorder=2)
    
    # Plot individual pattern scores as scatter dots over the bars
    for i, ck in enumerate(conc_keys):
        patterns_in_conc = [p for p in PATTERNS_ORDER if p in track_bs and CONC[p]==ck]
        for p in patterns_in_conc:
            w, d, l = wdl(track_bs[p])
            pscore = score(w, d, l)
            # Add some slight jitter to x if there are many dots (optional)
            jitter = np.random.uniform(-0.1, 0.1)
            ax.scatter(x[i] + jitter, pscore, color="black", s=50, edgecolor="white", zorder=4)
            # Add small text label for the extreme outlier B+B+N
            if p == "bishop_bishop_knight":
                ax.text(x[i] + jitter + 0.05, pscore, "B+B+N", va="center", ha="left", fontsize=10, color="black")
            elif p == "rook_4pawns":
                ax.text(x[i] + jitter + 0.05, pscore, "R+4P", va="center", ha="left", fontsize=10, color="black")
                
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(conc_levels)
    ax.set_ylabel("Q-Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    
    # Value labels for the bars
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f"Avg: {h:.3f}",
               ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    ax.set_title("Figure 2: Q-Score by Concentration (with Pattern Variance)", ha="center", va="top", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_concentration_fixed.png", bbox_inches="tight")
    print("Fixed Figure 2 generated!")

fig2_fixed()
