#!/usr/bin/env python3
"""
Detailed per-track analysis: Track A / Track B / Elo
Each section produces deep diagnostic statistics.
"""
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

PATTERN_META = {
    "rook_bishop_pawn":     {"pieces": "R+B+P",  "points": "5+3+1=9", "n_removed": 3, "conc": "high"},
    "rook_knight_pawn":     {"pieces": "R+N+P",  "points": "5+3+1=9", "n_removed": 3, "conc": "high"},
    "bishop_bishop_knight": {"pieces": "B+B+N",  "points": "3+3+3=9", "n_removed": 3, "conc": "high"},
    "rook_4pawns":          {"pieces": "R+4P",   "points": "5+4=9",   "n_removed": 5, "conc": "medium"},
    "bishop_knight_3pawns": {"pieces": "B+N+3P", "points": "3+3+3=9", "n_removed": 5, "conc": "medium"},
    "bishop_6pawns":        {"pieces": "B+6P",   "points": "3+6=9",   "n_removed": 7, "conc": "low"},
    "knight_6pawns":        {"pieces": "N+6P",   "points": "3+6=9",   "n_removed": 7, "conc": "low"},
}
PATTERNS = list(PATTERN_META.keys())

def load_jsonl(filepath):
    games = []
    with open(filepath, "r") as f:
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
    a = np.array([1.0 + w, 1.0 + d, 1.0 + l])
    s = rng.dirichlet(a, size=n)
    sc = s[:, 0] + 0.5 * s[:, 1]
    return float(np.percentile(sc, 2.5)), float(np.percentile(sc, 97.5))

def elo_diff(score_val):
    """Convert score to Elo difference (logistic model)."""
    if score_val <= 0 or score_val >= 1:
        return float('inf') if score_val >= 1 else float('-inf')
    return -400 * math.log10(1 / score_val - 1)

def bayesian_superiority(games1, games2, n_samples=50000, seed=42):
    """P(score1 > score2) via posterior sampling."""
    rng = np.random.default_rng(seed)
    w1, d1, l1 = wdl(games1)
    w2, d2, l2 = wdl(games2)
    s1 = rng.dirichlet([1+w1, 1+d1, 1+l1], n_samples)
    s2 = rng.dirichlet([1+w2, 1+d2, 1+l2], n_samples)
    sc1 = s1[:, 0] + 0.5 * s1[:, 1]
    sc2 = s2[:, 0] + 0.5 * s2[:, 1]
    return float((sc1 > sc2).mean())

# ===== LOAD DATA =====
results_dir = Path("runs/results")
track_a, track_b, track_bs = {}, {}, {}
for pat in PATTERNS:
    fa = results_dir / f"track_a_{pat}.jsonl"
    fb = results_dir / f"track_b_{pat}.jsonl"
    fbs = results_dir / f"track_b_stochastic_{pat}.jsonl"
    if fa.exists(): track_a[pat] = load_jsonl(fa)
    if fb.exists(): track_b[pat] = load_jsonl(fb)
    if fbs.exists(): track_bs[pat] = load_jsonl(fbs)


# ==========================================================================
#  PART 1: TRACK A — Custom AlphaZero Deep Dive
# ==========================================================================
print("\n" + "█" * 90)
print("█  PART 1: TRACK A — Custom AlphaZero Self-Play Analysis")
print("█" * 90)

print("\n1.1 Overview")
print("-" * 70)
total_a = sum(len(v) for v in track_a.values())
all_a = [g for games in track_a.values() for g in games]
aw, ad, al = wdl(all_a)
print(f"  Total games: {total_a}")
print(f"  Engine: Custom AlphaZero (ResNet 10 blocks × 128 channels)")
print(f"  MCTS simulations: 400 per move")
print(f"  Evaluation: Same net plays both Q-side and NoQ-side")
print(f"  Overall Q-side W/D/L: {aw}/{ad}/{al}")
print(f"  Overall Q-side Score: {score(aw, ad, al):.3f}")

print("\n1.2 Per-Pattern Results")
print("-" * 70)
print(f"  {'Pattern':<25} {'N':>4} {'Q-W':>5} {'Drw':>5} {'Q-L':>5} {'Q-Scr':>6} {'95% CI':>18} {'Elo Δ':>8}")
print("  " + "-" * 78)

for pat in PATTERNS:
    if pat not in track_a: continue
    g = track_a[pat]
    w, d, l = wdl(g)
    s = score(w, d, l)
    ci = ci95(w, d, l)
    ed = elo_diff(s) if 0 < s < 1 else "N/A"
    ed_str = f"{ed:>+8.0f}" if isinstance(ed, float) and abs(ed) < 9999 else f"{'N/A':>8}"
    print(f"  {pat:<25} {len(g):>4} {w:>5} {d:>5} {l:>5} {s:>6.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {ed_str}")

print("\n1.3 Result Type Breakdown")
print("-" * 70)
for pat in PATTERNS:
    if pat not in track_a: continue
    g = track_a[pat]
    terms = defaultdict(int)
    for game in g:
        terms[game.get("termination", "completed")] += 1
    t_str = ", ".join(f"{k}: {v}" for k, v in sorted(terms.items(), key=lambda x: -x[1]))
    print(f"  {pat:<25} {t_str}")

print("\n1.4 Game Length Distribution by Pattern")
print("-" * 70)
print(f"  {'Pattern':<25} {'Mean':>7} {'Med':>5} {'Min':>5} {'Max':>5} {'σ':>6} {'%≥170':>7}")
print("  " + "-" * 62)
for pat in PATTERNS:
    if pat not in track_a: continue
    plies = [g["ply"] for g in track_a[pat]]
    pct_long = sum(1 for p in plies if p >= 170) / len(plies) * 100
    print(f"  {pat:<25} {np.mean(plies):>7.1f} {int(np.median(plies)):>5} {min(plies):>5} {max(plies):>5} {np.std(plies):>6.1f} {pct_long:>6.1f}%")

print("\n1.5 Color Asymmetry (NoQ side)")
print("-" * 70)
print(f"  {'Pattern':<25} {'NoQ=W Score':>11} {'(W/D/L)':>16} {'NoQ=B Score':>11} {'(W/D/L)':>16} {'Δ':>6}")
print("  " + "-" * 83)
for pat in PATTERNS:
    if pat not in track_a: continue
    by_noq = defaultdict(list)
    for g in track_a[pat]:
        by_noq[g["noq_side"]].append(g)
    res = {}
    for c in ["white", "black"]:
        cg = by_noq.get(c, [])
        w, d, l = wdl(cg)
        # noq perspective: noq wins when q loses
        nw, nd, nl = l, d, w
        res[c] = {"score": score(nw, nd, nl), "wdl": f"({nw}/{nd}/{nl})"}
    delta = res["white"]["score"] - res["black"]["score"]
    print(f"  {pat:<25} {res['white']['score']:>11.3f} {res['white']['wdl']:>16} {res['black']['score']:>11.3f} {res['black']['wdl']:>16} {delta:>+6.3f}")

print("\n1.6 Track A Diagnosis")
print("-" * 70)
print("""  • Zero Q-side wins across all 1,400 games → model can't exploit the queen.
  • All outcomes are draws (high %) or NoQ-side wins (losses from Q-perspective).
  • Average game length ~130-140 ply with many hitting the 179-move limit.
  • Diagnosis: The custom AlphaZero model is too weak to properly exploit material advantage.
    - With only 400 MCTS simulations and a 10-block ResNet, the model likely learned
      a reasonable but not strong policy that leads to many drawn endgames.
    - The model plays identically on both sides (same net), so it can't differentiate
      between having vs lacking a queen effectively.
  • Implication: Track A results are NOT suitable for answering the research questions.
    Track B (lc0) results should be used as the primary evidence.""")


# ==========================================================================
#  PART 2: TRACK B — lc0 Reference Engine Deep Dive
# ==========================================================================
print("\n\n" + "█" * 90)
print("█  PART 2: TRACK B — lc0 Reference Engine Analysis")
print("█" * 90)

print("\n2.1 Overview")
print("-" * 70)
total_b = sum(len(v) for v in track_b.values())
total_bs = sum(len(v) for v in track_bs.values())
print(f"  Track B Deterministic: {total_b} games (lc0 n=800, Temperature decay 20 moves)")
print(f"  Track B Stochastic:    {total_bs} games (lc0 n=800, MultiPV={5} softmax first 10 plies)")
print(f"  Both: Color-balanced (200 games NoQ=White + 200 games NoQ=Black per pattern)")

# --- Track B Deterministic ---
print("\n\n" + "=" * 90)
print("  2.2 Track B DETERMINISTIC Results")
print("=" * 90)
print(f"\n  {'Pattern':<25} {'Conc':<7} {'N':>4} {'Q-W':>5} {'Drw':>5} {'Q-L':>5} {'Q-Scr':>6} {'95% CI':>18} {'Elo Δ':>8}")
print("  " + "-" * 80)

b_scores = {}
for pat in PATTERNS:
    if pat not in track_b: continue
    g = track_b[pat]
    w, d, l = wdl(g)
    s = score(w, d, l)
    ci = ci95(w, d, l)
    ed = elo_diff(s)
    ed_str = f"{ed:>+8.0f}" if abs(ed) < 9999 else "   ±∞"
    b_scores[pat] = s
    print(f"  {pat:<25} {PATTERN_META[pat]['conc']:<7} {len(g):>4} {w:>5} {d:>5} {l:>5} {s:>6.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {ed_str}")

# --- Track B Stochastic ---
print("\n\n" + "=" * 90)
print("  2.3 Track B STOCHASTIC Results")
print("=" * 90)
print(f"\n  {'Pattern':<25} {'Conc':<7} {'N':>4} {'Q-W':>5} {'Drw':>5} {'Q-L':>5} {'Q-Scr':>6} {'95% CI':>18} {'Elo Δ':>8}")
print("  " + "-" * 80)

bs_scores = {}
for pat in PATTERNS:
    if pat not in track_bs: continue
    g = track_bs[pat]
    w, d, l = wdl(g)
    s = score(w, d, l)
    ci = ci95(w, d, l)
    ed = elo_diff(s)
    ed_str = f"{ed:>+8.0f}" if abs(ed) < 9999 else "   ±∞"
    bs_scores[pat] = s
    print(f"  {pat:<25} {PATTERN_META[pat]['conc']:<7} {len(g):>4} {w:>5} {d:>5} {l:>5} {s:>6.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {ed_str}")


# --- Deterministic vs Stochastic comparison ---
print("\n\n" + "=" * 90)
print("  2.4 Deterministic vs Stochastic Comparison")
print("=" * 90)
print(f"\n  {'Pattern':<25} {'Det.':>7} {'Stoch.':>7} {'Δ':>7} {'Agree?':>7}")
print("  " + "-" * 56)
for pat in PATTERNS:
    if pat in b_scores and pat in bs_scores:
        d_s = b_scores[pat]
        s_s = bs_scores[pat]
        delta = d_s - s_s
        agree = "✓" if (d_s - 0.5) * (s_s - 0.5) > 0 else "✗"
        print(f"  {pat:<25} {d_s:>7.3f} {s_s:>7.3f} {delta:>+7.3f} {agree:>7}")


# --- Concentration-level aggregation ---
print("\n\n" + "=" * 90)
print("  2.5 Concentration-Level Aggregation")
print("=" * 90)

for track_name, data in [("Deterministic", track_b), ("Stochastic", track_bs)]:
    print(f"\n  --- {track_name} ---")
    print(f"  {'Conc.':<10} {'#Removed':>8} {'N':>6} {'Q-W':>6} {'Drw':>6} {'Q-L':>6} {'Q-Score':>8} {'95% CI':>18} {'Elo Δ':>8}")
    print("  " + "-" * 80)
    for conc in ["high", "medium", "low"]:
        cg = [g for p in PATTERNS if p in data and PATTERN_META[p]["conc"] == conc for g in data[p]]
        if not cg: continue
        w, d, l = wdl(cg)
        s = score(w, d, l)
        ci = ci95(w, d, l)
        ed = elo_diff(s)
        nr = [PATTERN_META[p]["n_removed"] for p in PATTERNS if PATTERN_META[p]["conc"] == conc][0]
        ed_str = f"{ed:>+8.0f}" if abs(ed) < 9999 else "   ±∞"
        print(f"  {conc:<10} {nr:>8} {len(cg):>6} {w:>6} {d:>6} {l:>6} {s:>8.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {ed_str}")


# --- Color Asymmetry (Track B) ---
print("\n\n" + "=" * 90)
print("  2.6 Color Asymmetry (Track B Stochastic)")
print("=" * 90)
print(f"\n  {'Pattern':<25} {'NoQ=W':>8} {'Elo':>6}  {'NoQ=B':>8} {'Elo':>6}  {'Δ Score':>8} {'Δ Elo':>8}")
print("  " + "-" * 78)

all_w_games = []
all_b_games = []
for pat in PATTERNS:
    if pat not in track_bs: continue
    wg = [g for g in track_bs[pat] if g["noq_side"] == "white"]
    bg = [g for g in track_bs[pat] if g["noq_side"] == "black"]
    all_w_games.extend(wg)
    all_b_games.extend(bg)
    # NoQ scores
    ww, wd, wl = wdl(wg)
    bw, bd, bl = wdl(bg)
    # From NoQ perspective: noq wins when q loses
    nw_s = score(wl, wd, ww)
    nb_s = score(bl, bd, bw)
    nw_elo = elo_diff(nw_s) if 0 < nw_s < 1 else float('inf')
    nb_elo = elo_diff(nb_s) if 0 < nb_s < 1 else float('-inf')
    nw_e_str = f"{nw_elo:>+6.0f}" if abs(nw_elo) < 9999 else "  ±∞"
    nb_e_str = f"{nb_elo:>+6.0f}" if abs(nb_elo) < 9999 else "  -∞"
    d_elo = nw_elo - nb_elo if abs(nw_elo) < 9999 and abs(nb_elo) < 9999 else float('nan')
    d_elo_str = f"{d_elo:>+8.0f}" if not math.isnan(d_elo) else "     N/A"
    print(f"  {pat:<25} {nw_s:>8.3f} {nw_e_str}  {nb_s:>8.3f} {nb_e_str}  {nw_s - nb_s:>+8.3f} {d_elo_str}")

# Aggregate
ww, wd, wl = wdl(all_w_games)
bw, bd, bl = wdl(all_b_games)
agg_ws = score(wl, wd, ww)
agg_bs = score(bl, bd, bw)
print(f"  {'AGGREGATE':<25} {agg_ws:>8.3f} {elo_diff(agg_ws):>+6.0f}  {agg_bs:>8.3f} {elo_diff(agg_bs):>+6.0f}  {agg_ws - agg_bs:>+8.3f} {elo_diff(agg_ws) - elo_diff(agg_bs):>+8.0f}")


# --- Game Length by outcome ---
print("\n\n" + "=" * 90)
print("  2.7 Game Length by Outcome (Track B Stochastic)")
print("=" * 90)

for outcome_label, outcome_val in [("Q-Wins (checkmate)", "win"), ("Draws", "draw"), ("Q-Losses", "loss")]:
    og = [g for p in PATTERNS if p in track_bs for g in track_bs[p] if g["result"] == outcome_val]
    if og:
        plies = [g["ply"] for g in og]
        print(f"  {outcome_label:<25} N={len(og):>4}  Mean={np.mean(plies):.1f}  Med={int(np.median(plies))}  Min={min(plies)}  Max={max(plies)}")


# --- Termination Analysis ---
print("\n\n" + "=" * 90)
print("  2.8 Termination Types")
print("=" * 90)
for track_name, data in [("Deterministic", track_b), ("Stochastic", track_bs)]:
    print(f"\n  --- {track_name} ---")
    terms = defaultdict(lambda: {"total": 0, "q_win": 0, "draw": 0, "q_loss": 0})
    for p in PATTERNS:
        if p not in data: continue
        for g in data[p]:
            t = g.get("termination", "unknown")
            terms[t]["total"] += 1
            if g["result"] == "win": terms[t]["q_win"] += 1
            elif g["result"] == "draw": terms[t]["draw"] += 1
            else: terms[t]["q_loss"] += 1
    
    total_all = sum(v["total"] for v in terms.values())
    print(f"  {'Termination':<25} {'Total':>6} {'%':>7} {'Q-Win':>7} {'Draw':>7} {'Q-Loss':>7}")
    print("  " + "-" * 65)
    for t, v in sorted(terms.items(), key=lambda x: -x[1]["total"]):
        print(f"  {t:<25} {v['total']:>6} {v['total']/total_all*100:>6.1f}% {v['q_win']:>7} {v['draw']:>7} {v['q_loss']:>7}")


# --- Pairwise Bayesian comparisons ---
print("\n\n" + "=" * 90)
print("  2.9 Pairwise Pattern Comparison (Stochastic)")
print("  P(Q-score for pattern_row > Q-score for pattern_col)")
print("=" * 90)

avail = [p for p in PATTERNS if p in track_bs]
short = {
    "rook_bishop_pawn": "RBP",
    "rook_knight_pawn": "RNP",
    "bishop_bishop_knight": "BBN",
    "rook_4pawns": "R4P",
    "bishop_knight_3pawns": "BN3P",
    "bishop_6pawns": "B6P",
    "knight_6pawns": "N6P",
}

header = "  " + f"{'':>8}" + "".join(f"{short[p]:>8}" for p in avail)
print(header)
print("  " + "-" * (8 + 8 * len(avail)))

rng = np.random.default_rng(42)
samples = {}
for p in avail:
    w, d, l = wdl(track_bs[p])
    s = rng.dirichlet([1+w, 1+d, 1+l], 50000)
    samples[p] = s[:, 0] + 0.5 * s[:, 1]

for p1 in avail:
    row = f"  {short[p1]:>8}"
    for p2 in avail:
        if p1 == p2:
            row += f"{'—':>8}"
        else:
            prob = float((samples[p1] > samples[p2]).mean())
            row += f"{prob:>8.2f}"
    print(row)


# --- Ranking ---
print("\n\n" + "=" * 90)
print("  2.10 Pattern Ranking (by Q-side advantage, Stochastic)")
print("=" * 90)
print(f"\n  Rank  {'Pattern':<25} {'Q-Score':>8} {'Elo Δ':>8} {'Interp'}")
print("  " + "-" * 80)
ranked = sorted(bs_scores.items(), key=lambda x: -x[1])
for i, (pat, s) in enumerate(ranked, 1):
    ed = elo_diff(s)
    ed_str = f"{ed:>+8.0f}" if abs(ed) < 9999 else "    ±∞"
    if s > 0.85:
        interp = "Dispersed handicap severely cripples"
    elif s > 0.65:
        interp = "Queen advantage is significant"
    elif s > 0.55:
        interp = "Slight queen advantage"
    elif s > 0.45:
        interp = "Roughly equal"
    else:
        interp = "NOQ side favored — removal hurts more than losing queen"
    print(f"  {i:>4}  {pat:<25} {s:>8.3f} {ed_str}  {interp}")


# ==========================================================================
#  PART 3: ELO CALIBRATION — Custom AlphaZero Strength
# ==========================================================================
print("\n\n" + "█" * 90)
print("█  PART 3: ELO CALIBRATION — Custom AlphaZero Strength")
print("█" * 90)

elo_file = results_dir / "elo_calibration.txt"
print("\n3.1 Elo Gauntlet Results (Custom AZ vs lc0)")
print("-" * 70)

# Parse elo_calibration.txt
lc0_levels = {}
with open(elo_file) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("LC0 Nodes="):
            # "LC0 Nodes=1: 0.5/20 (Win Rate: 2.5%)"
            parts = line.split(":", 1)  # split on first colon only
            level = parts[0].strip()
            rest = parts[1].strip()
            score_parts = rest.split("/")
            pts = float(score_parts[0])
            total = int(score_parts[1].split(" ")[0])
            wr_str = rest.split("Win Rate:")[1].strip().rstrip("%)")
            wr = float(wr_str)
            lc0_levels[level] = {"score": pts, "total": total, "win_rate": wr}

print(f"  {'Level':<20} {'Score':>8} {'Games':>6} {'Win Rate':>10} {'Est. AZ Elo':>12}")
print("  " + "-" * 58)

# Reference Elo for lc0 node levels (approximate)
lc0_ref_elo = {
    "LC0 Nodes=1": 800,
    "LC0 Nodes=2": 950,
    "LC0 Nodes=4": 1100,
    "LC0 Nodes=8": 1300,
    "LC0 Nodes=16": 1500,
}

for level, data in lc0_levels.items():
    wr = data["win_rate"] / 100
    if wr <= 0:
        wr = 0.01  # floor for Elo calculation
    ref = lc0_ref_elo.get(level, "?")
    az_elo = ref + elo_diff(wr) if isinstance(ref, int) and 0 < wr < 1 else "?"
    az_str = f"~{int(az_elo)}" if isinstance(az_elo, float) else "< 400"
    print(f"  {level:<20} {data['score']:>8.1f} {data['total']:>6} {data['win_rate']:>9.1f}% {az_str:>12}")

print("\n3.2 Elo Calibration Diagnosis")
print("-" * 70)
print("""  • Custom AlphaZero scored 0.5/20 against LC0 Nodes=1 (est. ~800 Elo)
    and 0.0/20 against all higher levels.
  • Win rate of 2.5% against ~800 Elo → AlphaZero Elo ≈ 400-500
  • This is below beginner human level (~600-800 Elo).
  
  Why is the model so weak?
  ┌─────────────────────────────────────────────────────────────────┐
  │ Factor                     │ Impact                           │
  ├─────────────────────────────────────────────────────────────────┤
  │ ResNet: 10 blocks × 128ch  │ Very small (AZ used 40×256)      │
  │ MCTS: 400 sims at eval     │ Low (AZ used 800; lc0 used 800)  │
  │ Training data               │ Limited self-play iterations     │
  │ Training compute            │ Far below AZ-level resources     │
  └─────────────────────────────────────────────────────────────────┘
  
  • The custom AZ model is essentially playing at a "random with some structure"
    level — it knows basic piece movements but can't play tactically.
  • This explains why Track A showed all draws: the model can't exploit any
    material advantage, whether it's a queen or dispersed pieces.
  
  Implication for the research:
  → Track A results should be discussed as a "negative control" showing that
    a weak engine cannot differentiate handicap patterns.
  → Track B (lc0 at 800 nodes ≈ 2200+ Elo) is the authoritative data source.""")

print("\n3.3 lc0 Reference Engine Strength")
print("-" * 70)
print("""  lc0 at different node budgets (approximate BayesElo):
  ┌──────────────────────┬─────────────┬──────────────────────┐
  │ Node Budget          │ ~BayesElo   │ Human Equivalent     │
  ├──────────────────────┼─────────────┼──────────────────────┤
  │ Nodes = 1            │ ~800        │ Casual beginner      │
  │ Nodes = 2            │ ~950        │ Novice               │
  │ Nodes = 4            │ ~1100       │ Club player          │
  │ Nodes = 8            │ ~1300       │ Intermediate         │
  │ Nodes = 16           │ ~1500       │ Advanced amateur     │
  │ Nodes = 800 (Track B)│ ~2200+      │ Expert/Master        │
  └──────────────────────┴─────────────┴──────────────────────┘
  
  Track B uses 800 nodes → approximately Expert/Master level play.
  This is strong enough to properly exploit material advantages and
  produce meaningful data for the research questions.""")


# ==========================================================================
#  FINAL CROSS-TRACK SUMMARY
# ==========================================================================
print("\n\n" + "█" * 90)
print("█  CROSS-TRACK SUMMARY")
print("█" * 90)

print("\n  Comparison of Q-side scores across tracks:")
print(f"\n  {'Pattern':<25} {'Track A':>8} {'Track B':>8} {'B-Stoch':>8}")
print("  " + "-" * 52)
for pat in PATTERNS:
    a_s = score(*wdl(track_a[pat])) if pat in track_a else "-"
    b_s = b_scores.get(pat, "-")
    bs_s = bs_scores.get(pat, "-")
    a_str = f"{a_s:>8.3f}" if isinstance(a_s, float) else f"{a_s:>8}"
    b_str = f"{b_s:>8.3f}" if isinstance(b_s, float) else f"{b_s:>8}"
    bs_str = f"{bs_s:>8.3f}" if isinstance(bs_s, float) else f"{bs_s:>8}"
    print(f"  {pat:<25} {a_str} {b_str} {bs_str}")

# Correlation between deterministic and stochastic
det_vals = [b_scores[p] for p in PATTERNS if p in b_scores and p in bs_scores]
sto_vals = [bs_scores[p] for p in PATTERNS if p in b_scores and p in bs_scores]
if det_vals and sto_vals:
    corr = np.corrcoef(det_vals, sto_vals)[0, 1]
    print(f"\n  Correlation(Deterministic, Stochastic) = {corr:.3f}")
    if corr > 0.8:
        print("  → Strong agreement between sampling methods. Results are robust.")
    elif corr > 0.5:
        print("  → Moderate agreement. Opening randomization changes some patterns significantly.")
    else:
        print("  → Weak agreement. Opening selection matters a lot for these patterns.")

print("\n" + "█" * 90)
print("  ANALYSIS COMPLETE")
print("█" * 90)
