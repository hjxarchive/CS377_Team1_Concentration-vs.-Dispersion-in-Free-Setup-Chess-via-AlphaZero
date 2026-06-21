#!/usr/bin/env python3
"""
Comprehensive analysis of all Track A, Track B, and Track B Stochastic results.
Produces summary statistics, win rates, concentration analysis, and color asymmetry.
"""
import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Pattern metadata
PATTERN_META = {
    "rook_bishop_pawn":    {"pieces": "R+B+P",   "points": "5+3+1=9", "num_removed": 3, "concentration": "high"},
    "rook_knight_pawn":    {"pieces": "R+N+P",   "points": "5+3+1=9", "num_removed": 3, "concentration": "high"},
    "bishop_bishop_knight":{"pieces": "B+B+N",   "points": "3+3+3=9", "num_removed": 3, "concentration": "high"},
    "rook_4pawns":         {"pieces": "R+4P",    "points": "5+4=9",   "num_removed": 5, "concentration": "medium"},
    "bishop_knight_3pawns":{"pieces": "B+N+3P",  "points": "3+3+3=9", "num_removed": 5, "concentration": "medium"},
    "bishop_6pawns":       {"pieces": "B+6P",    "points": "3+6=9",   "num_removed": 7, "concentration": "low"},
    "knight_6pawns":       {"pieces": "N+6P",    "points": "3+6=9",   "num_removed": 7, "concentration": "low"},
}

CONC_ORDER = ["high", "medium", "low"]

def load_jsonl(filepath):
    """Load a JSONL file into a list of dicts."""
    games = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def compute_stats(games):
    """Compute W/D/L counts and score from a list of game dicts."""
    wins = sum(1 for g in games if g["result"] == "win")
    draws = sum(1 for g in games if g["result"] == "draw")
    losses = sum(1 for g in games if g["result"] == "loss")
    total = len(games)
    score = (wins + 0.5 * draws) / total if total > 0 else 0
    return {
        "wins": wins, "draws": draws, "losses": losses,
        "total": total, "score": score
    }


def compute_ci(wins, draws, losses, n_samples=50000, seed=42):
    """Bayesian credible interval via Dirichlet-multinomial."""
    rng = np.random.default_rng(seed)
    alpha = np.array([1.0 + wins, 1.0 + draws, 1.0 + losses])
    samples = rng.dirichlet(alpha, size=n_samples)
    scores = samples[:, 0] + 0.5 * samples[:, 1]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def analyze_by_color(games):
    """Split games by which side is handicapped (noq_side) and compute stats."""
    # In these experiments, "noq_side" loses the queen, "q_side" loses the dispersed bundle.
    # "result" = "win" means the Q-side (with queen) wins => handicap side (noq) lost.
    # Actually, let's check the data structure more carefully.
    # result_score: 1.0 = q_side wins (side that keeps queen), 0.5 = draw, 0.0 = noq_side wins
    # Wait - let me re-read. The "result" field from perspective of whom?
    
    # From Track B data: q_side="black", noq_side="white", result="win", result_score=1.0
    # The "result" is from the perspective of the side that has the queen (q_side).
    # "win" = q_side wins = side with queen wins = handicapped side (noq) loses.
    
    # For color asymmetry: we want to know if being noq as white vs black matters.
    by_noq_color = defaultdict(list)
    for g in games:
        by_noq_color[g["noq_side"]].append(g)
    
    result = {}
    for color in ["white", "black"]:
        color_games = by_noq_color.get(color, [])
        if color_games:
            stats = compute_stats(color_games)
            # result_score is from q_side perspective.
            # noq_side score = 1 - result_score
            noq_wins = stats["losses"]  # q_side loss = noq_side win
            noq_draws = stats["draws"]
            noq_losses = stats["wins"]  # q_side win = noq_side loss
            noq_total = stats["total"]
            noq_score = (noq_wins + 0.5 * noq_draws) / noq_total if noq_total > 0 else 0
            result[color] = {
                "noq_wins": noq_wins, "noq_draws": noq_draws, "noq_losses": noq_losses,
                "total": noq_total, "noq_score": noq_score
            }
    return result


def main():
    results_dir = Path("runs/results")
    
    patterns = list(PATTERN_META.keys())
    
    # =====================================================
    # Load all data
    # =====================================================
    track_a = {}
    track_b = {}
    track_b_stochastic = {}
    
    for pat in patterns:
        fa = results_dir / f"track_a_{pat}.jsonl"
        fb = results_dir / f"track_b_{pat}.jsonl"
        fbs = results_dir / f"track_b_stochastic_{pat}.jsonl"
        
        if fa.exists():
            track_a[pat] = load_jsonl(fa)
        if fb.exists():
            track_b[pat] = load_jsonl(fb)
        if fbs.exists():
            track_b_stochastic[pat] = load_jsonl(fbs)
    
    # =====================================================
    # SECTION 1: Overall Summary
    # =====================================================
    print("=" * 90)
    print("  COMPREHENSIVE EXPERIMENT RESULTS ANALYSIS")
    print("  Concentration vs. Dispersion in Free-Setup Chess")
    print("=" * 90)
    
    total_a = sum(len(v) for v in track_a.values())
    total_b = sum(len(v) for v in track_b.values())
    total_bs = sum(len(v) for v in track_b_stochastic.values())
    print(f"\n📊 Total games played: {total_a + total_b + total_bs}")
    print(f"   Track A (Custom AlphaZero): {total_a} games across {len(track_a)} patterns")
    print(f"   Track B (lc0 deterministic): {total_b} games across {len(track_b)} patterns")
    print(f"   Track B Stochastic (lc0 + softmax): {total_bs} games across {len(track_b_stochastic)} patterns")
    
    # =====================================================
    # SECTION 2: Track A Results (Custom AlphaZero)
    # =====================================================
    print("\n\n" + "=" * 90)
    print("  TRACK A: Custom AlphaZero Self-Play Results")
    print("  (NoQ side = loses Queen, Q side = loses dispersed bundle)")
    print("=" * 90)
    print(f"{'Pattern':<25} {'Pieces':<10} {'Conc.':<8} {'N':>5} {'Q-Win':>7} {'Draw':>7} {'Q-Loss':>7} {'Q-Score':>8} {'95% CI':>20}")
    print("-" * 90)
    
    for pat in patterns:
        if pat not in track_a:
            continue
        games = track_a[pat]
        stats = compute_stats(games)
        meta = PATTERN_META[pat]
        ci = compute_ci(stats["wins"], stats["draws"], stats["losses"])
        print(f"{pat:<25} {meta['pieces']:<10} {meta['concentration']:<8} "
              f"{stats['total']:>5} {stats['wins']:>7} {stats['draws']:>7} {stats['losses']:>7} "
              f"{stats['score']:>8.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    # =====================================================
    # SECTION 3: Track B Results (lc0 Deterministic)
    # =====================================================
    print("\n\n" + "=" * 90)
    print("  TRACK B: lc0 (n=800, Deterministic) Results")
    print("  Q-Score = Score from queen-side perspective (> 0.5 = queen is better)")
    print("=" * 90)
    print(f"{'Pattern':<25} {'Pieces':<10} {'Conc.':<8} {'N':>5} {'Q-Win':>7} {'Draw':>7} {'Q-Loss':>7} {'Q-Score':>8} {'95% CI':>20}")
    print("-" * 90)
    
    track_b_scores = {}
    for pat in patterns:
        if pat not in track_b:
            continue
        games = track_b[pat]
        stats = compute_stats(games)
        meta = PATTERN_META[pat]
        ci = compute_ci(stats["wins"], stats["draws"], stats["losses"])
        track_b_scores[pat] = stats["score"]
        print(f"{pat:<25} {meta['pieces']:<10} {meta['concentration']:<8} "
              f"{stats['total']:>5} {stats['wins']:>7} {stats['draws']:>7} {stats['losses']:>7} "
              f"{stats['score']:>8.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    # =====================================================
    # SECTION 4: Track B Stochastic Results
    # =====================================================
    print("\n\n" + "=" * 90)
    print("  TRACK B STOCHASTIC: lc0 (n=800, MultiPV Softmax) Results")
    print("  Stochastic opening play for game diversity")
    print("=" * 90)
    print(f"{'Pattern':<25} {'Pieces':<10} {'Conc.':<8} {'N':>5} {'Q-Win':>7} {'Draw':>7} {'Q-Loss':>7} {'Q-Score':>8} {'95% CI':>20}")
    print("-" * 90)
    
    track_bs_scores = {}
    for pat in patterns:
        if pat not in track_b_stochastic:
            continue
        games = track_b_stochastic[pat]
        stats = compute_stats(games)
        meta = PATTERN_META[pat]
        ci = compute_ci(stats["wins"], stats["draws"], stats["losses"])
        track_bs_scores[pat] = stats["score"]
        print(f"{pat:<25} {meta['pieces']:<10} {meta['concentration']:<8} "
              f"{stats['total']:>5} {stats['wins']:>7} {stats['draws']:>7} {stats['losses']:>7} "
              f"{stats['score']:>8.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    # =====================================================
    # SECTION 5: Concentration-Level Analysis
    # =====================================================
    print("\n\n" + "=" * 90)
    print("  CONCENTRATION-LEVEL ANALYSIS")
    print("  Does dispersing the handicap across more pieces help or hurt?")
    print("=" * 90)
    
    for track_name, track_data in [("Track B (Deterministic)", track_b), 
                                     ("Track B (Stochastic)", track_b_stochastic),
                                     ("Track A (AlphaZero)", track_a)]:
        if not track_data:
            continue
        print(f"\n--- {track_name} ---")
        print(f"{'Concentration':<12} {'#Pieces':>8} {'N':>6} {'Q-Win':>7} {'Draw':>7} {'Q-Loss':>7} {'Q-Score':>8} {'95% CI':>20}")
        print("-" * 80)
        
        for conc in CONC_ORDER:
            conc_games = []
            for pat in patterns:
                if pat in track_data and PATTERN_META[pat]["concentration"] == conc:
                    conc_games.extend(track_data[pat])
            if conc_games:
                stats = compute_stats(conc_games)
                ci = compute_ci(stats["wins"], stats["draws"], stats["losses"])
                n_pieces = [PATTERN_META[p]["num_removed"] for p in patterns if PATTERN_META[p]["concentration"] == conc][0]
                print(f"{conc:<12} {n_pieces:>8} {stats['total']:>6} {stats['wins']:>7} {stats['draws']:>7} {stats['losses']:>7} "
                      f"{stats['score']:>8.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    # =====================================================
    # SECTION 6: Color Asymmetry (First-Move Advantage)
    # =====================================================
    print("\n\n" + "=" * 90)
    print("  COLOR ASYMMETRY ANALYSIS")
    print("  Does being handicapped (NoQ) as White vs Black matter?")
    print("  NoQ-Score = Score from handicapped side perspective (< 0.5 = handicap is real)")
    print("=" * 90)
    
    for track_name, track_data in [("Track B (Deterministic)", track_b), 
                                     ("Track B (Stochastic)", track_b_stochastic)]:
        if not track_data:
            continue
        print(f"\n--- {track_name} ---")
        print(f"{'Pattern':<25} {'NoQ=W Score':>12} {'NoQ=W N':>8} {'NoQ=B Score':>12} {'NoQ=B N':>8} {'Δ(W-B)':>8}")
        print("-" * 80)
        
        all_w_score_sum = 0
        all_b_score_sum = 0
        all_w_n = 0
        all_b_n = 0
        
        for pat in patterns:
            if pat not in track_data:
                continue
            color_data = analyze_by_color(track_data[pat])
            w = color_data.get("white", {})
            b = color_data.get("black", {})
            w_score = w.get("noq_score", 0)
            b_score = b.get("noq_score", 0)
            w_n = w.get("total", 0)
            b_n = b.get("total", 0)
            delta = w_score - b_score
            
            all_w_score_sum += w_score * w_n
            all_b_score_sum += b_score * b_n
            all_w_n += w_n
            all_b_n += b_n
            
            print(f"{pat:<25} {w_score:>12.3f} {w_n:>8} {b_score:>12.3f} {b_n:>8} {delta:>+8.3f}")
        
        if all_w_n > 0 and all_b_n > 0:
            avg_w = all_w_score_sum / all_w_n
            avg_b = all_b_score_sum / all_b_n
            print(f"{'AGGREGATE':<25} {avg_w:>12.3f} {all_w_n:>8} {avg_b:>12.3f} {all_b_n:>8} {avg_w - avg_b:>+8.3f}")
    
    # =====================================================
    # SECTION 7: Game Length Analysis
    # =====================================================
    print("\n\n" + "=" * 90)
    print("  GAME LENGTH ANALYSIS")
    print("=" * 90)
    
    for track_name, track_data in [("Track B (Deterministic)", track_b), 
                                     ("Track B (Stochastic)", track_b_stochastic),
                                     ("Track A (AlphaZero)", track_a)]:
        if not track_data:
            continue
        print(f"\n--- {track_name} ---")
        print(f"{'Pattern':<25} {'Mean Ply':>10} {'Median':>8} {'Min':>6} {'Max':>6} {'Std':>8}")
        print("-" * 70)
        
        for pat in patterns:
            if pat not in track_data:
                continue
            plies = [g["ply"] for g in track_data[pat]]
            if plies:
                print(f"{pat:<25} {np.mean(plies):>10.1f} {np.median(plies):>8.0f} "
                      f"{min(plies):>6} {max(plies):>6} {np.std(plies):>8.1f}")
    
    # =====================================================
    # SECTION 8: Termination Analysis
    # =====================================================
    print("\n\n" + "=" * 90)
    print("  TERMINATION TYPE ANALYSIS")
    print("=" * 90)
    
    for track_name, track_data in [("Track B (Deterministic)", track_b), 
                                     ("Track B (Stochastic)", track_b_stochastic)]:
        if not track_data:
            continue
        print(f"\n--- {track_name} ---")
        
        terminations = defaultdict(int)
        for pat in patterns:
            if pat not in track_data:
                continue
            for g in track_data[pat]:
                terminations[g.get("termination", "unknown")] += 1
        
        total = sum(terminations.values())
        print(f"{'Termination':<25} {'Count':>8} {'Percentage':>12}")
        print("-" * 50)
        for term, count in sorted(terminations.items(), key=lambda x: -x[1]):
            print(f"{term:<25} {count:>8} {count/total*100:>11.1f}%")
    
    # =====================================================
    # SECTION 9: Additivity Test
    # =====================================================
    print("\n\n" + "=" * 90)
    print("  ADDITIVITY TEST")
    print("  Is 9 points the same regardless of how you remove them?")
    print("=" * 90)
    
    for track_name, scores_dict in [("Track B (Deterministic)", track_b_scores),
                                     ("Track B (Stochastic)", track_bs_scores)]:
        if not scores_dict:
            continue
        print(f"\n--- {track_name} ---")
        
        vals = list(scores_dict.values())
        score_range = max(vals) - min(vals)
        best_pat = max(scores_dict, key=scores_dict.get)
        worst_pat = min(scores_dict, key=scores_dict.get)
        
        print(f"  Score range: {score_range:.3f}")
        print(f"  Best pattern for Q-side:  {best_pat} (score={scores_dict[best_pat]:.3f})")
        print(f"  Worst pattern for Q-side: {worst_pat} (score={scores_dict[worst_pat]:.3f})")
        
        # Bayesian pairwise test
        print(f"\n  Pairwise P(pattern_i Q-score > pattern_j Q-score):")
        rng = np.random.default_rng(42)
        pattern_samples = {}
        for track_data_full in [track_b, track_b_stochastic]:
            if track_name == "Track B (Deterministic)" and track_data_full is track_b:
                data = track_data_full
                break
            elif track_name == "Track B (Stochastic)" and track_data_full is track_b_stochastic:
                data = track_data_full
                break
        
        for pat in patterns:
            if pat not in data:
                continue
            stats = compute_stats(data[pat])
            alpha = np.array([1.0 + stats["wins"], 1.0 + stats["draws"], 1.0 + stats["losses"]])
            samples = rng.dirichlet(alpha, size=50000)
            pattern_samples[pat] = samples[:, 0] + 0.5 * samples[:, 1]
        
        # Print comparison: concentrated vs dispersed
        concentrated = [p for p in patterns if p in pattern_samples and PATTERN_META[p]["concentration"] == "high"]
        dispersed = [p for p in patterns if p in pattern_samples and PATTERN_META[p]["concentration"] == "low"]
        
        if concentrated and dispersed:
            print(f"\n  Concentrated (high) vs Dispersed (low):")
            for c in concentrated:
                for d in dispersed:
                    prob = float((pattern_samples[c] > pattern_samples[d]).mean())
                    print(f"    P({c} > {d}) = {prob:.3f}")
    
    # =====================================================
    # SECTION 10: Key Findings Summary
    # =====================================================
    print("\n\n" + "=" * 90)
    print("  KEY FINDINGS SUMMARY")
    print("=" * 90)
    
    # Track A findings
    if track_a:
        all_a_games = [g for games in track_a.values() for g in games]
        a_stats = compute_stats(all_a_games)
        print(f"\n🔬 Track A (Custom AlphaZero, {a_stats['total']} games):")
        print(f"   Overall Q-side score: {a_stats['score']:.3f}")
        if a_stats['score'] > 0.55:
            print("   → Queen side has clear advantage (concentration of material is better)")
        elif a_stats['score'] < 0.45:
            print("   → Dispersed side has clear advantage (dispersion compensates)")
        else:
            print("   → Results are very close / many draws (model may not differentiate well)")
    
    # Track B findings
    if track_b:
        all_b_games = [g for games in track_b.values() for g in games]
        b_stats = compute_stats(all_b_games)
        print(f"\n🔬 Track B Deterministic (lc0 n=800, {b_stats['total']} games):")
        print(f"   Overall Q-side score: {b_stats['score']:.3f}")
        
        # Per concentration
        for conc in CONC_ORDER:
            conc_games = [g for p in patterns if p in track_b and PATTERN_META[p]["concentration"] == conc for g in track_b[p]]
            if conc_games:
                cs = compute_stats(conc_games)
                print(f"   {conc:>8} concentration: Q-score = {cs['score']:.3f} (N={cs['total']})")
    
    if track_b_stochastic:
        all_bs_games = [g for games in track_b_stochastic.values() for g in games]
        bs_stats = compute_stats(all_bs_games)
        print(f"\n🔬 Track B Stochastic (lc0 n=800 + softmax, {bs_stats['total']} games):")
        print(f"   Overall Q-side score: {bs_stats['score']:.3f}")
        
        for conc in CONC_ORDER:
            conc_games = [g for p in patterns if p in track_b_stochastic and PATTERN_META[p]["concentration"] == conc for g in track_b_stochastic[p]]
            if conc_games:
                cs = compute_stats(conc_games)
                print(f"   {conc:>8} concentration: Q-score = {cs['score']:.3f} (N={cs['total']})")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
