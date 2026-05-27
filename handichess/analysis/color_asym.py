"""
Black/white asymmetry analysis.

Measures the interaction between first-move advantage and material
handicap: does being handicapped as white vs. black make a difference?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .winrate import dirichlet_multinomial_posterior


def analyze_color_asymmetry(df: pd.DataFrame) -> dict:
    """
    Analyze win rates broken down by handicap color.

    For each pattern, compare the handicap-side score when the
    handicapped player is white vs. black.

    Args:
        df: Game log DataFrame with columns: pattern_id, handicap_side, result.

    Returns:
        Dict mapping pattern_id → {white: posterior, black: posterior, delta}.
    """
    results = {}

    for pid, group in df.groupby("pattern_id"):
        pattern_result = {}

        for side in ["white", "black"]:
            side_games = group[group["handicap_side"] == side]
            wins = (side_games["result"] == "win").sum()
            draws = (side_games["result"] == "draw").sum()
            losses = (side_games["result"] == "loss").sum()

            pattern_result[side] = dirichlet_multinomial_posterior(
                int(wins), int(draws), int(losses)
            )

        # Compute delta: white_score - black_score
        # Positive delta = being handicapped as white is easier
        # (i.e., first-move advantage partially compensates the handicap)
        w_score = pattern_result["white"]["mean_score"]
        b_score = pattern_result["black"]["mean_score"]
        pattern_result["delta"] = w_score - b_score
        pattern_result["interpretation"] = (
            "First-move advantage partially compensates handicap"
            if pattern_result["delta"] > 0.01
            else "First-move advantage does NOT compensate handicap"
            if pattern_result["delta"] < -0.01
            else "No significant color asymmetry"
        )

        results[pid] = pattern_result

    return results


def compute_aggregate_color_effect(df: pd.DataFrame) -> dict:
    """
    Compute the overall color effect across all patterns.

    Returns:
        Dict with aggregate white/black scores and delta.
    """
    white_games = df[df["handicap_side"] == "white"]
    black_games = df[df["handicap_side"] == "black"]

    w_wins = (white_games["result"] == "win").sum()
    w_draws = (white_games["result"] == "draw").sum()
    w_losses = (white_games["result"] == "loss").sum()

    b_wins = (black_games["result"] == "win").sum()
    b_draws = (black_games["result"] == "draw").sum()
    b_losses = (black_games["result"] == "loss").sum()

    w_post = dirichlet_multinomial_posterior(int(w_wins), int(w_draws), int(w_losses))
    b_post = dirichlet_multinomial_posterior(int(b_wins), int(b_draws), int(b_losses))

    return {
        "white_handicap_score": w_post["mean_score"],
        "black_handicap_score": b_post["mean_score"],
        "delta": w_post["mean_score"] - b_post["mean_score"],
        "white_ci": w_post["ci_95"],
        "black_ci": b_post["ci_95"],
        "white_n": w_post["total"],
        "black_n": b_post["total"],
    }


def print_color_asymmetry_summary(results: dict) -> None:
    """Pretty-print color asymmetry results."""
    print("\n" + "=" * 75)
    print("Color Asymmetry Analysis (Handicap Side Score)")
    print("=" * 75)
    print(f"{'Pattern':<25} {'White-H':>9} {'Black-H':>9} {'Delta':>8} {'Interpretation'}")
    print("-" * 75)

    for pid in sorted(results.keys()):
        r = results[pid]
        w = r["white"]["mean_score"]
        b = r["black"]["mean_score"]
        delta = r["delta"]
        interp = r["interpretation"]
        print(f"{pid:<25} {w:>9.3f} {b:>9.3f} {delta:>+8.3f}  {interp}")

    print("=" * 75)
