"""
Win rate analysis with Dirichlet-multinomial posterior estimation.

Estimates handicap-side win rates per removal pattern, with credible
intervals, and tests whether the win rates are the same across patterns
(additivity test: "is 9 points the same however you remove them?").
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def compute_win_rate(wins: int, draws: int, losses: int) -> float:
    """
    Compute the score (win rate) for the handicap side.
    Score = (wins + 0.5 * draws) / total.
    """
    total = wins + draws + losses
    if total == 0:
        return 0.0
    return (wins + 0.5 * draws) / total


def dirichlet_multinomial_posterior(
    wins: int,
    draws: int,
    losses: int,
    prior_alpha: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_samples: int = 50_000,
    seed: int = 42,
) -> dict:
    """
    Bayesian estimation of win/draw/loss probabilities using
    a Dirichlet-multinomial model.

    Prior: Dirichlet(alpha)  — default uniform (1,1,1).
    Posterior: Dirichlet(alpha + counts).

    Args:
        wins, draws, losses: Observed counts.
        prior_alpha: Dirichlet prior parameters (W, D, L).
        num_samples: Number of posterior samples.
        seed: Random seed.

    Returns:
        Dict with posterior summaries:
          - mean_score: expected score (E[W] + 0.5*E[D])
          - ci_95: 95% credible interval for the score
          - mean_probs: (P(W), P(D), P(L)) posterior means
          - samples: raw posterior samples of (pw, pd, pl)
    """
    rng = np.random.default_rng(seed)

    # Posterior Dirichlet parameters
    alpha_post = np.array([
        prior_alpha[0] + wins,
        prior_alpha[1] + draws,
        prior_alpha[2] + losses,
    ])

    # Sample from posterior
    samples = rng.dirichlet(alpha_post, size=num_samples)
    pw, pd, pl = samples[:, 0], samples[:, 1], samples[:, 2]

    # Score = P(W) + 0.5 * P(D)
    scores = pw + 0.5 * pd

    return {
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "ci_95": (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))),
        "std_score": float(scores.std()),
        "mean_probs": {
            "win": float(pw.mean()),
            "draw": float(pd.mean()),
            "loss": float(pl.mean()),
        },
        "counts": {"wins": wins, "draws": draws, "losses": losses},
        "total": wins + draws + losses,
    }


def analyze_pattern_winrates(
    df: pd.DataFrame,
    prior_alpha: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """
    Analyze win rates for each pattern in the game log DataFrame.

    Args:
        df: DataFrame with columns: pattern_id, result ("win"/"draw"/"loss").
        prior_alpha: Dirichlet prior.

    Returns:
        Dict mapping pattern_id → posterior summary.
    """
    results = {}
    for pid, group in df.groupby("pattern_id"):
        wins = (group["result"] == "win").sum()
        draws = (group["result"] == "draw").sum()
        losses = (group["result"] == "loss").sum()
        results[pid] = dirichlet_multinomial_posterior(
            int(wins), int(draws), int(losses), prior_alpha
        )
    return results


def additivity_test(
    pattern_results: dict,
    num_samples: int = 50_000,
    seed: int = 42,
) -> dict:
    """
    Test whether the handicap-side score is the same across all patterns
    (i.e., "9 points is 9 points regardless of how you remove them").

    Method: Compare posterior score distributions pairwise.
    Compute P(score_i > score_j) for all pairs.

    Also computes the range (max - min) of posterior mean scores
    as a practical effect size.

    Args:
        pattern_results: Dict from analyze_pattern_winrates().
        num_samples: Posterior samples for comparison.
        seed: Random seed.

    Returns:
        Dict with:
          - pairwise_prob: P(score_i > score_j) for all pairs
          - score_range: max - min of mean scores
          - homogeneous: bool, True if all scores overlap (simple heuristic)
    """
    rng = np.random.default_rng(seed)
    pattern_ids = sorted(pattern_results.keys())

    # Re-sample scores for each pattern
    pattern_scores = {}
    for pid in pattern_ids:
        r = pattern_results[pid]
        c = r["counts"]
        alpha = np.array([1.0 + c["wins"], 1.0 + c["draws"], 1.0 + c["losses"]])
        samples = rng.dirichlet(alpha, size=num_samples)
        pattern_scores[pid] = samples[:, 0] + 0.5 * samples[:, 1]

    # Pairwise comparisons
    pairwise = {}
    for i, p1 in enumerate(pattern_ids):
        for p2 in pattern_ids[i + 1:]:
            prob_gt = float((pattern_scores[p1] > pattern_scores[p2]).mean())
            pairwise[f"{p1}_vs_{p2}"] = {
                "P(score_{p1} > score_{p2})": prob_gt,
                "P(score_{p2} > score_{p1})": 1.0 - prob_gt,
            }

    # Effect size
    mean_scores = {pid: pattern_scores[pid].mean() for pid in pattern_ids}
    score_range = max(mean_scores.values()) - min(mean_scores.values())

    # Simple homogeneity heuristic: all 95% CIs overlap
    cis = {pid: (np.percentile(pattern_scores[pid], 2.5),
                 np.percentile(pattern_scores[pid], 97.5)) for pid in pattern_ids}

    # Check if any CI is completely outside another
    homogeneous = True
    for i, p1 in enumerate(pattern_ids):
        for p2 in pattern_ids[i + 1:]:
            if cis[p1][1] < cis[p2][0] or cis[p2][1] < cis[p1][0]:
                homogeneous = False
                break

    return {
        "pairwise": pairwise,
        "mean_scores": mean_scores,
        "score_range": float(score_range),
        "homogeneous": homogeneous,
        "interpretation": (
            "Additivity holds: 9 points is equivalent regardless of removal pattern"
            if homogeneous else
            "Additivity violated: some removal patterns yield different effective handicaps"
        ),
    }


def print_winrate_summary(pattern_results: dict) -> None:
    """Pretty-print win rate results."""
    print("\n" + "=" * 70)
    print("Win Rate Analysis (Handicap Side)")
    print("=" * 70)
    print(f"{'Pattern':<25} {'Score':>7} {'95% CI':>18} {'W':>5} {'D':>5} {'L':>5} {'N':>5}")
    print("-" * 70)

    for pid in sorted(pattern_results.keys()):
        r = pattern_results[pid]
        ci = r["ci_95"]
        c = r["counts"]
        print(
            f"{pid:<25} {r['mean_score']:>7.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] "
            f"{c['wins']:>5} {c['draws']:>5} {c['losses']:>5} "
            f"{r['total']:>5}"
        )
    print("=" * 70)
