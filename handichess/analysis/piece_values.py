"""
Effective piece value estimation via logistic regression.

Model:
  logit P(handicap side wins) = β₀ + βQ·ΔQ + βR·ΔR + βB·ΔB + βN·ΔN + βP·ΔP + β_color·color

Where Δ = count of each piece type removed by the handicap side.
β_piece estimates the effective value (in log-odds) of each piece type.

Caveat: All patterns sum to 9 points, so the design matrix is rank-deficient
for estimating absolute values. We can estimate *relative* values and need
an anchor (e.g., fix pawn = 1).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit


# Standard piece values for comparison
STANDARD_VALUES = {"Q": 9, "R": 5, "B": 3, "N": 3, "P": 1}


def prepare_regression_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix X and target y from game log DataFrame.

    Args:
        df: DataFrame with columns:
            - result_score (1.0, 0.5, 0.0)
            - material_diff (dict with Q, R, B, N, P counts)
            - handicap_side ("white" or "black")

    Returns:
        X: (n_games, 7) feature matrix [bias, ΔQ, ΔR, ΔB, ΔN, ΔP, color]
        y: (n_games,) binary outcome (1 = handicap wins, 0 = loss)
    """
    # Filter out draws for binary logistic regression, or treat draws as 0.5
    records = []
    for _, row in df.iterrows():
        # Material diff vector
        md = row.get("material_diff", {})
        if isinstance(md, str):
            import json
            md = json.loads(md)

        features = [
            1.0,  # intercept
            md.get("Q", 0),
            md.get("R", 0),
            md.get("B", 0),
            md.get("N", 0),
            md.get("P", 0),
            1.0 if row["handicap_side"] == "white" else 0.0,  # color
        ]
        records.append((features, row["result_score"]))

    X = np.array([r[0] for r in records], dtype=np.float64)
    y = np.array([r[1] for r in records], dtype=np.float64)

    return X, y


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    regularization: float = 0.01,
) -> dict:
    """
    Fit logistic regression to estimate effective piece values.

    Uses scipy minimize with L2 regularization.
    Handles draws by treating them as y=0.5 (fractional outcome).

    Args:
        X: Feature matrix (n, 7).
        y: Target outcomes in [0, 1].
        regularization: L2 penalty strength.

    Returns:
        Dict with:
          - coefficients: raw β values
          - piece_values: estimated piece values (relative to pawn)
          - standard_comparison: comparison to standard values
    """
    n_features = X.shape[1]

    def neg_log_likelihood(beta):
        logits = X @ beta
        # Clip for numerical stability
        logits = np.clip(logits, -20, 20)
        p = expit(logits)
        # Cross-entropy loss (handles y=0.5 correctly)
        eps = 1e-10
        ll = y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
        # L2 regularization (exclude intercept)
        reg = regularization * np.sum(beta[1:] ** 2)
        return -np.sum(ll) + reg

    # Initialize
    beta0 = np.zeros(n_features)
    result = minimize(neg_log_likelihood, beta0, method="L-BFGS-B")

    beta = result.x
    coeff_names = ["intercept", "ΔQ", "ΔR", "ΔB", "ΔN", "ΔP", "color"]

    coefficients = dict(zip(coeff_names, beta))

    # Convert to relative piece values (normalize so pawn = 1)
    piece_coeffs = {
        "Q": abs(beta[1]),
        "R": abs(beta[2]),
        "B": abs(beta[3]),
        "N": abs(beta[4]),
        "P": abs(beta[5]),
    }

    pawn_value = piece_coeffs["P"] if piece_coeffs["P"] > 0 else 1.0
    piece_values = {k: v / pawn_value for k, v in piece_coeffs.items()}

    # Comparison
    comparison = {}
    for piece in STANDARD_VALUES:
        comparison[piece] = {
            "standard": STANDARD_VALUES[piece],
            "estimated": round(piece_values.get(piece, 0), 2),
            "ratio": round(piece_values.get(piece, 0) / STANDARD_VALUES[piece], 2)
            if STANDARD_VALUES[piece] > 0 else None,
        }

    return {
        "coefficients": coefficients,
        "piece_values_raw": piece_coeffs,
        "piece_values_normalized": piece_values,
        "standard_comparison": comparison,
        "color_effect": beta[6],
        "converged": result.success,
        "message": result.message,
    }


def analyze_piece_values(df: pd.DataFrame, **kwargs) -> dict:
    """
    End-to-end piece value analysis from a game log DataFrame.

    Args:
        df: Game log DataFrame.

    Returns:
        Regression results dict.
    """
    X, y = prepare_regression_data(df)
    return fit_logistic_regression(X, y, **kwargs)


def print_piece_value_summary(results: dict) -> None:
    """Pretty-print piece value estimation results."""
    print("\n" + "=" * 60)
    print("Effective Piece Value Estimation")
    print("=" * 60)

    print("\nRaw Coefficients (log-odds):")
    for name, val in results["coefficients"].items():
        print(f"  {name:>12}: {val:+.4f}")

    print("\nNormalized Piece Values (Pawn = 1):")
    pv = results["piece_values_normalized"]
    print(f"  {'Piece':>6} {'Standard':>10} {'Estimated':>10} {'Ratio':>8}")
    print("  " + "-" * 36)
    for piece in ["Q", "R", "B", "N", "P"]:
        comp = results["standard_comparison"][piece]
        print(
            f"  {piece:>6} {comp['standard']:>10} "
            f"{comp['estimated']:>10.2f} {comp['ratio']:>8.2f}"
        )

    print(f"\nColor effect (white advantage): {results['color_effect']:+.4f}")
    print(f"Converged: {results['converged']}")
    print("=" * 60)
