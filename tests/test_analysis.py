"""Tests for analysis modules (synthetic data verification)."""

import numpy as np
import pandas as pd
import pytest

from handichess.analysis.winrate import (
    compute_win_rate,
    dirichlet_multinomial_posterior,
    analyze_pattern_winrates,
    additivity_test,
)
from handichess.analysis.piece_values import (
    prepare_regression_data,
    fit_logistic_regression,
)
from handichess.analysis.color_asym import (
    analyze_color_asymmetry,
    compute_aggregate_color_effect,
)


class TestWinRate:
    """Test win rate computation."""

    def test_all_wins(self):
        assert compute_win_rate(10, 0, 0) == 1.0

    def test_all_losses(self):
        assert compute_win_rate(0, 0, 10) == 0.0

    def test_all_draws(self):
        assert compute_win_rate(0, 10, 0) == 0.5

    def test_mixed(self):
        # 5W + 2D + 3L → (5 + 1) / 10 = 0.6
        assert abs(compute_win_rate(5, 2, 3) - 0.6) < 1e-10

    def test_empty(self):
        assert compute_win_rate(0, 0, 0) == 0.0


class TestDirichletPosterior:
    """Test Dirichlet-multinomial posterior estimation."""

    def test_basic_structure(self):
        result = dirichlet_multinomial_posterior(10, 5, 15)
        assert "mean_score" in result
        assert "ci_95" in result
        assert "mean_probs" in result
        assert "counts" in result

    def test_strong_win_signal(self):
        """Many wins should give a high mean score."""
        result = dirichlet_multinomial_posterior(100, 0, 0)
        assert result["mean_score"] > 0.9

    def test_strong_loss_signal(self):
        """Many losses should give a low mean score."""
        result = dirichlet_multinomial_posterior(0, 0, 100)
        assert result["mean_score"] < 0.1

    def test_balanced(self):
        """Equal W/L should give score near 0.5."""
        result = dirichlet_multinomial_posterior(50, 0, 50)
        assert 0.4 < result["mean_score"] < 0.6

    def test_ci_ordering(self):
        """Lower CI bound < mean < upper CI bound."""
        result = dirichlet_multinomial_posterior(30, 10, 60)
        ci_low, ci_high = result["ci_95"]
        assert ci_low < result["mean_score"] < ci_high

    def test_more_data_tighter_ci(self):
        """More data should narrow the credible interval."""
        small = dirichlet_multinomial_posterior(5, 2, 3)
        large = dirichlet_multinomial_posterior(50, 20, 30)
        small_width = small["ci_95"][1] - small["ci_95"][0]
        large_width = large["ci_95"][1] - large["ci_95"][0]
        assert large_width < small_width


class TestAdditivity:
    """Test additivity test with synthetic data."""

    def test_homogeneous_patterns(self):
        """Same win rates → additivity should hold."""
        results = {}
        for pid in ["A", "B", "C"]:
            results[pid] = dirichlet_multinomial_posterior(30, 10, 60)

        test = additivity_test(results)
        assert test["homogeneous"] is True

    def test_heterogeneous_patterns(self):
        """Very different win rates → additivity should fail."""
        results = {
            "A": dirichlet_multinomial_posterior(90, 5, 5),
            "B": dirichlet_multinomial_posterior(5, 5, 90),
        }
        test = additivity_test(results)
        assert test["homogeneous"] is False
        assert test["score_range"] > 0.5


class TestColorAsymmetry:
    """Test color asymmetry analysis."""

    def test_basic_asymmetry(self):
        """Create synthetic data with known asymmetry."""
        data = []
        for _ in range(50):
            data.append({"pattern_id": "test", "handicap_side": "white", "result": "win"})
        for _ in range(50):
            data.append({"pattern_id": "test", "handicap_side": "black", "result": "loss"})

        df = pd.DataFrame(data)
        results = analyze_color_asymmetry(df)

        assert "test" in results
        assert results["test"]["delta"] > 0.5  # White much better

    def test_aggregate_effect(self):
        data = []
        for _ in range(30):
            data.append({"pattern_id": "p1", "handicap_side": "white", "result": "win"})
            data.append({"pattern_id": "p1", "handicap_side": "black", "result": "loss"})

        df = pd.DataFrame(data)
        agg = compute_aggregate_color_effect(df)
        assert agg["delta"] > 0
