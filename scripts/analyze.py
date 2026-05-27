#!/usr/bin/env python3
"""Run Track C analysis on game logs."""

import argparse
import logging

from handichess.common.gamelog import GameLog
from handichess.analysis.winrate import (
    analyze_pattern_winrates,
    additivity_test,
    print_winrate_summary,
)
from handichess.analysis.piece_values import (
    analyze_piece_values,
    print_piece_value_summary,
)
from handichess.analysis.color_asym import (
    analyze_color_asymmetry,
    compute_aggregate_color_effect,
    print_color_asymmetry_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Analyze game results")
    parser.add_argument("log_path", type=str, help="Path to game log (.jsonl)")
    parser.add_argument("--skip-regression", action="store_true",
                        help="Skip piece value regression")
    args = parser.parse_args()

    log = GameLog(args.log_path)
    df = log.to_dataframe()

    if df.empty:
        print("No games found in log file.")
        return

    print(f"Loaded {len(df)} games from {args.log_path}")
    print(f"Patterns: {df['pattern_id'].unique().tolist()}")
    print(f"Engines: {df['engine'].unique().tolist()}")

    # 1. Win rate analysis
    print("\n" + "=" * 70)
    print("1. Win Rate Analysis")
    pattern_results = analyze_pattern_winrates(df)
    print_winrate_summary(pattern_results)

    # 2. Additivity test
    if len(pattern_results) >= 2:
        print("\n2. Additivity Test")
        add_test = additivity_test(pattern_results)
        print(f"  Score range: {add_test['score_range']:.4f}")
        print(f"  Homogeneous: {add_test['homogeneous']}")
        print(f"  Interpretation: {add_test['interpretation']}")

        if add_test["pairwise"]:
            print("\n  Pairwise comparisons:")
            for pair, probs in add_test["pairwise"].items():
                for k, v in probs.items():
                    print(f"    {k}: {v:.3f}")

    # 3. Color asymmetry
    print("\n3. Color Asymmetry Analysis")
    color_results = analyze_color_asymmetry(df)
    print_color_asymmetry_summary(color_results)

    agg = compute_aggregate_color_effect(df)
    print(f"\n  Aggregate: white-H={agg['white_handicap_score']:.3f}, "
          f"black-H={agg['black_handicap_score']:.3f}, "
          f"delta={agg['delta']:+.3f}")

    # 4. Piece value regression
    if not args.skip_regression and "material_diff" in df.columns:
        print("\n4. Effective Piece Value Estimation")
        try:
            pv_results = analyze_piece_values(df)
            print_piece_value_summary(pv_results)
        except Exception as e:
            print(f"  Regression failed: {e}")
            print("  (May need more diverse patterns for identifiability)")


if __name__ == "__main__":
    main()
