#!/usr/bin/env python3
"""Run lc0 self-play games from handicap positions (Track B)."""

import argparse
import logging

from handichess.track_b.lc0_runner import Lc0Runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run lc0 handicap games")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Specific pattern ID (default: all Phase 1)")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per pattern")
    parser.add_argument("--nodes", type=int, default=800,
                        help="Nodes per move")
    parser.add_argument("--engine", "--lc0-path", dest="engine", type=str, default="lc0",
                        help="Path to lc0 binary")
    parser.add_argument("--weights", "--weights-path", dest="weights", type=str, default="",
                        help="Path to lc0 weights file")
    parser.add_argument("--backend", type=str, default="cuda-auto",
                        help="Lc0 neural network backend")
    parser.add_argument("--threads", type=int, default=1,
                        help="Lc0 CPU worker threads")
    parser.add_argument("--stochastic-plies", type=int, default=0,
                        help="Use MultiPV sampling for the first N half-moves")
    parser.add_argument("--multipv", type=int, default=1,
                        help="Number of LC0 principal variations to sample from")
    parser.add_argument("--score-temperature-cp", type=float, default=120.0,
                        help="Softmax temperature in centipawns for MultiPV sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stochastic move sampling")
    parser.add_argument("--output", type=str, default="runs/lc0_games.jsonl",
                        help="Output log file")
    parser.add_argument("--phase", type=int, default=1,
                        help="Phase filter (1 or 2)")
    args = parser.parse_args()

    runner = Lc0Runner(
        engine_path=args.engine,
        weights_path=args.weights,
        nodes=args.nodes,
        threads=args.threads,
        backend=args.backend,
        stochastic_plies=args.stochastic_plies,
        multipv=args.multipv,
        score_temperature_cp=args.score_temperature_cp,
        seed=args.seed,
    )

    if args.pattern:
        from handichess.common.gamelog import GameLog
        log = GameLog(args.output)
        runner.run_pattern(args.pattern, args.games, log)
    else:
        runner.run_all_patterns(args.games, args.output, phase=args.phase)


if __name__ == "__main__":
    main()
