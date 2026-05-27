"""
lc0 game runner — Track B pipeline.

Runs self-play games using the lc0 engine (via UCI protocol) from
handicap starting positions. This is the "strong reference" track
that produces the main scientific results.

Requires:
  - lc0 binary installed and accessible
  - A neural network weights file (.pb.gz)
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Optional

import chess
import chess.engine

from handichess.common.handicap import (
    get_patterns,
    get_pattern_by_id,
    generate_position,
)
from handichess.common.gamelog import GameLog, GameRecord, result_from_outcome

logger = logging.getLogger(__name__)


class Lc0Runner:
    """
    Runs lc0 self-play games from handicap positions.

    Two lc0 instances play against each other (or one instance alternates),
    with fixed nodes per move for strength control.
    """

    def __init__(
        self,
        engine_path: str = "lc0",
        weights_path: str = "",
        nodes: int = 800,
        threads: int = 1,
        backend: str = "cpu",
        exploration_plies: int = 16,
        temperature: float = 1.0,
    ):
        self.engine_path = engine_path
        self.weights_path = weights_path
        self.nodes = nodes
        self.threads = threads
        self.backend = backend
        self.exploration_plies = exploration_plies
        self.temperature = temperature

    def _start_engine(self) -> chess.engine.SimpleEngine:
        """Start an lc0 UCI engine instance."""
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

        # Configure engine
        options = {
            "Threads": self.threads,
            "Backend": self.backend,
        }
        if self.weights_path:
            options["WeightsFile"] = self.weights_path

        for key, value in options.items():
            try:
                engine.configure({key: value})
            except chess.engine.EngineError:
                logger.warning(f"Could not set engine option: {key}={value}")

        return engine

    def play_game(
        self,
        start_fen: str,
        pattern_id: str,
        q_side: str,
    ) -> GameRecord:
        """
        Play a single game from a match-up position.

        Both sides are played by the same lc0 instance (self-play).

        Args:
            start_fen: Starting FEN string.
            pattern_id: Pattern identifier for logging.
            q_side: "white" or "black" (the side with the Queen).

        Returns:
            GameRecord with the game result.
        """
        engine = self._start_engine()

        try:
            board = chess.Board(start_fen)
            ply = 0

            while not board.is_game_over(claim_draw=True):
                if ply >= 512:
                    break

                # Set search limit
                limit = chess.engine.Limit(nodes=self.nodes)

                # Play move
                result = engine.play(board, limit)
                if result.move is None:
                    break

                board.push(result.move)
                ply += 1

            # Determine outcome
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                # Max moves reached
                outcome_str = "1/2-1/2"
                termination = "max_moves"
            else:
                if outcome.winner is None:
                    outcome_str = "1/2-1/2"
                elif outcome.winner == chess.WHITE:
                    outcome_str = "1-0"
                else:
                    outcome_str = "0-1"
                termination = outcome.termination.name.lower()

            result_str, result_score = result_from_outcome(outcome_str, q_side)

            return GameRecord(
                pattern_id=pattern_id,
                q_side=q_side,
                result=result_str,
                result_score=result_score,
                ply=ply,
                start_fen=start_fen,
                termination=termination,
                engine=f"lc0_n{self.nodes}",
                nodes=self.nodes,
            )

        finally:
            engine.quit()

    def run_pattern(
        self,
        pattern_id: str,
        num_games: int,
        log: GameLog,
    ) -> dict:
        """
        Run multiple games for a single pattern, color-balanced.

        Args:
            pattern_id: Pattern to evaluate.
            num_games: Total games (split evenly between white/black handicap).
            log: GameLog to write results to.

        Returns:
            Summary statistics.
        """
        pattern = get_pattern_by_id(pattern_id)
        games_per_side = num_games // 2

        wins = draws = losses = 0

        for noq_color in ["white", "black"]:
            pos = generate_position(pattern, chess.WHITE if noq_color == "white" else chess.BLACK)
            q_side = "black" if noq_color == "white" else "white"

            for g in range(games_per_side):
                logger.info(
                    f"Game {g+1}/{games_per_side} | "
                    f"pattern={pattern_id} | noq_color={noq_color} (q_side={q_side})"
                )

                record = self.play_game(pos.fen, pattern_id, q_side)
                log.write(record)

                if record.result == "win":
                    wins += 1
                elif record.result == "draw":
                    draws += 1
                else:
                    losses += 1

                logger.info(
                    f"  Result: {record.result} in {record.ply} ply "
                    f"({record.termination})"
                )

        total = wins + draws + losses
        stats = {
            "pattern_id": pattern_id,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": (wins + 0.5 * draws) / max(total, 1),
            "total": total,
        }
        logger.info(
            f"Pattern {pattern_id} complete: "
            f"{wins}W/{draws}D/{losses}L = {stats['score']:.3f}"
        )
        return stats

    def run_all_patterns(
        self,
        num_games_per_pattern: int,
        log_path: str,
        phase: Optional[int] = None,
    ) -> list[dict]:
        """
        Run games for all patterns (or a specific phase).

        Args:
            num_games_per_pattern: Games per pattern.
            log_path: Path for the game log file.
            phase: Optional phase filter (1 or 2).

        Returns:
            List of per-pattern statistics.
        """
        log = GameLog(log_path)
        patterns = get_patterns(phase=phase)
        all_stats = []

        for pattern in patterns:
            stats = self.run_pattern(
                pattern.pattern_id,
                num_games_per_pattern,
                log,
            )
            all_stats.append(stats)

        return all_stats

    def evaluate_position(self, fen: str, nodes: Optional[int] = None) -> dict:
        """
        Get lc0's evaluation of a position (for sanity checking).

        Args:
            fen: Position FEN.
            nodes: Override nodes for this evaluation.

        Returns:
            Dict with score and principal variation.
        """
        engine = self._start_engine()
        try:
            board = chess.Board(fen)
            limit = chess.engine.Limit(nodes=nodes or self.nodes)
            info = engine.analyse(board, limit)

            score = info.get("score")
            pv = info.get("pv", [])

            return {
                "fen": fen,
                "score_cp": score.white().score(mate_score=10000) if score else None,
                "score_mate": score.white().mate() if score and score.white().mate() is not None else None,
                "pv": [m.uci() for m in pv[:5]],
                "depth": info.get("depth"),
                "nodes": info.get("nodes"),
            }
        finally:
            engine.quit()
