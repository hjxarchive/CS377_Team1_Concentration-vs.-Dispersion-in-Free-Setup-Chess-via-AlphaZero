"""
Game log schema and I/O.

Each game is logged as a JSON object (one per line in a .jsonl file).
All analysis code consumes these logs exclusively.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator

import pandas as pd


@dataclass
class GameRecord:
    """
    Record of a single completed game.

    Fields:
        pattern_id:       Removal pattern identifier (e.g. "rook_bishop_pawn").
        q_side:           "white" or "black" — which side has the Queen.
        result:           Game result from the *q_side's* perspective:
                          "win", "draw", or "loss".
        result_score:     Numeric score: 1.0 (win), 0.5 (draw), 0.0 (loss).
        ply:              Total number of half-moves played.
        start_fen:        Starting FEN of the game.
        termination:      How the game ended: "checkmate", "stalemate",
                          "threefold", "fifty_moves", "insufficient",
                          "max_moves", "resignation".
        engine:           Engine/agent identifier (e.g. "lc0", "az_v3", "random").
        nodes:            Nodes/simulations per move (for reproducibility).
        timestamp:        ISO-format timestamp of game completion.
        extra:            Optional dict for additional metadata.
    """
    pattern_id: str
    q_side: str                 # "white" | "black"
    noq_side: str               # "white" | "black"
    result: str                 # "win" | "draw" | "loss"
    result_score: float         # 1.0 | 0.5 | 0.0
    ply: int
    start_fen: str
    termination: str
    engine: str
    nodes: int
    timestamp: str = ""
    extra: Optional[dict] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        # Validate
        assert self.q_side in ("white", "black"), (
            f"Invalid q_side: {self.q_side}"
        )
        assert self.noq_side in ("white", "black"), (
            f"Invalid noq_side: {self.noq_side}"
        )
        assert self.result in ("win", "draw", "loss"), (
            f"Invalid result: {self.result}"
        )
        assert self.result_score in (0.0, 0.5, 1.0), (
            f"Invalid result_score: {self.result_score}"
        )


def result_from_outcome(
    outcome: str,
    q_side: str,
) -> tuple[str, float]:
    """
    Convert a game outcome string to (result, result_score) from the
    Q side's perspective.

    Args:
        outcome: "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw).
        q_side: "white" or "black".

    Returns:
        (result, result_score) tuple.
    """
    if outcome == "1/2-1/2":
        return "draw", 0.5
    elif outcome == "1-0":
        if q_side == "white":
            return "win", 1.0
        else:
            return "loss", 0.0
    elif outcome == "0-1":
        if q_side == "black":
            return "win", 1.0
        else:
            return "loss", 0.0
    else:
        raise ValueError(f"Unknown outcome: {outcome}")


# ── I/O ──────────────────────────────────────────────────────────────────────

class GameLog:
    """
    Append-only game log backed by a JSONL file.

    Usage:
        log = GameLog("runs/experiment_1/games.jsonl")
        log.write(record)
        for record in log.read():
            ...
        df = log.to_dataframe()
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: GameRecord) -> None:
        """Append a single game record to the log file."""
        with open(self.path, "a") as f:
            d = asdict(record)
            # Remove None extra field to keep logs clean
            if d.get("extra") is None:
                del d["extra"]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def write_many(self, records: list[GameRecord]) -> None:
        """Append multiple game records at once."""
        with open(self.path, "a") as f:
            for record in records:
                d = asdict(record)
                if d.get("extra") is None:
                    del d["extra"]
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def read(self) -> Iterator[GameRecord]:
        """Read all game records from the log file."""
        if not self.path.exists():
            return
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield GameRecord(**d)

    def read_all(self) -> list[GameRecord]:
        """Read all records into a list."""
        return list(self.read())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the log to a pandas DataFrame for analysis."""
        records = self.read_all()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in records])

    def count(self) -> int:
        """Count total number of games in the log."""
        if not self.path.exists():
            return 0
        with open(self.path, "r") as f:
            return sum(1 for line in f if line.strip())

    def filter(
        self,
        pattern_id: Optional[str] = None,
        q_side: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> list[GameRecord]:
        """Filter records by pattern, side, or engine."""
        results = []
        for record in self.read():
            if pattern_id and record.pattern_id != pattern_id:
                continue
            if q_side and record.q_side != q_side:
                continue
            if engine and record.engine != engine:
                continue
            results.append(record)
        return results


def merge_logs(log_paths: list[str | Path], output_path: str | Path) -> GameLog:
    """Merge multiple log files into one."""
    output = GameLog(output_path)
    for path in log_paths:
        source = GameLog(path)
        for record in source.read():
            output.write(record)
    return output
