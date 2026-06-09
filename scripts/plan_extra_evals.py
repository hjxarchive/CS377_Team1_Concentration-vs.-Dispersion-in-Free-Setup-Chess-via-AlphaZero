#!/usr/bin/env python3
"""Plan +200 follow-up evaluations for statistically ambiguous patterns."""

from __future__ import annotations

import argparse
import json
import shlex
from collections import defaultdict
from pathlib import Path

from handichess.analysis.winrate import dirichlet_multinomial_posterior


DEFAULT_PATTERNS = [
    "rook_bishop_pawn",
    "rook_knight_pawn",
    "bishop_bishop_knight",
    "rook_4pawns",
    "bishop_knight_3pawns",
    "bishop_6pawns",
    "knight_6pawns",
]


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def counts_from_records(records: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0})

    for row in records:
        pattern = row.get("pattern_id") or row.get("pattern")
        if not pattern:
            continue

        if "result" in row:
            result = row["result"]
            if result == "win":
                counts[pattern]["wins"] += 1
            elif result == "draw":
                counts[pattern]["draws"] += 1
            elif result == "loss":
                counts[pattern]["losses"] += 1
            else:
                raise ValueError(f"Unknown result {result!r} in {pattern}")
            continue

        if {"wins", "draws", "losses"}.issubset(row):
            counts[pattern]["wins"] += int(row["wins"])
            counts[pattern]["draws"] += int(row["draws"])
            counts[pattern]["losses"] += int(row["losses"])

    return dict(counts)


def summarize_patterns(
    counts_by_pattern: dict[str, dict[str, int]],
    threshold: float,
    min_effect: float,
    min_games: int,
    max_games: int,
) -> tuple[dict[str, dict], list[str]]:
    summaries = {}
    ambiguous = []

    for pattern in sorted(counts_by_pattern):
        c = counts_by_pattern[pattern]
        posterior = dirichlet_multinomial_posterior(c["wins"], c["draws"], c["losses"])
        mean = posterior["mean_score"]
        ci_low, ci_high = posterior["ci_95"]
        total = posterior["total"]
        ci_crosses_threshold = ci_low <= threshold <= ci_high
        small_effect = abs(mean - threshold) < min_effect
        needs_more = total >= min_games and total < max_games
        is_ambiguous = needs_more and (ci_crosses_threshold or small_effect)

        summaries[pattern] = {
            **posterior,
            "ambiguous": is_ambiguous,
            "ci_crosses_threshold": ci_crosses_threshold,
            "small_effect": small_effect,
        }
        if is_ambiguous:
            ambiguous.append(pattern)

    return summaries, ambiguous


def lc0_commands(args: argparse.Namespace, patterns: list[str]) -> list[str]:
    commands = []
    for pattern in patterns:
        commands.append(shell_join([
            "python", "scripts/run_lc0.py",
            "--engine", args.engine,
            "--nodes", args.nodes,
            "--games", args.extra_games,
            "--pattern", pattern,
            "--output", args.output_log,
        ]))
    return commands


def arena_commands(args: argparse.Namespace, patterns: list[str]) -> list[str]:
    commands = []
    devices = args.devices or ["cpu"]
    noq_colors = args.noq_colors or ["white", "black"]
    games_per_color = args.extra_games // len(noq_colors)
    if games_per_color * len(noq_colors) != args.extra_games:
        raise ValueError("--extra-games must divide evenly across --noq-colors")
    idx = 0
    for pattern in patterns:
        for baseline in args.baselines:
            for noq_color in noq_colors:
                device = devices[idx % len(devices)]
                seed = args.seed + idx
                idx += 1
                commands.append(shell_join([
                    "conda", "run", "-n", args.conda_env,
                    "python", "scripts/run_arena.py",
                    "--game", "chess",
                    "--checkpoint", args.checkpoint,
                    "--baseline", baseline,
                    "--games", games_per_color,
                    "--pattern", pattern,
                    "--noq-color", noq_color,
                    "--device", device,
                    "--num-simulations", args.num_simulations,
                    "--seed", seed,
                    "--output", args.output_log,
                ]))
    return commands


def print_summary(summaries: dict[str, dict], ambiguous: list[str]) -> None:
    print(f"{'Pattern':<25} {'N':>5} {'Score':>7} {'95% CI':>18} {'Decision':>12}")
    print("-" * 72)
    for pattern in sorted(summaries):
        s = summaries[pattern]
        ci_low, ci_high = s["ci_95"]
        decision = "+200" if s["ambiguous"] else "keep"
        print(
            f"{pattern:<25} {s['total']:>5} {s['mean_score']:>7.3f} "
            f"[{ci_low:.3f}, {ci_high:.3f}] {decision:>12}"
        )
    print("-" * 72)
    print("Ambiguous patterns:", ", ".join(ambiguous) if ambiguous else "(none)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select patterns needing +200 follow-up games and emit run commands."
    )
    parser.add_argument("--input", required=True,
                        help="JSONL input: game-level GameLog or arena summary JSONL")
    parser.add_argument("--track", choices=["lc0", "arena"], required=True,
                        help="Which command template to emit")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Null score threshold")
    parser.add_argument("--min-effect", type=float, default=0.07,
                        help="Treat |score-threshold| below this as ambiguous")
    parser.add_argument("--min-games", type=int, default=200,
                        help="Only consider patterns with at least this many games")
    parser.add_argument("--max-games", type=int, default=400,
                        help="Do not schedule patterns already at this many games")
    parser.add_argument("--extra-games", type=int, default=200,
                        help="Follow-up games per selected pattern")
    parser.add_argument("--output-log", required=True,
                        help="JSONL path where follow-up results should be appended")
    parser.add_argument("--commands-out", default=None,
                        help="Optional shell script path for emitted commands")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Commented xargs parallelism hint in command script")

    # LC0 command options.
    parser.add_argument("--engine", default="lc0")
    parser.add_argument("--nodes", type=int, default=800)

    # Arena command options.
    parser.add_argument("--checkpoint", default="runs/checkpoints/final.pt")
    parser.add_argument("--baselines", nargs="+", default=["greedy", "weak_mcts"])
    parser.add_argument("--devices", nargs="+", default=["cuda:4", "cuda:5", "cuda:6", "cuda:7"])
    parser.add_argument("--noq-colors", nargs="+", default=["white", "black"])
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conda-env", default="handichess")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    counts = counts_from_records(records)
    summaries, ambiguous = summarize_patterns(
        counts,
        threshold=args.threshold,
        min_effect=args.min_effect,
        min_games=args.min_games,
        max_games=args.max_games,
    )

    print_summary(summaries, ambiguous)

    if args.track == "lc0":
        commands = lc0_commands(args, ambiguous)
    else:
        commands = arena_commands(args, ambiguous)

    if not commands:
        return

    print("\nCommands:")
    for cmd in commands:
        print(cmd)

    if args.commands_out:
        out = Path(args.commands_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("set -euo pipefail\n\n")
            f.write(f"parallel={args.parallel}\n")
            f.write("running=0\n\n")
            f.write("run_job() {\n")
            f.write("  local cmd=\"$1\"\n")
            f.write("  echo \"[start] $cmd\"\n")
            f.write("  bash -lc \"$cmd\"\n")
            f.write("}\n\n")
            for cmd in commands:
                f.write(f"run_job {shlex.quote(cmd)} &\n")
                f.write("running=$((running + 1))\n")
                f.write("if (( running >= parallel )); then\n")
                f.write("  wait -n\n")
                f.write("  running=$((running - 1))\n")
                f.write("fi\n\n")
            f.write("wait\n")
        out.chmod(0o755)
        print(f"\nWrote commands to {out}")


if __name__ == "__main__":
    main()
