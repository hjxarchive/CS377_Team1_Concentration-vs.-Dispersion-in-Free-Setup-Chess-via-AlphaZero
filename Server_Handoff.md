# 🛠️ Track A Evaluation Fixes & Server-Side Instructions

This document summarizes the changes made to the `Track A` (AlphaZero from scratch) evaluation pipeline to solve the "100% exact draw phenomenon" and missing endgame resolution.

## 📌 The Problem
When evaluating the trained AlphaZero network via `evaluate_matchup_patterns` (400 games):
1. **100% Exact Draws**: The games were completely deterministic. The agent picked the exact same sequence of moves for all 400 games, resulting in 400 identical draws.
2. **Missing PGNs**: The `GameLog` was saving win/draw/loss counts, but the actual moves (PGN) were not being saved, making it impossible to analyze *why* the agent drew.

## ✅ What Was Fixed

### 1. Added Early-Game Diversity (Temperature & Noise)
**File**: `handichess/track_a/arena.py`
- Added probabilistic sampling during the first 10 half-moves.
- Set `temperature=1.0` and `add_noise=True` for `move_count < 10`.
- After 10 half-moves, it gracefully switches back to `temperature=0.0` (deterministic `argmax`).
- *Result: Every single game now branches into a completely unique game tree, allowing for a statistically sound 400-game evaluation.*

### 2. Max Moves Rollback (180 Half-Moves)
**File**: `handichess/track_a/game/chess_std.py`
- We temporarily tested increasing `max_moves` to 400, but rolled it back to `180` half-moves (90 full moves) as originally designed by the research team to save CPU/GPU compute time.
- *Result: Games that drag on due to the agent's endgame shuffling (a known limitation of non-tablebase AlphaZero) will safely terminate and log as a draw.*

### 3. Automatic PGN Saving (Track A & Track B)
**Files**: `handichess/track_a/arena.py` & `handichess/track_b/lc0_runner.py`
- Integrated `chess.pgn` into the evaluation loops for both Track A (AlphaZero) and Track B (Lc0).
- The `GameRecord` object now stores the full PGN string in the `extra` field for every played game across all tracks.
- *Result: The JSONL log files will now contain `"extra": {"pgn": "..."}` for every game, allowing you to easily parse and visualize the games in Lichess or Chessbase.*

---

## 🚀 Instructions for Server-Side Execution

The codebase is now fully patched and ready for the large-scale evaluation. The master execution script has been updated to use the fixed, PGN-saving Track A pipeline.

**Next Steps for the Server Agent:**
1. Execute the master bash script: `bash scripts/run_all_experiments.sh`
   *(This single script will now automatically manage 4 GPUs, distribute the load for all 7 handicap patterns, run Track A using the newly integrated `eval_track_a.py`, run Track B via Lc0, and finally run the ELO calibration.)*
2. **Monitor the Logs**: Ensure that the 7 pattern-specific log files (e.g., `runs/results/track_a_rook_bishop_pawn.jsonl`, `track_b_...jsonl`) are being populated correctly and that the `"extra": {"pgn": "..."}` strings are present.
3. Once the evaluations finish (which may take hours on the cluster), you can extract the PGN strings from the 14 output JSONL files to perform qualitative analysis of the games.

> [!TIP]
> The PGN string is saved in the `extra` dictionary for every game. In Python, you can extract all PGNs from the log file via `json.loads(line)['extra']['pgn']`.
