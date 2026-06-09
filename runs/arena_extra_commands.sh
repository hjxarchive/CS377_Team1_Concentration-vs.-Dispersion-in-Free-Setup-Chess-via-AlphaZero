#!/usr/bin/env bash
set -euo pipefail

parallel=4
running=0

run_job() {
  local cmd="$1"
  echo "[start] $cmd"
  bash -lc "$cmd"
}

run_job 'conda run -n handichess python scripts/run_arena.py --game chess --checkpoint runs/checkpoints/final.pt --baseline greedy --games 200 --pattern rook_bishop_pawn --device cuda:4 --num-simulations 200 --seed 42 --output runs/arena_extra.jsonl' &
running=$((running + 1))
if (( running >= parallel )); then
  wait -n
  running=$((running - 1))
fi

run_job 'conda run -n handichess python scripts/run_arena.py --game chess --checkpoint runs/checkpoints/final.pt --baseline weak_mcts --games 200 --pattern rook_bishop_pawn --device cuda:5 --num-simulations 200 --seed 42 --output runs/arena_extra.jsonl' &
running=$((running + 1))
if (( running >= parallel )); then
  wait -n
  running=$((running - 1))
fi

wait
