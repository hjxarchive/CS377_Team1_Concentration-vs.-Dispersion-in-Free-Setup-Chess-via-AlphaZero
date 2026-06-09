#!/usr/bin/env bash
set -euo pipefail

parallel=4
running=0

checkpoint=${CHECKPOINT:-runs/checkpoints/final.pt}
output=${OUTPUT:-runs/arena_main.jsonl}
games_per_color=${GAMES_PER_COLOR:-100}
simulations=${SIMULATIONS:-200}
conda_env=${CONDA_ENV:-handichess}

patterns=(
  rook_bishop_pawn
  rook_knight_pawn
  bishop_bishop_knight
  rook_4pawns
  bishop_knight_3pawns
  bishop_6pawns
  knight_6pawns
)

baselines=(
  greedy
  weak_mcts
)

noq_colors=(
  white
  black
)

devices=(
  cuda:4
  cuda:5
  cuda:6
  cuda:7
)

run_job() {
  local pattern="$1"
  local baseline="$2"
  local noq_color="$3"
  local device="$4"
  local seed="$5"

  echo "[start] pattern=${pattern} baseline=${baseline} noq_color=${noq_color} device=${device} seed=${seed}"
  conda run -n "${conda_env}" python scripts/run_arena.py \
    --game chess \
    --checkpoint "${checkpoint}" \
    --baseline "${baseline}" \
    --games "${games_per_color}" \
    --pattern "${pattern}" \
    --noq-color "${noq_color}" \
    --device "${device}" \
    --num-simulations "${simulations}" \
    --seed "${seed}" \
    --output "${output}"
  echo "[done] pattern=${pattern} baseline=${baseline} noq_color=${noq_color} device=${device} seed=${seed}"
}

idx=0
for pattern in "${patterns[@]}"; do
  for baseline in "${baselines[@]}"; do
    for noq_color in "${noq_colors[@]}"; do
      device="${devices[$((idx % ${#devices[@]}))]}"
      seed=$((42 + idx))
      run_job "${pattern}" "${baseline}" "${noq_color}" "${device}" "${seed}" &
      running=$((running + 1))
      idx=$((idx + 1))

      if (( running >= parallel )); then
        wait -n
        running=$((running - 1))
      fi
    done
  done
done

wait
echo "[all done] wrote ${output}"
