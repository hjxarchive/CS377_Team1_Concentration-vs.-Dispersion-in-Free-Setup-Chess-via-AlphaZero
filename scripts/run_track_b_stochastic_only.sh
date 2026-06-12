#!/bin/bash
set -euo pipefail

PATTERNS=(rook_bishop_pawn rook_knight_pawn bishop_bishop_knight rook_4pawns bishop_knight_3pawns bishop_6pawns knight_6pawns)

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
LC0_PATH=${LC0_PATH:-./lc0}
LC0_WEIGHTS=${LC0_WEIGHTS:-runs/checkpoints/lc0_weights.pb.gz}
LC0_BACKEND=${LC0_BACKEND:-cuda-auto}
TRACK_B_GAMES=${TRACK_B_GAMES:-400}
TRACK_B_NODES=${TRACK_B_NODES:-800}
STOCHASTIC_PLIES=${STOCHASTIC_PLIES:-10}
MULTIPV=${MULTIPV:-5}
SCORE_TEMPERATURE_CP=${SCORE_TEMPERATURE_CP:-120}

echo "========================================="
echo " Starting Stochastic Track B (LC0) Evaluation "
echo "========================================="
gpu_idx=0
for pattern in "${PATTERNS[@]}"; do
  echo "Launching stochastic Track B pattern: $pattern on GPU $gpu_idx"

  CUDA_VISIBLE_DEVICES=$gpu_idx "$PYTHON_BIN" scripts/run_lc0.py \
    --pattern "$pattern" \
    --games "$TRACK_B_GAMES" \
    --nodes "$TRACK_B_NODES" \
    --lc0-path "$LC0_PATH" \
    --weights-path "$LC0_WEIGHTS" \
    --backend "$LC0_BACKEND" \
    --stochastic-plies "$STOCHASTIC_PLIES" \
    --multipv "$MULTIPV" \
    --score-temperature-cp "$SCORE_TEMPERATURE_CP" \
    --seed "$((377 + gpu_idx))" \
    --output "runs/results/track_b_stochastic_${pattern}.jsonl" &

  gpu_idx=$(( (gpu_idx + 1) % 4 ))

  if [ $gpu_idx -eq 0 ]; then
    wait
  fi
done
wait

echo "Stochastic Track B evaluation finished!"
