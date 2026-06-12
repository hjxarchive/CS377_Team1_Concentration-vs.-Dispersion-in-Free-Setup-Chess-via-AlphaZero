#!/bin/bash
set -euo pipefail

PATTERNS=(rook_bishop_pawn rook_knight_pawn bishop_bishop_knight rook_4pawns bishop_knight_3pawns bishop_6pawns knight_6pawns)

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
LC0_PATH=${LC0_PATH:-./lc0}
LC0_WEIGHTS=${LC0_WEIGHTS:-runs/checkpoints/lc0_weights.pb.gz}
LC0_BACKEND=${LC0_BACKEND:-cuda-auto}
TRACK_B_GAMES=${TRACK_B_GAMES:-400}
TRACK_B_NODES=${TRACK_B_NODES:-800}
ELO_GAMES=${ELO_GAMES:-20}
ELO_SIMS=${ELO_SIMS:-400}

echo "========================================="
echo " Starting Track B (LC0) Evaluation "
echo "========================================="
gpu_idx=0
for pattern in "${PATTERNS[@]}"; do
  echo "Launching Track B pattern: $pattern on GPU $gpu_idx"

  CUDA_VISIBLE_DEVICES=$gpu_idx "$PYTHON_BIN" scripts/run_lc0.py \
    --pattern "$pattern" \
    --games "$TRACK_B_GAMES" \
    --nodes "$TRACK_B_NODES" \
    --lc0-path "$LC0_PATH" \
    --weights-path "$LC0_WEIGHTS" \
    --backend "$LC0_BACKEND" \
    --output "runs/results/track_b_${pattern}.jsonl" &

  gpu_idx=$(( (gpu_idx + 1) % 4 ))

  if [ $gpu_idx -eq 0 ]; then
    wait
  fi
done
wait

echo "========================================="
echo " Starting Elo Calibration "
echo "========================================="
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/evaluate_elo.py \
  --engine lc0 \
  --lc0-path "$LC0_PATH" \
  --weights "$LC0_WEIGHTS" \
  --backend "$LC0_BACKEND" \
  --sims "$ELO_SIMS" \
  --games "$ELO_GAMES" > runs/results/elo_calibration.txt

echo "Track B and Elo calibration finished!"
