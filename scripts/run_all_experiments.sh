#!/bin/bash
# ==========================================
# CS377 Final Experiment Runner (Track A & B)
# Optimized for 4-GPU Cluster (40-hour limit)
# ==========================================

# 7 Patterns
PATTERNS=(rook_bishop_pawn rook_knight_pawn bishop_bishop_knight rook_4pawns bishop_knight_3pawns bishop_6pawns knight_6pawns)

# GPU Assignment
gpu_idx=0

echo "========================================="
echo " Starting Track A (AlphaZero) Evaluation "
echo "========================================="
for pattern in "${PATTERNS[@]}"; do
  echo "Launching Track A pattern: $pattern on GPU $gpu_idx"
  
  # --num-simulations 400: 지능(정확도) 2배 증가!
  # --games 400: 표본 크기 2배 증가! (통계적 유의성 확보)
  CUDA_VISIBLE_DEVICES=$gpu_idx .venv/bin/python scripts/eval_track_a.py \
    --checkpoint runs/checkpoints/final.pt \
    --device cuda \
    --pattern $pattern \
    --games 400 \
    --num-simulations 400 \
    --output runs/results/track_a_results.jsonl &
    
  gpu_idx=$(( (gpu_idx + 1) % 4 ))
  
  # 4개의 GPU가 모두 차면, 백그라운드 작업이 끝날 때까지 대기
  if [ $gpu_idx -eq 0 ]; then
    wait
  fi
done
wait # 남은 작업 대기

echo "========================================="
echo " Starting Track B (LC0) Evaluation "
echo "========================================="
gpu_idx=0
for pattern in "${PATTERNS[@]}"; do
  echo "Launching Track B pattern: $pattern on GPU $gpu_idx"
  
  CUDA_VISIBLE_DEVICES=$gpu_idx .venv/bin/python scripts/run_lc0.py \
    --pattern $pattern \
    --games 400 \
    --nodes 800 \
    --lc0-path "lc0" \
    --weights-path "runs/checkpoints/lc0_weights.pb.gz" \
    --backend "cuda" \
    --output runs/results/track_b_results.jsonl &
    
  gpu_idx=$(( (gpu_idx + 1) % 4 ))
  
  if [ $gpu_idx -eq 0 ]; then
    wait
  fi
done
wait

echo "========================================="
echo " Starting Elo Calibration "
echo "========================================="
# GPU 하나만 사용해서 빠르게 진행
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_elo.py \
    --engine lc0 \
    --lc0-path "lc0" \
    --sims 400 \
    --games 20 > runs/results/elo_calibration.txt

echo "All 4-GPU parallel experiments finished!"
