#!/usr/bin/env bash
set -euo pipefail

parallel=4
running=0

engine=${LC0_ENGINE:-lc0}
weights=${LC0_WEIGHTS:-}
backend=${LC0_BACKEND:-cuda-auto}
output=${OUTPUT:-runs/lc0_main.jsonl}
games=${GAMES:-200}
nodes=${NODES:-800}
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

devices=(
  4
  5
  6
  7
)

if ! command -v "${engine}" >/dev/null 2>&1 && [[ ! -x "${engine}" ]]; then
  echo "LC0 engine not found: ${engine}" >&2
  echo "Set LC0_ENGINE=/path/to/lc0 and rerun this script." >&2
  exit 1
fi

run_job() {
  local pattern="$1"
  local device="$2"

  echo "[start] pattern=${pattern} cuda_visible_devices=${device} engine=${engine} weights=${weights:-<autodiscover>} backend=${backend} nodes=${nodes}"
  CUDA_VISIBLE_DEVICES="${device}" conda run -n "${conda_env}" python scripts/run_lc0.py \
    --engine "${engine}" \
    --weights "${weights}" \
    --backend "${backend}" \
    --nodes "${nodes}" \
    --games "${games}" \
    --pattern "${pattern}" \
    --output "${output}"
  echo "[done] pattern=${pattern} cuda_visible_devices=${device}"
}

idx=0
for pattern in "${patterns[@]}"; do
  device="${devices[$((idx % ${#devices[@]}))]}"
  idx=$((idx + 1))
  run_job "${pattern}" "${device}" &
  running=$((running + 1))

  if (( running >= parallel )); then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "[all done] wrote ${output}"
