#!/usr/bin/env bash
set -euo pipefail

parallel=2
running=0

engine=${LC0_ENGINE:-external/lc0/build/release/lc0}
weights=${LC0_WEIGHTS:-external/lc0/build/release/t1-256x10-distilled-swa-2432500.pb.gz}
backend=${LC0_BACKEND:-cuda-auto}
output=${OUTPUT:-runs/floor_pilot.jsonl}
games=${GAMES:-50}
nodes=${NODES:-100}
conda_env=${CONDA_ENV:-handichess}

patterns=(
  bishop_6pawns
  knight_6pawns
)

devices=(
  4
  5
)

if ! command -v "${engine}" >/dev/null 2>&1 && [[ ! -x "${engine}" ]]; then
  echo "LC0 engine not found: ${engine}" >&2
  exit 1
fi

run_job() {
  local pattern="$1"
  local device="$2"

  echo "[start] pilot pattern=${pattern} cuda_visible_devices=${device}"
  CUDA_VISIBLE_DEVICES="${device}" conda run -n "${conda_env}" python scripts/run_lc0.py \
    --engine "${engine}" \
    --weights "${weights}" \
    --backend "${backend}" \
    --nodes "${nodes}" \
    --games "${games}" \
    --pattern "${pattern}" \
    --output "${output}"
  echo "[done] pilot pattern=${pattern} cuda_visible_devices=${device}"
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
