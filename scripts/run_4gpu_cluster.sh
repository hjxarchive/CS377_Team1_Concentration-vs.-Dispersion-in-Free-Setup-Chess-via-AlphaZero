#!/bin/bash
# ==============================================================================
# CS377 HandiChess: 4x RTX 4080 Cluster Run Script
# ==============================================================================
# This script supports two ways of execution:
#   1. Slurm Job Scheduler (Recommended for shared clusters) -> sbatch scripts/run_4gpu_cluster.sh
#   2. Direct Tmux/Nohup execution (For dedicated VM instances) -> bash scripts/run_4gpu_cluster.sh
# ==============================================================================

# SBATCH HEADERS (Active only when submitted via sbatch)
#SBATCH --job-name=handichess-4gpu
#SBATCH --output=runs/logs/train_%j.log
#SBATCH --error=runs/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                  # Request 4 GPUs
#SBATCH --cpus-per-task=16            # Request 16 CPU cores for dataloading/MCTS parallelism
#SBATCH --mem=64G                     # Request 64GB System RAM
#SBATCH --time=72:00:00               # 3 days time limit (adjust as needed)

# Exit on error
set -e

# Ensure logging and checkpoint directories exist
mkdir -p runs/logs
mkdir -p runs/checkpoints

# 1. Environment Setup
echo "=== [1/3] Setting up environment ==="
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment (.venv)"
else
    echo "Warning: .venv directory not found. Using system python."
fi

export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"

# 2. Multi-GPU Visibility Configuration
echo "=== [2/3] Configuring Multi-GPU ==="
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# Auto-detect number of visible GPUs in PyTorch
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Visible GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 3. Launching Experiment
echo "=== [3/3] Launching AlphaZero Train ==="
echo "Running Track A main training across all 4 GPUs..."

python scripts/train.py \
    --game chess \
    --device cuda \
    --iterations 100 \
    --games-per-iter 256 \
    --simulations 800 \
    --max-moves 180 \
    --checkpoint-dir runs/checkpoints

echo "=== Training process finished ==="
