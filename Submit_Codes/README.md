# CS377 Team 1 — Concentration vs. Dispersion in Free-Setup Chess via AlphaZero

## Submission Code Structure

This directory contains the **core code** for the final project, organized into 4 Jupyter notebooks.

### Notebooks

| Notebook | Description |
|----------|-------------|
| `01_environment_and_handicap_setup.ipynb` | Experimental environment: 7 removal patterns, FEN generation, game log schema |
| `02_alphazero_track_a.ipynb` | Complete AlphaZero implementation: Game ABC, Encoding, ResNet, MCTS, Self-Play, Training, Arena |
| `03_lc0_track_b.ipynb` | lc0-based evaluation pipeline: UCI self-play, MultiPV sampling, CLI scripts |
| `04_analysis_and_figures.ipynb` | All analysis code: Dirichlet win rates, piece values, color asymmetry, paper figures |

### Config Files

| File | Description |
|------|-------------|
| `config/patterns.yaml` | 7 removal pattern definitions (9 material points each) |
| `config/default.yaml` | Default hyperparameters for training, MCTS, and evaluation |

### Dependencies

- Python 3.9+
- PyTorch
- python-chess
- numpy, scipy, pandas, matplotlib
- (Track B only) lc0 engine binary + weights

### Quick Start

```bash
# Train on TicTacToe (verification)
python scripts/train.py --game tictactoe --iterations 50

# Train on Chess (handicap positions)
python scripts/train.py --game chess --iterations 100 --games-per-iter 256

# Run lc0 evaluation (Track B)
python scripts/run_lc0.py --pattern rook_bishop_pawn --games 100 --nodes 800

# Generate analysis and figures
python scripts/full_analysis.py
python scripts/generate_figures.py
```
