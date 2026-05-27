# HandiChess: Concentration vs. Dispersion in Material Handicap Chess

**CS377 Reinforcement Learning · Team 1**

## Overview

This project investigates whether concentrating a 9-point material handicap
(removing the queen) or dispersing it across multiple pieces yields different
effective handicap strengths under near-optimal play. We test the **additivity**
of the standard piece-value system using a setup-conditioned AlphaZero and
lc0 as reference.

## Key Research Questions

1. **Main RQ**: Is the effective handicap the same whether 9 points are removed
   as one piece (queen) or spread across multiple pieces?
2. **Sub-RQ 1**: Does the handicapped side's win rate change monotonically
   with the degree of dispersion?
3. **Sub-RQ 2**: What are the effective piece values under handicap play?
4. **Sub-RQ 3**: How does the first-move advantage interact with material
   handicap (black vs. white asymmetry)?

## Project Structure

```
handichess/
  config/            # Removal patterns, hyperparameters
  common/
    handicap.py      # Pattern → FEN generator
    gamelog.py        # Game log schema / IO
  track_b/
    lc0_runner.py    # UCI-based lc0 pipeline
  track_a/
    game/            # Game ABC, TicTacToe, Chess
    encoding.py      # Board planes + 8×8×73 actions
    net.py           # ResNet (policy + value)
    mcts.py          # PUCT MCTS
    selfplay.py      # Self-play with handicap init
    trainer.py       # Training loop
    arena.py         # Evaluation matches
    baseline.py      # Weak baselines
  analysis/
    winrate.py       # Dirichlet-multinomial win rates
    piece_values.py  # Logistic regression for piece values
    color_asym.py    # Black/white asymmetry analysis
tests/               # Unit and integration tests
scripts/             # CLI entry points
```

## Setup

```bash
# Clone and install
git clone <repo-url>
cd CS377_Team1_Concentration-vs.-Dispersion-in-Free-Setup-Chess-via-AlphaZero
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Quick Start

```bash
# 1. Generate handicap positions
python scripts/gen_positions.py

# 2. Run lc0 games (Track B)
python scripts/run_lc0.py --pattern queen --games 100 --nodes 800

# 3. Train AlphaZero (Track A)
python scripts/train.py --game tictactoe   # verify core first
python scripts/train.py --game chess        # then chess

# 4. Evaluate
python scripts/run_arena.py

# 5. Analyze results
python scripts/analyze.py
```

## Removal Patterns (9 points each)

| Pattern | Score | Pieces Removed | Concentration |
|---------|-------|---------------|--------------|
| Queen | 9 | 1 | Highest |
| Rook + Bishop + Pawn | 5+3+1 | 3 | High |
| Rook + Knight + Pawn | 5+3+1 | 3 | High |
| Bishop + Bishop + Knight | 3+3+3 | 3 | High |
| Rook + 4 Pawns | 5+4 | 5 | Medium |
| Bishop + Knight + 3 Pawns | 3+3+3 | 5 | Medium |
| Bishop + 6 Pawns | 3+6 | 7 | Low |
| Knight + 6 Pawns | 3+6 | 7 | Low |

## License

MIT
