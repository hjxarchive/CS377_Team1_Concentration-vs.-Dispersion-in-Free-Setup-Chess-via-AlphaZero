# CS377 Final Project: Concentration vs. Dispersion in Free-Setup Chess via AlphaZero
**Team 1 Submission Codes**

## Overview
This folder contains the core implementation of our final project. Due to the 5MB upload limit on KLMS, we couldn't submit our entire Git repository, which contains hundreds of megabytes of raw game logs, Lc0 engine binaries, and large neural network weights. 

To make our code easy to review, we have consolidated our entire pipeline into four heavily commented Jupyter Notebooks. These notebooks contain all our core logic—from environment generation and our from-scratch AlphaZero implementation to our evaluation pipelines and statistical analysis. 

We recommend reading through the notebooks sequentially to understand our methodology and implementation details.

## Code Structure

### 1. `01_environment_and_handicap_setup.ipynb`
This notebook defines the experimental setting used throughout our project.
- **Handicap FEN Generation**: Logic for creating the "NoQ vs Q" matchup positions. We define how the 9 points of material are removed for both sides and generate the valid initial board states.
- **Game Log Schema**: The data structures (`GameRecord` and `GameLog`) we designed to record and process thousands of self-play games efficiently.
- **Configuration**: Includes the YAML parsing logic for loading our handicap patterns from `config/patterns.yaml`.
*Runnable?* Yes, if you install `chess`, `yaml`, and `pandas`, this notebook will execute fully and visualize the starting boards.

### 2. `02_alphazero_track_a.ipynb`
This contains our from-scratch implementation of the AlphaZero algorithm (ScratchZero).
- **Architecture**: The PyTorch implementation of the dual-headed ResNet policy/value network, along with our board state encoding logic.
- **MCTS**: The Monte Carlo Tree Search algorithm, implementing Dirichlet noise and UCB-based action selection.
- **Training Pipeline**: The Self-Play data generation and Arena evaluation loops.
*Runnable?* The class definitions and PyTorch models can be executed and inspected, but running a full training loop would require a GPU and significant time.

### 3. `03_lc0_track_b.ipynb`
This notebook contains our wrapper for the Lc0 (Leela Chess Zero) engine, which we used to evaluate our handicap patterns against a world-class network.
- **UCI Communication**: Python logic to spawn and interact with the `lc0` binary subprocess.
- **MultiPV Sampling**: Scripts to extract Q-values and search node distributions directly from the engine's search tree.
*Runnable?* Executing this notebook requires the `lc0` binary and its corresponding neural network weights (`BT2-3650000.pb.gz`). Since we had to exclude these binaries to meet the 5MB limit, the notebook cannot be run out of the box.

### 4. `04_analysis_and_figures.ipynb`
This contains the data processing and visualization code used to generate the figures in our final report.
- **Win Rate Analysis**: Uses a Dirichlet-multinomial posterior model to estimate Q-Scores and 95% credible intervals.
- **Visualization**: Generates the exact charts used in our paper, including the Minor Pieces Lost bar chart (Figure 2) and the Major vs. Minor Heatmap (Figure 3).
*Runnable?* This notebook parses the raw `.jsonl` game logs that we generated during the project. Because those data files also exceed the 5MB limit and were excluded, the code is provided for review purposes to show exactly how we derived our results and figures.

## How to Evaluate
Because the heavy dependencies (PyTorch weights, Lc0 binaries, and large JSONL data files) had to be excluded, these notebooks are primarily intended to be read as code documentation. They demonstrate the completeness of our implementation and the exact logic we used to reach our conclusions.
