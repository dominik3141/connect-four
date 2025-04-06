# Connect Four AI - Value-Based Reinforcement Learning

This project implements a Connect Four AI using a value-based Reinforcement Learning approach, trained exclusively through self-play. It includes a web interface for playing against the trained model or a Minimax opponent, along with tools for training and evaluation.

## Overview

The goal is to train a neural network (a "Value Model") to evaluate Connect Four board positions. This model predicts the expected outcome of the game (win, loss, or draw) from the perspective of the current player. The AI learns by playing against copies of itself, refining its value predictions based on the game outcomes.

## Core Components

1.  **Value Model (`src/model.py`)**: A multi-layer perceptron (MLP) that takes the board state (represented as separate planes for each player) and whose turn it is, outputting a value estimate in the range [-1, 1].
2.  **Self-Play Training (`src/self_play.py`, `src/main.py`)**: An "online" value model is trained against a periodically updated "frozen" version of itself. Training utilizes Temporal Difference (TD) learning, where the model updates its predictions based on the rewards received and its own estimates of future states. Exploration is managed using temperature-scaled softmax and epsilon-greedy strategies during move selection based on predicted next-state values.
3.  **Minimax Opponent (`src/minimax.py`, `libs/minimax.c`)**: A classic Minimax algorithm implemented in C for performance and interfaced via Python's `ctypes`. This serves as a strong baseline opponent for evaluating the trained value model.
4.  **Web Interface (`src/app.py`, `src/index.html`)**: A Flask-based web application allowing users to:
    - Play against the trained Value Model AI.
    - Play against the Minimax algorithm with adjustable depth.
    - Load and replay previously saved games.
5.  **Evaluation (`src/self_play.py`, `evaluate.py`)**: The training process includes regular evaluations:
    - Online model vs. Frozen model win rate.
    - Online model vs. a simple "Stacker" heuristic.
    - Online model vs. Minimax at various depths.
    - A separate script (`evaluate.py`) allows head-to-head evaluation of any two saved model checkpoints.
6.  **Logging**: Training progress, evaluation results, model parameters, and sample game visualizations are logged using Weights & Biases (`wandb`).

## Training Process

- The `src/main.py` script orchestrates the training loop.
- In each iteration, batches of games are played between the online and frozen value models.
- The online model's weights are updated based on the calculated TD errors (using the Mean Squared Error loss between its predictions and the TD targets).
- Periodically (`target_update_freq`), the online model is evaluated against the frozen model and other baselines (Stacker, Minimax).
- If the online model significantly outperforms the frozen model (win rate > `win_rate_threshold`), its weights are copied to the frozen model. Model checkpoints (both online and frozen) are saved locally and logged as artifacts to `wandb`.

## Getting Started

### Dependencies

- Python 3.x
- PyTorch
- NumPy
- Flask
- Weights & Biases (`wandb`)
- A C compiler (like GCC) to build the Minimax library.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install Python packages:**
    ```bash
    pip install torch numpy flask wandb
    ```
3.  **Compile the Minimax C library:**
    ```bash
    cd libs
    gcc -shared -o libconnect4.so -fPIC minimax.c
    cd ..
    ```
    _(Ensure `libs/libconnect4.so` exists after compilation)_

### Running

1.  **Training:**

    ```bash
    python -m src.main
    ```

    - Configure hyperparameters within `src/main.py`.
    - Set `use_wandb=True` to enable Weights & Biases logging (requires login).
    - Models (`value_model.pth`, `online_value_model.pth`) will be saved in the root directory by default.

2.  **Playing via Web Interface:**

    ```bash
    python -m src.app
    ```

    - Ensure a trained model (`value_model.pth`) exists in the root directory.
    - Access the interface in your browser (default: `http://localhost:5001`).

3.  **Evaluating Saved Models:**
    ```bash
    python evaluate.py --model1_path path/to/model1.pth --model2_path path/to/model2.pth --num_games 100
    ```

## Future Work / Potential Improvements

- Implement AlphaZero version to compare parameter size necessary for perfect play.
