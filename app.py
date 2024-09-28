from flask import Flask, request, jsonify, send_from_directory
from engine import ConnectFour, is_legal, make_move, is_in_terminal_state
from ai import (
    DecisionModel,
    get_next_move,
)
from minimax import minimax_move
import torch
import os
import numpy as np
from typing import List

app = Flask(__name__)

# Initialize the model
model = DecisionModel()
model_path = "model.pth"

# Load the trained model if available
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    # If no trained model, you might want to train it or handle accordingly
    print("No trained model found, training model...")
    pass


def deserialize_board(board_state: List[List[int]]) -> ConnectFour:
    return ConnectFour(np.array(board_state))


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/start", methods=["GET"])
def start_game():
    game = ConnectFour()
    return jsonify({"board": game.state.tolist()})


@app.route("/move", methods=["POST"])
def player_move():
    data = request.get_json()
    board_state: List[List[int]] = data.get("board")
    move: int = data.get("move")
    opponent: str = data.get("opponent", "ai")
    depth: int = data.get("depth", 1)

    if (
        not board_state
        or move is None
        or not isinstance(move, int)
        or not (0 <= move <= 6)
    ):
        return jsonify({"error": "Invalid input."}), 400

    game = deserialize_board(board_state)

    if not is_legal(game, move):
        return jsonify({"error": "Illegal move."}), 400

    try:
        game = make_move(game, 1, move)  # Player is 1
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    status = is_in_terminal_state(game)
    if status != 0:
        return jsonify({"board": game.state.tolist(), "status": status})

    # AI's turn
    if status == 0:
        try:
            if opponent == "minimax":
                ai_move = minimax_move(game, depth)
            else:
                ai_move, _ = get_next_move(model, game)
            game = make_move(game, 2, ai_move)  # AI is player 2
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    status = is_in_terminal_state(game)
    return jsonify({"board": game.state.tolist(), "status": status})


if __name__ == "__main__":
    app.run(debug=True)
