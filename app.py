from flask import Flask, request, jsonify, send_from_directory
from engine import ConnectFour, is_legal, make_move, is_in_terminal_state
from model import DecisionModel, get_next_model_move
from minimax import minimax_move
from utils import SavedGame
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
                ai_move, _ = get_next_model_move(model, game)
            game = make_move(game, 2, ai_move)  # AI is player 2
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    status = is_in_terminal_state(game)
    return jsonify({"board": game.state.tolist(), "status": status})


@app.route("/list_saved_games", methods=["GET"])
def list_saved_games():
    saved_games = []
    for filename in os.listdir("saved_games"):
        if filename.endswith(".json"):
            filepath = os.path.join("saved_games", filename)
            game = SavedGame.load_from_file(filepath)
            saved_games.append(game.to_dict())
    return jsonify({"saved_games": saved_games})


@app.route("/load_game/<game_id>", methods=["GET"])
def load_game(game_id):
    for filename in os.listdir("saved_games"):
        if filename.endswith(f"{game_id}.json"):
            filepath = os.path.join("saved_games", filename)
            game = SavedGame.load_from_file(filepath)
            return jsonify(game.to_dict())
    return jsonify({"error": "Game file not found."}), 404


if __name__ == "__main__":
    os.makedirs("saved_games", exist_ok=True)
    # app.run(host="0.0.0.0", port=5001, debug=True)

    # only respond to requests from localhost
    app.run(port=5001, debug=True)
