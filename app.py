from flask import Flask, request, jsonify, send_from_directory
from engine import ConnectFour, is_legal, make_move, is_in_terminal_state
from ai import (
    DecisionModel,
    get_next_move,
)
from minimax import minimax_move
import torch
import os

app = Flask(__name__)

# Initialize the game state and model
game = ConnectFour()
model = DecisionModel()
model_path = "model_minimax.pth"

# Load the trained model if available
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    # If no trained model, you might want to train it or handle accordingly
    print("No trained model found, training model...")
    pass


def serialize_board(board: ConnectFour):
    return board.state.tolist()


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/start", methods=["GET"])
def start_game():
    global game
    game = ConnectFour()
    return jsonify({"board": serialize_board(game)})


@app.route("/move", methods=["POST"])
def player_move():
    global game
    data = request.get_json()
    move = data.get("move")

    if move is None or not isinstance(move, int) or not (0 <= move <= 6):
        return jsonify({"error": "Invalid move."}), 400

    if not is_legal(game, move):
        return jsonify({"error": "Illegal move."}), 400

    try:
        game = make_move(game, 1, move)  # Player is 1
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    status = is_in_terminal_state(game)
    if status != 0:
        return jsonify({"board": serialize_board(game), "status": status})

    return jsonify({"board": serialize_board(game), "status": status})


@app.route("/ai_move", methods=["GET"])
def ai_move():
    global game, model
    if is_in_terminal_state(game) != 0:
        return jsonify(
            {"board": serialize_board(game), "status": is_in_terminal_state(game)}
        )

    try:
        move, prob = get_next_move(model, game)
        game = make_move(game, 2, move)  # AI is player 2
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    status = is_in_terminal_state(game)
    return jsonify({"board": serialize_board(game), "status": status})


@app.route("/minimax_move", methods=["POST"])
def get_minimax_move():
    global game
    data = request.get_json()
    depth = data.get("depth", 1)  # Default depth is 1 if not provided

    if is_in_terminal_state(game) != 0:
        return jsonify(
            {"board": serialize_board(game), "status": is_in_terminal_state(game)}
        )

    try:
        move = minimax_move(game, depth)
        game = make_move(game, 2, move)  # AI is player 2
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    status = is_in_terminal_state(game)
    return jsonify({"board": serialize_board(game), "status": status})


@app.route("/reset", methods=["GET"])
def reset_game():
    global game
    game = ConnectFour()
    return jsonify({"board": serialize_board(game)})


if __name__ == "__main__":
    app.run(debug=True)
