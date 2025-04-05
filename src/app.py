from flask import Flask, request, jsonify, send_from_directory
from .engine import ConnectFour, is_legal, make_move, is_in_terminal_state
from .model import ValueModel, get_next_value_based_move
from .minimax import minimax_move
from .utils import SavedGame
import torch
import os
import numpy as np
from typing import List
from datetime import datetime
import traceback
import logging

app = Flask(__name__)

# Initialize the logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the Value Model
value_model = ValueModel()
model_path = "value_model.pth"

# Load the trained value model if available
if os.path.exists(model_path):
    logger.info(f"Loading trained value model from {model_path}")
    # Load weights onto the correct device (CPU is usually fine for inference)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_model.load_state_dict(torch.load(model_path, map_location=device))
    value_model.to(device)  # Ensure model is on the correct device
    value_model.eval()  # Set model to evaluation mode
    logger.info("Value model loaded successfully.")
else:
    # Fail if no trained model is found
    logger.error(f"No trained value model found at {model_path}. Exiting.")
    raise Exception(f"No trained value model found at {model_path}")


def deserialize_board(board_state: List[List[int]]) -> ConnectFour:
    return ConnectFour(np.array(board_state, dtype=int))  # Ensure dtype is int


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
    move_col: int = data.get("move")  # Renamed for clarity
    opponent: str = data.get("opponent", "ai")  # 'ai' now means value model
    depth: int = data.get("depth", 1)  # Only used for minimax

    if board_state is None or move_col is None:
        logger.warning("Invalid input: Missing board state or move.")
        return jsonify({"error": "Invalid input: Missing board state or move."}), 400

    # Basic validation for move column index
    if not isinstance(move_col, int) or not (0 <= move_col <= 6):
        logger.warning(f"Invalid move column index: {move_col}")
        return jsonify(
            {"error": f"Invalid move column index: {move_col}. Must be 0-6."}
        ), 400

    try:
        game = deserialize_board(board_state)
    except Exception as e:
        logger.error(f"Error deserializing board: {e}")
        return jsonify({"error": "Failed to process board state."}), 400

    if not is_legal(game, move_col):
        logger.warning(f"Illegal move attempted: Column {move_col}")
        return jsonify(
            {"error": f"Illegal move: Column {move_col} is full or invalid."}
        ), 400

    # --- Player's Move (Player 1) ---
    try:
        logger.debug(f"Player 1 attempting move in column {move_col}")
        game = make_move(game, 1, move_col)
        logger.debug(f"Board state after Player 1 move:\n{game.state}")
    except ValueError as e:
        logger.error(f"Error making player move: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error during player move: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500

    status = is_in_terminal_state(game)
    logger.debug(f"Status after Player 1 move: {status}")
    if status != 0:
        # Game ended after player's move
        return jsonify({"board": game.state.tolist(), "status": status})

    # --- AI's Move (Player 2) ---
    ai_move: int = -1  # Initialize ai_move
    try:
        if opponent == "minimax":
            logger.debug(f"Minimax (Player 2) choosing move with depth {depth}...")
            ai_move = minimax_move(game, player=2, depth=depth)
            logger.debug(f"Minimax chose move: {ai_move}")
        elif opponent == "ai":  # Use the Value Model
            logger.debug("Value Model (Player 2) choosing move...")
            # Use the loaded value_model to get the next move
            # Use low temperature for near-greedy selection in the app
            ai_move, _, _, _ = get_next_value_based_move(
                value_model=value_model,
                board=game,
                current_player=2,
                temperature=0.001,  # Near-greedy
                epsilon=0.0,  # No exploration
            )
            logger.debug(f"Value Model chose move: {ai_move}")
        else:
            logger.error(f"Unknown opponent type: {opponent}")
            return jsonify({"error": f"Unknown opponent type: {opponent}"}), 400

        logger.debug(f"AI (Player 2) attempting move in column {ai_move}")
        game = make_move(game, 2, ai_move)  # AI is player 2
        logger.debug(f"Board state after AI move:\n{game.state}")

    except ValueError as e:
        # This could happen if minimax or value model returns an illegal move (shouldn't happen ideally)
        # Or if make_move raises an error unexpectedly
        logger.error(
            f"Error making AI move ({opponent}, chosen: {ai_move}): {e}", exc_info=True
        )
        # Return the board state *before* the failed AI move attempt
        # status is still 0 here, as the error occurred *during* AI's turn
        return jsonify(
            {
                "board": deserialize_board(board_state).state.tolist(),
                "status": 0,
                "error": f"AI failed to make a move: {str(e)}",
            }
        ), 500
    except Exception as e:
        logger.error(
            f"Unexpected error during AI move ({opponent}): {e}", exc_info=True
        )
        return jsonify({"error": "An unexpected error occurred during AI turn."}), 500

    # Check status *after* AI move
    status = is_in_terminal_state(game)
    logger.debug(f"Status after AI move: {status}")
    return jsonify({"board": game.state.tolist(), "status": status})


@app.route("/list_saved_games", methods=["GET"])
def list_saved_games():
    try:
        saved_games_dir = "saved_games"
        if not os.path.exists(saved_games_dir):
            return jsonify(
                {"saved_games": []}
            )  # Return empty list if directory doesn't exist

        saved_games = []
        for filename in os.listdir(saved_games_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(saved_games_dir, filename)
                try:
                    game = SavedGame.load_from_file(filepath)
                    saved_games.append(game)
                except Exception as e:
                    logger.error(f"Error loading game from {filepath}: {str(e)}")

        # Sort the games by timestamp, most recent first
        saved_games.sort(
            key=lambda x: datetime.fromisoformat(x.timestamp), reverse=True
        )

        # Take only the 100 most recent games
        recent_games = saved_games[:100]

        # Convert the games to dictionaries for JSON serialization
        recent_games_dict = [game.to_dict() for game in recent_games]

        return jsonify({"saved_games": recent_games_dict})
    except Exception as e:
        logger.error(f"Error in list_saved_games: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An internal server error occurred"}), 500


@app.route("/load_game/<game_id>", methods=["GET"])
def load_game(game_id):
    saved_games_dir = "saved_games"
    filename = f"{game_id}.json"  # Construct the expected filename
    filepath = os.path.join(saved_games_dir, filename)

    if os.path.exists(filepath):
        try:
            game = SavedGame.load_from_file(filepath)
            return jsonify(game.to_dict())
        except Exception as e:
            logger.error(f"Error loading specific game {filepath}: {e}", exc_info=True)
            return jsonify({"error": "Failed to load game file."}), 500
    else:
        logger.warning(f"Game file not found: {filepath}")
        return jsonify({"error": "Game file not found."}), 404


if __name__ == "__main__":
    os.makedirs("saved_games", exist_ok=True)
    # Use host="0.0.0.0" to allow external connections if needed, otherwise defaults to localhost
    app.run(host="0.0.0.0", port=5001, debug=True)
