from src.engine import ConnectFour
import src.engine as engine
from typing import Callable, Dict
from model import DecisionModel, get_next_model_move
from minimax import minimax_move
from functools import partial
from utils import safe_log_to_wandb
import random


def play_against_opponent(
    model: DecisionModel,
    opponent_move_fn: Callable[[ConnectFour], int],  # opponent move function
    model_player: int = 2,
) -> int:
    """
    Simulates a game of Connect Four between the given model and an opponent.
    The model and opponent take turns making moves until the game reaches a terminal state.
    The function returns the outcome of the game.
    """
    board = ConnectFour()
    current_player = 1

    while True:
        if engine.is_in_terminal_state(board) != 0:
            break

        if current_player == model_player:
            move, _ = get_next_model_move(model, board)
        else:
            move = opponent_move_fn(board)

        board = engine.make_move(board, current_player, move)

        current_player = 3 - current_player  # Switch player

    return engine.is_in_terminal_state(board)


def evaluate_model(
    model: DecisionModel, num_games: int = 100, depth_for_minimax: int = 2
) -> Dict[str, Dict[str, int]]:
    opponents = {
        "random": lambda board, player: engine.random_move(board),
    }

    results = {opponent: {"wins": 0, "losses": 0, "draws": 0} for opponent in opponents}
    results["minimax"] = {"wins": 0, "losses": 0, "draws": 0}

    for _ in range(num_games):
        # Randomly decide who starts
        model_player = random.choice([1, 2])
        opponent_player = 3 - model_player

        # Create minimax opponent dynamically
        minimax_opponent = partial(
            minimax_move, player=opponent_player, depth=depth_for_minimax
        )

        for opponent_name, opponent_fn in opponents.items():
            outcome = play_against_opponent(
                model,
                lambda board: opponent_fn(board, opponent_player),
                model_player=model_player,
            )

            if outcome == model_player:
                results[opponent_name]["wins"] += 1
            elif outcome == opponent_player:
                results[opponent_name]["losses"] += 1
            else:
                results[opponent_name]["draws"] += 1

        # Evaluate against minimax separately
        outcome = play_against_opponent(
            model, lambda board: minimax_opponent(board), model_player=model_player
        )

        if outcome == model_player:
            results["minimax"]["wins"] += 1
        elif outcome == opponent_player:
            results["minimax"]["losses"] += 1
        else:
            results["minimax"]["draws"] += 1

    return results


def log_evaluation_results(eval_results: Dict[str, Dict[str, int]]) -> None:
    """
    Logs the win rate for each opponent to the provided run object.

    eval_results is a dictionary with the following structure:
    {
        "opponent_name": {
            "wins": int,
            "losses": int,
            "draws": int
        }
    }
    """
    for opponent, results in eval_results.items():
        total_games = sum(results.values())
        win_rate = results["wins"] / total_games if total_games > 0 else 0
        safe_log_to_wandb({f"{opponent}_win_rate": win_rate})
