from engine import ConnectFour
import engine
from typing import Callable, Dict
from model import DecisionModel, get_next_model_move
from minimax import minimax_move
from functools import partial
import wandb


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
    """
    Conducts a comprehensive evaluation of the given model against multiple opponents.
    The model plays a specified number of games against each opponent, alternating between
    playing as player 1 and player 2. The function returns a dictionary containing the
    results (wins, losses, draws) for each opponent.
    """
    opponents = {
        "random": engine.random_move,
        "minimax": partial(minimax_move, depth=depth_for_minimax),
    }

    results = {opponent: {"wins": 0, "losses": 0, "draws": 0} for opponent in opponents}

    # play num_games against each opponent
    for opponent_name, opponent_fn in opponents.items():
        for game_num in range(num_games):
            # Model plays as player 2
            outcome = play_against_opponent(model, opponent_fn, model_player=2)
            if outcome == 2:
                results[opponent_name]["wins"] += 1
            elif outcome == 1:
                results[opponent_name]["losses"] += 1
            else:
                results[opponent_name]["draws"] += 1

            # Model plays as player 1
            outcome = play_against_opponent(model, opponent_fn, model_player=1)
            if outcome == 1:
                results[opponent_name]["wins"] += 1
            elif outcome == 2:
                results[opponent_name]["losses"] += 1
            else:
                results[opponent_name]["draws"] += 1

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
        wandb.log({f"{opponent}_win_rate": win_rate})
