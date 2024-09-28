from engine import ConnectFour
import engine
from typing import Callable, Dict, Tuple
from model import DecisionModel, get_next_move
import torch
from torch import Tensor


def minimax_move(board: ConnectFour, depth: int = 4) -> int:
    def evaluate(board: ConnectFour) -> float:
        status = engine.is_in_terminal_state(board)
        if status == 1:
            return -1000
        elif status == 2:
            return 1000
        elif status == 3:
            return 0

        score = 0
        for row in range(6):  # Iterate over rows
            for col in range(7):  # Iterate over columns
                if board.state[row][col] == 0:
                    score += count_potential_wins(
                        board, row, col, 2
                    ) - count_potential_wins(board, row, col, 1)
        return score

    def count_potential_wins(
        board: ConnectFour, row: int, col: int, player: int
    ) -> int:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        count = 0
        for dy, dx in directions:
            if can_win_direction(board, row, col, dy, dx, player):
                count += 1
        return count

    def can_win_direction(
        board: ConnectFour, row: int, col: int, dy: int, dx: int, player: int
    ) -> bool:
        for i in range(4):
            y, x = row + i * dy, col + i * dx
            if y < 0 or y >= 6 or x < 0 or x >= 7:
                return False
            if board.state[y][x] not in (0, player):
                return False
        return True

    def minimax(
        board: ConnectFour, depth: int, maximizing_player: bool
    ) -> Tuple[int, float]:
        if depth == 0 or engine.is_in_terminal_state(board) != 0:
            return -1, evaluate(board)

        if maximizing_player:
            max_eval = float("-inf")
            best_move = -1
            for move in range(7):
                if engine.is_legal(board, move):
                    new_board = engine.make_move(board, 2, move)
                    _, eval = minimax(new_board, depth - 1, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
            return best_move, max_eval
        else:
            min_eval = float("inf")
            best_move = -1
            for move in range(7):
                if engine.is_legal(board, move):
                    new_board = engine.make_move(board, 1, move)
                    _, eval = minimax(new_board, depth - 1, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
            return best_move, min_eval

    best_move, _ = minimax(board, depth, True)
    return best_move


def play_against_opponent(
    model: DecisionModel,
    opponent_move_fn: Callable[[ConnectFour], int],
    model_player: int = 2,
) -> int:
    board = ConnectFour()
    current_player = 1

    while True:
        if current_player == model_player:
            move, _ = get_next_move(model, board)
        else:
            move = opponent_move_fn(board)

        board = engine.make_move(board, current_player, move)

        if engine.is_in_terminal_state(board) != 0:
            break

        current_player = 3 - current_player  # Switch player

    return engine.is_in_terminal_state(board)


def evaluate_model_comprehensive(
    model: DecisionModel, num_games: int = 100
) -> Dict[str, Dict[str, int]]:
    opponents = {
        "random": engine.random_move,
        "minimax": minimax_move,
    }

    results = {opponent: {"wins": 0, "losses": 0, "draws": 0} for opponent in opponents}

    for opponent_name, opponent_fn in opponents.items():
        for _ in range(num_games):
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


def evaluate_model(model: DecisionModel, iterations: int = 100) -> float:
    wins = 0
    for _ in range(iterations):
        _, status = play_against_random_player(model)
        if status == 2:
            wins += 1
    return wins / iterations


def play_against_random_player(model: DecisionModel) -> Tuple[Tensor, int]:
    board = ConnectFour()
    ai_move_probs = []

    while True:
        move = engine.random_move(board)
        board = engine.make_move(board, 1, move)

        if engine.is_in_terminal_state(board) != 0:
            break

        move, prob = get_next_move(model, board)
        board = engine.make_move(board, 2, move)
        ai_move_probs.append(prob)

        if engine.is_in_terminal_state(board) != 0:
            break

    status = engine.is_in_terminal_state(board)

    ai_move_probs.reverse()
    ai_move_probs_tensor = torch.stack(ai_move_probs)

    return ai_move_probs_tensor, status
