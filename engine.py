import numpy as np
from typing import TypeAlias, List
import random


# The state of a game of Connect Four is represented by a 6x7 matrix
# The entries can be:
#   0 (empty spot)
#   1 (player 1)
#   2 (player 2)
class ConnectFour:
    def __init__(self):
        self.state = np.zeros((6, 7), dtype=int)

    def __repr__(self) -> str:
        return str(self.state)


Move: TypeAlias = int  # Column number (0-6)


def is_legal(board: ConnectFour, move: Move) -> bool:
    # Check if the column is full
    return board.state[0, move] == 0


def make_move(board: ConnectFour, player: int, move: Move) -> ConnectFour:
    if not is_legal(board, move):
        raise ValueError("Illegal move")

    # Find the lowest empty row in the chosen column
    for row in range(5, -1, -1):
        if board.state[row, move] == 0:
            board.state[row, move] = player
            break

    return board


def random_move(board: ConnectFour) -> Move:
    """
    Returns a random, but legal, move
    """
    empty_columns: List[Move] = [col for col in range(7) if board.state[0, col] == 0]

    if not empty_columns:
        # If this error ever occurs, then one forgot to check for a stalemate first
        raise ValueError("No legal moves available")

    move = random.choice(empty_columns)

    # Just to make sure, we also check if the move is legal
    if not is_legal(board, move):
        raise ValueError("Tried to choose an illegal move")

    return move


def play_random_game() -> ConnectFour:
    # Create a new board
    board = ConnectFour()

    # Play
    while True:
        for player in [1, 2]:
            move = random_move(board)
            board = make_move(board, player, move)

            if is_in_terminal_state(board) != 0:
                return board


def is_in_terminal_state(board: ConnectFour) -> int:
    for player in [1, 2]:
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if np.all(board.state[row, col : col + 4] == player):
                    return player

        # Check vertical
        for row in range(3):
            for col in range(7):
                if np.all(board.state[row : row + 4, col] == player):
                    return player

        # Check diagonal (positive slope)
        for row in range(3):
            for col in range(4):
                if np.all(
                    np.array([board.state[row + i, col + i] for i in range(4)])
                    == player
                ):
                    return player

        # Check diagonal (negative slope)
        for row in range(3, 6):
            for col in range(4):
                if np.all(
                    np.array([board.state[row - i, col + i] for i in range(4)])
                    == player
                ):
                    return player

    # Check for stalemate
    if not np.any(board.state[0] == 0):
        return 3  # Stalemate

    return 0  # No terminal state


def _assess_game(end_state: ConnectFour):
    result = is_in_terminal_state(end_state)
    if result in [1, 2]:
        print(f"Player {result} has won the game")
        print(end_state)
    elif result == 3:
        print("Game ended in a stalemate")
        print(end_state)


def calc_statistics_for_random_games(n_games: int):
    """
    Calculates the number of wins for player 1, player 2, and stalemates
    for a given number of random games.
    """
    wins_player_1 = 0
    wins_player_2 = 0
    stalemates = 0

    for _ in range(n_games):
        end_state = play_random_game()
        result = is_in_terminal_state(end_state)

        if result == 1:
            wins_player_1 += 1
        elif result == 2:
            wins_player_2 += 1
        elif result == 3:
            stalemates += 1

    print(f"Wins for player 1: {wins_player_1}")
    print(f"Wins for player 2: {wins_player_2}")
    print(f"Stalemates: {stalemates}")
