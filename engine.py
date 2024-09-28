import numpy as np
from typing import TypeAlias, List
import random


# The state of a game of Connect Four is represented by a 6x7 matrix
# The entries can be:
#   0 (empty spot)
#   1 (player 1)
#   2 (player 2)
class ConnectFour:
    def __init__(self, state: np.ndarray = None):
        """
        Initialize the ConnectFour board.
        If no state is provided, initialize an empty board.
        """
        if state is None:
            self.state = np.zeros((6, 7), dtype=int)
        else:
            self.state = state

    def __repr__(self) -> str:
        return str(self.state)


Move: TypeAlias = int  # Column number (0-6)


def is_legal(board: ConnectFour, move: Move) -> bool:
    """
    Checks if a move is legal
    """
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
    while True:
        empty_columns: List[Move] = [
            col for col in range(7) if board.state[0, col] == 0
        ]

        if not empty_columns:
            # If this error ever occurs, then one forgot to check for a stalemate first
            raise ValueError("No legal moves available")

        move = random.choice(empty_columns)

        # make sure the move is legal, if not, try again
        if not is_legal(board, move):
            continue

        return move


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
