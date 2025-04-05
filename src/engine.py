import numpy as np
from typing import TypeAlias, List
import random


# The state of a game of Connect Four is represented by a 6x7 matrix
# The entries can be:
#   0 (empty spot)
#   1 (player 1)
#   2 (player 2)
class ConnectFour:
    def __init__(self, state: np.ndarray | None = None):
        """Initialize the ConnectFour board with an optional starting state"""
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
    """
    Creates a *new* board state representing the board after the player makes the specified move.
    Does not modify the original board.
    """
    if not is_legal(board, move):
        raise ValueError("Illegal move")

    # Create a copy of the current state to avoid modifying the original board
    new_state = board.state.copy()

    # Find the lowest empty row in the chosen column
    for row in range(5, -1, -1):
        if new_state[row, move] == 0:
            new_state[row, move] = player
            break
    # If the loop finishes without break, the column was full, but is_legal should prevent this.

    # Return a new ConnectFour object with the new state
    return ConnectFour(state=new_state)


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
    """Returns the winning player number (1,2), 3 for stalemate, or 0 if game continues"""
    state = board.state

    for player in [1, 2]:
        # Horizontal win detection
        mask = state == player
        horiz = mask[:, :-3] & mask[:, 1:-2] & mask[:, 2:-1] & mask[:, 3:]
        if horiz.any():
            return player

        # Vertical win detection
        vert = mask[:-3, :] & mask[1:-2, :] & mask[2:-1, :] & mask[3:, :]
        if vert.any():
            return player

        # Positive slope diagonal (bottom-right) win
        for row in range(3):
            for col in range(4):
                if all(state[row + i, col + i] == player for i in range(4)):
                    return player

        # Negative slope diagonal (top-right) win
        for row in range(
            3, 6
        ):  # rows 3,4,5 (where a negative slope diagonal can start)
            for col in range(4):
                if all(state[row - i, col + i] == player for i in range(4)):
                    return player

    return 3 if not np.any(state[0] == 0) else 0
