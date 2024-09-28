from engine import ConnectFour
from typing import Tuple
import copy
from engine import is_in_terminal_state, make_move, is_legal


def minimax_move(board: ConnectFour, depth: int = 1) -> int:
    """
    Determine the best move for the AI player using the minimax algorithm with alpha-beta pruning.
    """

    def evaluate(board: ConnectFour) -> float:
        """
        Evaluate the current board state. Positive scores favor the AI player,
        negative scores favor the human player.
        """
        status = is_in_terminal_state(board)
        if status == 1:
            return -1000
        elif status == 2:
            return 1000
        elif status == 3:
            return 0

        score = 0
        for col in range(7):  # Iterate over columns
            row = next((r for r in range(5, -1, -1) if board.state[r][col] == 0), -1)
            if row != -1:
                score += count_potential_wins(
                    board, row, col, 2
                ) - count_potential_wins(board, row, col, 1)
        return score

    def count_potential_wins(
        board: ConnectFour, row: int, col: int, player: int
    ) -> int:
        """
        Count the number of potential winning lines for a player at a given position.
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        count = 0
        for dy, dx in directions:
            if can_win_direction(board, row, col, dy, dx, player):
                count += 1
        return count

    def can_win_direction(
        board: ConnectFour, row: int, col: int, dy: int, dx: int, player: int
    ) -> bool:
        """
        Check if a player can potentially win in a given direction from a specific position.
        """
        for i in range(4):
            y, x = row + i * dy, col + i * dx
            if y < 0 or y >= 6 or x < 0 or x >= 7:
                return False
            if board.state[y][x] not in (0, player):
                return False
        return True

    def minimax(
        board: ConnectFour,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
    ) -> Tuple[int, float]:  # returns best move and evaluation score
        """
        Implement the minimax algorithm with alpha-beta pruning.
        Returns the best move and its evaluation score.
        """
        if depth == 0 or is_in_terminal_state(board) != 0:
            return -1, evaluate(board)

        best_move = -1
        if maximizing_player:
            max_eval = float("-inf")
            for move in range(7):
                if is_legal(board, move):
                    new_board = copy.deepcopy(board)
                    make_move(new_board, 2, move)
                    _, eval = minimax(new_board, depth - 1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return best_move, max_eval
        else:
            min_eval = float("inf")
            for move in range(7):
                if is_legal(board, move):
                    new_board = copy.deepcopy(board)
                    make_move(new_board, 1, move)
                    _, eval = minimax(new_board, depth - 1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return best_move, min_eval

    best_move, _ = minimax(board, depth, float("-inf"), float("inf"), True)
    return best_move
