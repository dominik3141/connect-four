from engine import ConnectFour
from typing import Tuple
import copy
from engine import is_in_terminal_state, make_move, is_legal
from model import DecisionModel, get_next_model_move
from torch import Tensor
import torch
import wandb


def minimax_move(board: ConnectFour, depth: int = 3) -> int:
    """
    Determine the best legal move for the AI player using the minimax algorithm with alpha-beta pruning.
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
                    if eval > max_eval or best_move == -1:
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
                    if eval < min_eval or best_move == -1:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return best_move, min_eval

    best_move, _ = minimax(board, depth, float("-inf"), float("inf"), True)

    # Ensure the best move is legal
    if not is_legal(board, best_move):
        # If the best move is not legal, choose the first legal move
        for move in range(7):
            if is_legal(board, move):
                return move

    # final check if still not legal
    if not is_legal(board, best_move):
        raise ValueError("No legal moves available")

    return best_move


def play_against_minimax(
    model: DecisionModel, temperature: float = 1.0, epsilon: float = 0, depth: int = 3
) -> Tuple[Tensor, int]:
    """
    Play a game where the model plays against the minimax opponent.
    Returns the move probabilities for the model and the game outcome.
    """
    board = ConnectFour()
    ai_move_probs = []
    current_player = 1

    while True:
        if current_player == 1:
            move = minimax_move(board, depth=depth)
        elif current_player == 2:
            move, prob = get_next_model_move(
                model, board, temperature=temperature, epsilon=epsilon
            )
            ai_move_probs.append(prob)

        # make the move
        board = make_move(board, current_player, move)

        # check if the game is in a terminal state
        if is_in_terminal_state(board) != 0:
            break

        current_player = 3 - current_player  # Switch player

    status = is_in_terminal_state(board)

    # Reverse the move probabilities for correct discounting
    ai_move_probs.reverse()
    ai_move_probs_tensor = torch.stack(ai_move_probs)

    return ai_move_probs_tensor, status


def train_against_minimax(
    model: DecisionModel,
    iterations: int = 100,
    learning_rate: float = 0.01,
    run=None,
    eval_interval: int = 100,
    eval_games: int = 100,
    temperature: float = 1.0,
    epsilon: float = 0,
    depth: int = 3,
) -> DecisionModel:
    """
    Train the model against the minimax opponent.
    """
    from ai import loss_fn
    from evaluations import evaluate_model, log_evaluation_results

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # a new game is played every iteration
    for i in range(iterations):
        optimizer.zero_grad()

        # play a game against minimax
        ai_move_probs, status = play_against_minimax(
            model, temperature=temperature, epsilon=epsilon, depth=depth
        )

        # calculate the loss
        loss = loss_fn(ai_move_probs, status, player=2)  # always train as player 2

        # log the loss to wandb
        wandb.log({"loss": loss})

        # backpropagate the loss
        loss.backward()
        optimizer.step()

        # evaluate the model every eval_interval iterations
        if i % eval_interval == 0:
            eval_results = evaluate_model(model, num_games=eval_games)
            log_evaluation_results(run, eval_results)

    return model
