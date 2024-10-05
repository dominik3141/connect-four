from engine import ConnectFour
from typing import Tuple, List, Optional
from engine import is_in_terminal_state, make_move, is_legal
from model import DecisionModel, get_next_model_move
from torch import Tensor
import torch
from utils import safe_log_to_wandb, SavedGame
import random
import numpy as np


def minimax_move(board: ConnectFour, player: int, depth: int = 1) -> int:
    """
    Determine the best legal move for the given player using the minimax algorithm with alpha-beta pruning.
    """

    def evaluate(board: ConnectFour, player: int) -> float:
        """
        Evaluate the current board state. Positive scores favor the current player,
        negative scores favor the opponent.
        """
        WINDOW_LENGTH = 4
        score = 0

        # Center column preference
        center_array = [board.state[r][3] for r in range(6)]
        center_count = center_array.count(player)
        score += center_count * 3

        # Score Horizontal
        for r in range(6):
            row_array = board.state[r]
            for c in range(7 - 3):
                window = row_array[c : c + WINDOW_LENGTH]
                score += evaluate_window(window, player)

        # Score Vertical
        for c in range(7):
            col_array = [board.state[r][c] for r in range(6)]
            for r in range(6 - 3):
                window = col_array[r : r + WINDOW_LENGTH]
                score += evaluate_window(window, player)

        # Score positive sloped diagonals
        for r in range(6 - 3):
            for c in range(7 - 3):
                window = [board.state[r + i][c + i] for i in range(WINDOW_LENGTH)]
                score += evaluate_window(window, player)

        # Score negative sloped diagonals
        for r in range(3, 6):
            for c in range(7 - 3):
                window = [board.state[r - i][c + i] for i in range(WINDOW_LENGTH)]
                score += evaluate_window(window, player)

        return score

    def evaluate_window(window: np.ndarray, player: int) -> int:
        """
        Evaluate the score of a window (4 consecutive cells) for the given player.
        """
        score = 0
        opp_player = 3 - player

        if np.count_nonzero(window == player) == 4:
            # 4 in a row is a win
            score += 100
        elif (
            # 3 in a row is a good position
            np.count_nonzero(window == player) == 3
            and np.count_nonzero(window == 0) == 1
        ):
            score += 5
        elif (
            # 2 in a row
            np.count_nonzero(window == player) == 2
            and np.count_nonzero(window == 0) == 2
        ):
            score += 2

        if (
            # 3 in a row for opponent
            np.count_nonzero(window == opp_player) == 3
            and np.count_nonzero(window == 0) == 1
        ):
            score -= 4

        return score

    def is_terminal_node(board: ConnectFour) -> bool:
        """
        Check if the current board is a terminal node (win, loss, or draw).
        """
        return is_in_terminal_state(board) != 0 or len(get_valid_locations(board)) == 0

    def get_valid_locations(board: ConnectFour) -> List[int]:
        """
        Get a list of valid columns where a move can be made.
        """
        return [col for col in range(7) if is_legal(board, col)]

    def get_next_open_row(board: ConnectFour, col: int) -> Optional[int]:
        """
        Get the next open row in the specified column.
        """
        for r in range(5, -1, -1):
            if board.state[r][col] == 0:
                return r
        return None

    def make_move(board: ConnectFour, player: int, col: int):
        """
        Place the player's piece in the specified column.
        """
        row = get_next_open_row(board, col)
        if row is not None:
            board.state[row][col] = player

    def undo_move(board: ConnectFour, col: int):
        """
        Remove the top piece from the specified column.
        """
        for r in range(6):
            if board.state[r][col] != 0:
                board.state[r][col] = 0
                break

    def minimax(
        board: ConnectFour,
        depth: int,
        alpha: float,
        beta: float,
        current_player: int,
        original_player: int,
    ) -> Tuple[Optional[int], float]:
        """
        Minimax algorithm with alpha-beta pruning.
        Returns the best column and its evaluation score.
        """
        valid_locations = get_valid_locations(board)
        is_terminal = is_terminal_node(board)

        if depth == 0 or is_terminal:
            if is_terminal:
                status = is_in_terminal_state(board)
                if status == original_player:
                    return (None, float("inf"))
                elif status == 3 - original_player:
                    return (None, float("-inf"))
                else:  # Game is a draw
                    return (None, 0)
            else:
                return (None, evaluate(board, original_player))

        if current_player == original_player:
            # Maximizing player
            value = float("-inf")
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                make_move(board, current_player, col)
                new_score = minimax(
                    board, depth - 1, alpha, beta, 3 - current_player, original_player
                )[1]
                undo_move(board, col)
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_col, value
        else:
            # Minimizing player
            value = float("inf")
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                make_move(board, current_player, col)
                new_score = minimax(
                    board, depth - 1, alpha, beta, 3 - current_player, original_player
                )[1]
                undo_move(board, col)
                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_col, value

    best_move, _ = minimax(board, depth, float("-inf"), float("inf"), player, player)

    # Ensure the best move is legal
    if best_move is None or not is_legal(board, best_move):
        valid_locations = get_valid_locations(board)
        if valid_locations:
            best_move = random.choice(valid_locations)
        else:
            raise ValueError("No valid moves available")

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
            move = minimax_move(board, player=1, depth=depth)
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

    # print the board if the ai won
    if status == 2:
        print("AI won!")
        print(board)

    # Reverse the move probabilities for correct discounting
    ai_move_probs.reverse()
    ai_move_probs_tensor = torch.stack(ai_move_probs)

    return ai_move_probs_tensor, status


def play_batch_against_minimax(
    model: DecisionModel,
    batch_size: int,
    temperature: float = 1.0,
    epsilon: float = 0,
    depth: int = 3,
    run=None,
) -> Tuple[List[Tensor], List[int], float]:
    """
    Play a batch of games where the model plays against the minimax opponent.
    Returns a list of move probabilities for the model and a list of game outcomes.
    """
    batch_probs = []
    batch_outcomes = []

    for _ in range(batch_size):
        probs, outcome = play_against_minimax(model, temperature, epsilon, depth)
        batch_probs.append(probs)
        batch_outcomes.append(outcome)

    return batch_probs, batch_outcomes


def train_against_minimax(
    model: DecisionModel,
    iterations: int = 1000,
    learning_rate: float = 0.01,
    eval_interval: int = 100,
    eval_games: int = 100,
    temperature: float = 1.0,
    epsilon: float = 0,
    gamma: float = 0.9,
    depth: int = 3,
    batch_size: int = 128,
) -> DecisionModel:
    """
    Train the model against the minimax opponent.
    """
    from ai import loss_fn
    from evaluations import evaluate_model, log_evaluation_results

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(iterations):
        optimizer.zero_grad()

        # Play a batch of games against minimax
        batch_probs, batch_outcomes = play_batch_against_minimax(
            model, batch_size, temperature, epsilon, depth
        )

        # calculate the winrate
        win_rate = sum(o == 2 for o in batch_outcomes) / batch_size

        # Calculate the loss for the batch
        batch_loss = torch.tensor(0.0, requires_grad=True)
        for probs, outcome in zip(batch_probs, batch_outcomes):
            loss = loss_fn(probs, outcome, win_rate, player=2, gamma=gamma)
            batch_loss = batch_loss + loss

        # Normalize the loss
        batch_loss = batch_loss / batch_size

        # model confidence
        model_confidence = (
            sum(sum(q.item() for q in probs) / len(probs) for probs in batch_probs)
            / batch_size
        )

        # Log the average loss, win rate and model confidence to wandb
        safe_log_to_wandb(
            {
                "loss": batch_loss.item(),
                "win_rate": win_rate,
                "model_confidence": model_confidence,
            }
        )

        # Backpropagate the loss
        batch_loss.backward()
        optimizer.step()

        # Evaluate the model every eval_interval iterations
        if i % eval_interval == 0:
            eval_results = evaluate_model(model, num_games=eval_games)
            log_evaluation_results(eval_results)

    return model


def train_against_minimax_supervised(
    model: DecisionModel,
    batches: int = 1000,
    learning_rate: float = 0.01,
    eval_interval: int = 100,
    eval_games: int = 100,
    temperature: float = 1.0,
    epsilon: float = 0,
    gamma: float = 0.9,
    depth_teacher: int = 1,
    depth_opponent: int = 1,
    batch_size: int = 128,
    save_prob: float = 0.0,
) -> DecisionModel:
    """
    Train the model using two minimax players. One minimax opponent and one minimax teacher.
    The model is trained to predict the teacher's moves.
    Randomly decides who starts each game within a batch.
    Stops early if accuracy is above 95% for 5 or more consecutive batches.
    Saves games with probability save_prob.
    """
    from torch.nn import functional as F
    from evaluations import evaluate_model, log_evaluation_results
    import random

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # variables for early stopping
    high_accuracy_count = 0
    accuracy_threshold = 0.95

    for i in range(batches):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, requires_grad=True)
        correct_predictions = 0
        total_moves = 0
        wins = 0
        draws = 0

        for _ in range(batch_size):
            board = ConnectFour()
            game_moves = []  # list of tuples (model_output, teacher_move)
            game_states = [board.state.tolist()]  # For saving games

            # Randomly decide who starts
            model_player = random.choice([1, 2])

            current_player = 1

            while True:
                if (
                    current_player == model_player
                ):  # Model's turn (trying to predict teacher's move)
                    teacher_move = minimax_move(
                        board, player=current_player, depth=depth_teacher
                    )
                    model_output = model(torch.Tensor(board.state))
                    model_move, _ = get_next_model_move(
                        model, board, temperature, epsilon
                    )

                    game_moves.append((model_output, teacher_move))

                    if model_move == teacher_move:
                        correct_predictions += 1
                    total_moves += 1

                    # we execute the move that the teacher chose
                    move = teacher_move
                else:  # Minimax opponent's turn
                    move = minimax_move(
                        board, player=current_player, depth=depth_opponent
                    )

                board = make_move(board, current_player, move)
                game_states.append(board.state.tolist())  # For saving games

                if is_in_terminal_state(board) != 0:
                    status = is_in_terminal_state(board)
                    if status == model_player:
                        wins += 1
                    elif status == 3:
                        draws += 1
                    break

                current_player = 3 - current_player

            # Calculate loss for this game
            for model_output, teacher_move in game_moves:
                loss = F.cross_entropy(
                    model_output.unsqueeze(0), torch.tensor([teacher_move])
                )
                batch_loss = batch_loss + loss

            # Save the game with probability save_prob
            if random.random() < save_prob:
                saved_game = SavedGame(
                    depth_player1=depth_opponent
                    if model_player == 2
                    else depth_teacher,
                    depth_player2=depth_teacher
                    if model_player == 2
                    else depth_opponent,
                    result=status,
                    states=game_states,
                )
                saved_game.save_to_file("saved_games")

        # Normalize the loss
        batch_loss = batch_loss / batch_size

        # Backpropagate the loss
        batch_loss.backward()
        optimizer.step()

        # Calculate accuracy and rates
        accuracy = correct_predictions / total_moves if total_moves > 0 else 0
        win_rate = wins / batch_size
        draw_rate = draws / batch_size

        # Print statistics
        print(f"Batch {i+1}/{batches}")
        print(f"Loss: {batch_loss.item():.4f}, Accuracy: {accuracy:.4f}")
        print(f"Win rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}")

        # Log metrics to wandb
        safe_log_to_wandb(
            {
                "supervised_loss": batch_loss.item(),
                "accuracy": accuracy,
                "win_rate": win_rate,
                "draw_rate": draw_rate,
            }
        )

        # Check for early stopping
        if accuracy > accuracy_threshold:
            high_accuracy_count += 1
            if high_accuracy_count >= 5:
                print(
                    f"Early stopping at batch {i+1} due to high accuracy for 5 consecutive batches."
                )
                break
        else:
            high_accuracy_count = 0

        # Evaluate the model every eval_interval iterations
        if i % eval_interval == 0:
            eval_results = evaluate_model(model, num_games=eval_games)
            log_evaluation_results(eval_results)

    return model


def minimax_games(
    num_games: int,
    depth_player1: int = 1,
    depth_player2: int = 1,
    save_prob: float = 0.0,
):
    """
    Have two minimax players play against each other num_games times and save each game with a probability of save_prob.
    """
    saved_games = []

    for _ in range(num_games):
        board = ConnectFour()
        game_states = [board.state.tolist()]
        current_player = 1

        while True:
            if current_player == 1:
                move = minimax_move(board, player=1, depth=depth_player1)
            else:
                move = minimax_move(board, player=2, depth=depth_player2)

            board = make_move(board, current_player, move)
            game_states.append(board.state.tolist())

            if is_in_terminal_state(board) != 0:
                break

            current_player = 3 - current_player

        result = is_in_terminal_state(board)
        saved_game = SavedGame(
            depth_player1=depth_player1,
            depth_player2=depth_player2,
            result=result,
            states=game_states,
        )
        saved_games.append(saved_game)

        # Save the game to a file with probability save_prob
        if random.random() < save_prob:
            saved_game.save_to_file("saved_games")

    # Print statistics
    print(f"Number of games: {num_games}")
    print(
        f"Number of wins for player 1: {sum(1 for game in saved_games if game.result == 1)}"
    )
    print(
        f"Number of wins for player 2: {sum(1 for game in saved_games if game.result == 2)}"
    )
    print(f"Number of draws: {sum(1 for game in saved_games if game.result == 3)}")
    print(f"Approximate number of games saved: {int(num_games * save_prob)}")
    print("Games saved to: saved_games/")

    return saved_games


if __name__ == "__main__":
    minimax_games(100, depth_player1=1, depth_player2=5, save_prob=0.0)
