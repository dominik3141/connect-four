from engine import ConnectFour
from typing import Tuple, List
import copy
from engine import is_in_terminal_state, make_move, is_legal
from model import DecisionModel, get_next_model_move
from torch import Tensor
import torch
from utils import safe_log_to_wandb, SavedGame


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
    depth_teacher: int = 3,
    depth_opponent: int = 1,
    batch_size: int = 128,
) -> DecisionModel:
    """
    Train the model using two minimax players. One minimax opponent and one minimax teacher.
    The model is trained to predict the teacher's moves.
    Stops early if accuracy is above 95% for 5 or more consecutive batches.
    """
    from torch.nn import functional as F
    from evaluations import evaluate_model, log_evaluation_results

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

            current_player = 1
            while True:
                if current_player == 1:  # Minimax opponent's turn
                    move = minimax_move(board, depth=depth_opponent)
                else:  # Model's turn (trying to predict teacher's move)
                    teacher_move = minimax_move(board, depth=depth_teacher)
                    model_output = model(torch.Tensor(board.state))
                    move, _ = get_next_model_move(model, board, temperature, epsilon)

                    game_moves.append((model_output, teacher_move))

                    if move == teacher_move:
                        correct_predictions += 1
                    total_moves += 1

                board = make_move(board, current_player, move)

                current_player = 3 - current_player

                if is_in_terminal_state(board) != 0:
                    status = is_in_terminal_state(board)
                    if status == 2:
                        wins += 1
                    elif status == 3:
                        draws += 1
                    break

            # Calculate loss for this game
            for model_output, teacher_move in game_moves:
                loss = F.cross_entropy(
                    model_output.unsqueeze(0), torch.tensor([teacher_move])
                )
                batch_loss = batch_loss + loss

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


def minimax_games(num_games: int, depth_player1: int = 1, depth_player2: int = 1):
    """
    Have two minimax players play against each other num_games times and save each game.
    """
    saved_games = []

    for _ in range(num_games):
        board = ConnectFour()
        game_states = [board.state.tolist()]
        current_player = 1

        while True:
            if current_player == 1:
                move = minimax_move(board, depth=depth_player1)
            else:
                move = minimax_move(board, depth=depth_player2)

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

        # Save the game to a file
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
    print("Games saved to: saved_games/")

    return saved_games


if __name__ == "__main__":
    minimax_games(1, depth_player1=5, depth_player2=5)
