from engine import ConnectFour
from typing import Tuple, List, Optional, Dict
from engine import is_in_terminal_state, make_move, is_legal
from model import DecisionModel, get_next_model_move
from torch import Tensor
import torch
from utils import safe_log_to_wandb, SavedGame
import random
import numpy as np
import time
import ctypes

# Load the shared library
lib = ctypes.CDLL("libs/libconnect4.so")  # Use the correct path for your system

# Define the argument and return types for the C function
lib.minimax_move.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # Board state (flattened)
    ctypes.c_int,  # Player
    ctypes.c_int,  # Depth
]
lib.minimax_move.restype = ctypes.c_int  # The best move as an integer


def minimax_move(board: ConnectFour, player: int, depth: int = 1) -> int:
    # Flatten the board state to pass it to the C function
    flat_state = board.state.flatten().astype(np.int32)
    flat_state_p = flat_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Call the C function
    best_move = lib.minimax_move(flat_state_p, player, depth)

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


def benchmark_minimax(num_games: int, test_depths: List[int]) -> Dict[int, float]:
    """
    Benchmark the minimax algorithm by playing num_games against itself with different depths.
    Returns a dictionary with depths as keys and average time per move as values.
    """
    results: Dict[int, float] = {}

    for depth in test_depths:
        total_time = 0
        total_moves = 0

        for _ in range(num_games):
            board = ConnectFour()
            current_player = 1

            while not is_in_terminal_state(board):
                start_time = time.time()
                move = minimax_move(board, current_player, depth)
                end_time = time.time()

                total_time += end_time - start_time
                total_moves += 1

                board = make_move(board, current_player, move)
                current_player = 3 - current_player

        avg_time_per_move = total_time / total_moves
        results[depth] = avg_time_per_move

    return results


# Example usage
if __name__ == "__main__":
    # Initialize a sample board with a winning row for Player 1
    sample_state = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [2, 2, 1, 2, 0, 0, 0],
        ]
    )

    board = ConnectFour(sample_state)
    print("Board State:")
    print(board)

    best_move = minimax_move(board, player=1, depth=7)
    print(f"Best Move: {best_move}")
