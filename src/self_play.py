from ai import loss_fn
from model import DecisionModel, get_next_model_move
from engine import ConnectFour, make_move, is_in_terminal_state
from evaluations import evaluate_model, log_evaluation_results
import torch
from torch import Tensor
from typing import Tuple
from utils import safe_log_to_wandb


def play_against_self(
    model: DecisionModel, temperature: float = 1.0, epsilon: float = 0
) -> Tuple[Tensor, Tensor, int]:
    """
    Play a game where the model plays against itself.
    Returns the move probabilities for both players and the game outcome.
    """
    board = ConnectFour()
    ai1_move_probs = []
    ai2_move_probs = []
    current_player = 1

    while True:
        move, prob = get_next_model_move(
            model, board, temperature=temperature, epsilon=epsilon
        )
        board = make_move(board, current_player, move)

        if current_player == 1:
            ai1_move_probs.append(prob)
        else:
            ai2_move_probs.append(prob)

        if is_in_terminal_state(board) != 0:
            break

        current_player = 3 - current_player  # Switch player

    status = is_in_terminal_state(board)

    # Reverse the move probabilities for correct discounting
    ai1_move_probs.reverse()
    ai2_move_probs.reverse()

    ai1_move_probs_tensor = torch.stack(ai1_move_probs)
    ai2_move_probs_tensor = torch.stack(ai2_move_probs)

    return ai1_move_probs_tensor, ai2_move_probs_tensor, status


def train_using_self_play(
    model: DecisionModel,
    iterations: int = 100,
    learning_rate: float = 0.01,
    eval_interval: int = 100,
    temperature: float = 1.0,
    epsilon: float = 0,
    eval_games: int = 100,
    eval_depth: int = 2,
) -> DecisionModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(iterations):
        optimizer.zero_grad()

        ai1_move_probs, ai2_move_probs, status = play_against_self(
            model, temperature=temperature, epsilon=epsilon
        )

        loss1 = loss_fn(ai1_move_probs, status, player=1)
        loss2 = loss_fn(ai2_move_probs, status, player=2)

        loss = loss1 + loss2

        # log the loss to wandb
        safe_log_to_wandb({"loss": loss})

        loss.backward()
        optimizer.step()

        # Evaluate the model every eval_interval iterations
        if i % eval_interval == 0:
            eval_results = evaluate_model(
                model, num_games=eval_games, depth_for_minimax=eval_depth
            )
            log_evaluation_results(eval_results)

    return model
