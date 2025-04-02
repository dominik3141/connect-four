from model import DecisionModel, ValueModel, get_next_model_move
from engine import ConnectFour, make_move, is_in_terminal_state
import torch
from torch import Tensor
from typing import Tuple
from utils import safe_log_to_wandb


def board_to_tensor(board: ConnectFour) -> Tensor:
    """Convert a ConnectFour board to a tensor representation."""
    # Convert numpy array to tensor and transpose to match our expected format
    # The state is 6x7, we want 7x6
    tensor = torch.from_numpy(board.state).float()
    tensor = tensor.t()  # transpose to get 7x6

    # Convert player 2's pieces from 2 to -1
    tensor[tensor == 2] = -1

    # Make the tensor contiguous in memory
    return tensor.contiguous()


def play_against_self(
    policy_model: DecisionModel,
    value_model: ValueModel,
    temperature: float = 1.0,
    epsilon: float = 0,
    learning_rate: float = 0.01,
) -> Tuple[Tensor, Tensor, int]:
    """
    Play a game where the model plays against itself.
    Returns the move probabilities for both players and the game outcome.
    """
    board = ConnectFour()
    player_1_value_estimates = []
    player_2_value_estimates = []
    current_player = 1
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)

    board_tensor = board_to_tensor(board)
    old_value = value_model(board_tensor, current_player)
    while True:
        move, prob = get_next_model_move(
            policy_model, board, temperature=temperature, epsilon=epsilon
        )
        # execute the move
        board = make_move(board, current_player, move)

        # calculate the loss for the move based on the value model
        board_tensor = board_to_tensor(board)
        value = value_model(board_tensor, current_player)

        # Calculate the TD error (simplified advantage)
        td_error = (
            value - old_value
        ).detach()  # Detach: critic guides actor, but actor gradients shouldn't flow back into critic here

        # Calculate policy loss: -log_prob * TD_error
        # We want to increase the log_prob of actions that led to a positive TD_error
        # Loss is negative log_prob * TD_error to achieve this via gradient descent
        # Add epsilon for numerical stability against log(0)
        epsilon_log = 1e-9
        policy_loss = -torch.log(prob + epsilon_log) * td_error

        # adjust the policy model based on the loss
        optimizer_policy.zero_grad()
        policy_loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer_policy.step()

        if current_player == 1:
            player_1_value_estimates.append(value)  # keep computation graph
        else:
            player_2_value_estimates.append(value)  # keep computation graph

        if is_in_terminal_state(board) != 0:
            break

        current_player = 3 - current_player  # Switch player
        old_value = value

    status = is_in_terminal_state(board)

    # Reverse the move probabilities for correct discounting
    player_1_value_estimates.reverse()
    player_2_value_estimates.reverse()

    player_1_value_estimates_tensor = torch.stack(player_1_value_estimates)
    player_2_value_estimates_tensor = torch.stack(player_2_value_estimates)

    return player_1_value_estimates_tensor, player_2_value_estimates_tensor, status


def train_using_self_play(
    policy_model: DecisionModel,
    value_model: ValueModel,
    iterations: int = 100,
    learning_rate: float = 0.01,
    temperature: float = 1.0,
    epsilon: float = 0,
    discount_factor: float = 0.1,
) -> None:
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=learning_rate)
    running_loss = 0.0
    iteration_count = 0

    for i in range(iterations):
        optimizer_value.zero_grad()

        player_1_value_estimates, player_2_value_estimates, status = play_against_self(
            policy_model, value_model, temperature=temperature, epsilon=epsilon
        )

        # calculate the loss for the value model
        # this should be the difference between the predicted outcome and the actual outcome
        # the actual outcome is 1 if player 1 wins, -1 if player 2 wins, and 0 if it's a draw

        # calculate loss for player 1
        """
        How this calculation should look like if using loops:
        loss = 0
        for prediction in player_1_value_estimates:
            loss+=math.abs(score(status, 1)-prediction)
        """
        discount = torch.exp(
            torch.arange(
                len(player_1_value_estimates), device=player_1_value_estimates.device
            )
            * -discount_factor
        )
        losses_1 = (
            torch.abs(
                torch.full_like(player_1_value_estimates, _score(status, 1))
                - player_1_value_estimates
            )
            * discount
        ).sum()
        losses_2 = (
            torch.abs(
                torch.full_like(player_2_value_estimates, _score(status, 2))
                - player_2_value_estimates
            )
            * discount[: len(player_2_value_estimates)]
        ).sum()
        loss = losses_1 + losses_2

        # Track running loss
        running_loss += loss.item()
        iteration_count += 1

        # Only log after we have at least 100 iterations worth of data
        if iteration_count >= 100 and i % 100 == 0:
            print(f"Average loss for the value model: {running_loss / iteration_count}")

            safe_log_to_wandb(
                {
                    "loss_value_model": running_loss / iteration_count,
                }
            )

            running_loss = 0.0  # Reset running loss
            iteration_count = 0  # Reset iteration count

        loss.backward()
        optimizer_value.step()


def _score(status: int, player: int) -> int:
    """
    Calculate the score of a game.
    """
    if status == player:  # win
        return 1
    elif status == 3:  # stalemate
        return 0.5
    elif status == 0:  # game has not ended
        raise ValueError("Game has not ended")
    else:
        return 0
