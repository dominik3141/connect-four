import torch
from torch import Tensor
import wandb
from utils import safe_log_to_wandb


def loss_fn(
    probs: Tensor,
    outcome: int,
    win_ratio: float = None,
    player: int = 2,
    gamma: float = 0.5,
) -> Tensor:
    def calc_win_reward(win_ratio: float) -> float:
        return max(
            200 * (1 - win_ratio), 10
        )  # if winning is sparse, the reward is higher

    if win_ratio is not None:
        win_reward = calc_win_reward(win_ratio)
    else:  # if no win ratio is provided, the reward is 5
        win_reward = 5

    if wandb.run is not None:
        safe_log_to_wandb({"win_reward": win_reward})

    if outcome == 3:  # Draw
        reward = -0.5
    elif outcome == player:  # Player wins
        reward = win_reward
    elif outcome == 0:  # invalid outcome
        # crash the program
        raise ValueError("Invalid outcome")
    else:  # Player loses
        reward = -1.0

    # Modify reward based on confidence (probs)
    # For wins: lower confidence => higher reward
    # For losses: higher confidence => higher penalty
    # if outcome == player:
    #     # Reverse confidence scaling for smart moves with low confidence
    #     discounted_losses = reward * (2 - probs)  # More reward for lower confidence
    # else:
    #     # Increase penalty for high-confidence bad moves
    #     discounted_losses = reward * probs  # More penalty for higher confidence

    num_moves = len(probs)
    discount_factors = torch.tensor([gamma**i for i in range(num_moves)])
    discounted_losses = reward * discount_factors * probs

    loss = torch.sum(discounted_losses)

    # normalize the loss smarter
    loss = loss / sum(discount_factors)

    # change the sign of the loss (in order for rewards to be maximized)
    loss = -loss

    # log the reward to wandb
    if wandb.run is not None:
        safe_log_to_wandb({"reward": reward})

    print(f"DEBUG: reward: {reward}, discounted_losses: {discounted_losses}")
    print(f"DEBUG: probs: {probs}")
    print(f"DEBUG: loss: {loss}")

    return loss
