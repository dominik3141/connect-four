import torch
from torch import Tensor


def loss_fn(
    probs: Tensor,
    outcome: int,
    win_ratio: float = None,
    player: int = 2,
    gamma: float = 0.5,
    debug: bool = False,
) -> Tensor:
    if outcome == 2:  # AI wins
        reward = 1.0
    elif outcome == 1:  # Player wins
        reward = -1.0
    elif outcome == 3:  # Draw
        reward = 0.0
    else:  # invalid outcome
        raise ValueError("Invalid outcome")

    num_moves = len(probs)
    discount_factors = torch.tensor([gamma**i for i in range(num_moves)])
    discounted_losses = reward * discount_factors * probs

    loss = torch.sum(discounted_losses)

    # normalize the loss smarter
    loss = loss / sum(discount_factors)

    # change the sign of the loss (in order for rewards to be maximized)
    loss = -loss

    if debug:
        print(f"DEBUG: reward: {reward}, discounted_losses: {discounted_losses}")
        print(f"DEBUG: probs: {probs}")
        print(f"DEBUG: loss: {loss}")

    return loss
