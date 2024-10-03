import torch
from torch import Tensor
import wandb
from model import DecisionModel
from minimax import train_against_minimax


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
        wandb.log({"win_reward": win_reward})

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
        wandb.log({"reward": reward})

    print(f"DEBUG: reward: {reward}, discounted_losses: {discounted_losses}")
    print(f"DEBUG: probs: {probs}")
    print(f"DEBUG: loss: {loss}")

    return loss


if __name__ == "__main__":
    # HYPERPARAMETERS
    learning_rate = 0.005
    iterations = 10000
    eval_interval = 200
    eval_games = 10
    eval_depth = 1
    temperature = 1.0  # temperature for softmax
    epsilon = 0.0  # epsilon-greedy parameter
    gamma = 0.95
    train_depth = 1  # depth for minimax
    batch_size = 128
    load_model = False

    # initialize the model
    model = DecisionModel()

    # load the weights from the previous run
    if load_model:
        model.load_state_dict(torch.load("model.pth"))

    # Initialize wandb
    run = wandb.init(project="connect_four")

    # log the model architecture
    wandb.watch(model, log="all", log_freq=batch_size)
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "iterations": iterations,
            "eval_interval": eval_interval,
            "eval_games": eval_games,
            "eval_depth": eval_depth,
            "model_architecture": str(model),
            "temperature": temperature,
            "epsilon": epsilon,
            "gamma": gamma,
            "train_depth": train_depth,
            "batch_size": batch_size,
            "load_model": load_model,
        }
    )

    model = train_against_minimax(
        model,
        iterations=iterations,
        learning_rate=learning_rate,
        eval_interval=eval_interval,
        temperature=temperature,
        epsilon=epsilon,
        depth=train_depth,
        batch_size=batch_size,
        gamma=gamma,
    )

    # save the model
    torch.save(model.state_dict(), "model.pth")

    # save the model to wandb
    wandb.save("model.pth")

    # Finish the wandb run
    run.finish()
