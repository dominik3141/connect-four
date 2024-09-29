import torch
from torch import Tensor
import wandb
from model import DecisionModel
from minimax import train_against_minimax


def loss_fn(
    probs: Tensor, outcome: int, player: int = 2, gamma: float = 0.95, run=None
) -> Tensor:
    if outcome == 3:  # Draw
        reward = -2.0
    elif outcome == player:  # Player wins
        reward = 4.0
    else:  # Player loses
        reward = -5.0

    num_moves = len(probs)
    discount_factors = torch.tensor([gamma**i for i in range(num_moves)])
    discounted_rewards = discount_factors * reward

    # Calculate log probabilities
    epsilon = 1e-8
    log_probs = torch.log(1 + probs + epsilon)  # Add small epsilon to avoid log(0)

    loss_non_log = torch.sum(discounted_rewards * probs)  # element-wise product

    print(f"DEBUG: loss_non_log: {-loss_non_log}")

    loss = torch.sum(discounted_rewards * log_probs)  # element-wise product

    # normalize the loss
    loss = loss / num_moves

    # change the sign of the loss (in order for rewards to be maximized)
    loss = -loss

    # log the reward to wandb
    if run is not None:
        run.log({"reward": reward})

    print(f"DEBUG: reward: {reward}, discounted_rewards: {discounted_rewards}")
    print(f"DEBUG: probs: {probs}, log_probs: {log_probs}")
    print(f"DEBUG: loss: {loss}")

    return loss


if __name__ == "__main__":
    # HYPERPARAMETERS
    learning_rate = 0.01
    iterations = 1000
    eval_interval = 200
    eval_games = 10
    eval_depth = 1
    temperature = 1.0  # temperature for softmax
    epsilon = 0.0  # epsilon-greedy parameter
    train_depth = 2  # depth for minimax
    batch_size = 64

    # initialize the model
    model = DecisionModel()

    # load the weights from the previous run
    # model.load_state_dict(torch.load("model.pth"))

    # Initialize wandb
    run = wandb.init(project="connect_four")

    # log the model architecture
    wandb.watch(model, log="all", log_freq=10)
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
            "train_depth": train_depth,
            "batch_size": batch_size,
        }
    )

    model = train_against_minimax(
        model,
        iterations=iterations,
        learning_rate=learning_rate,
        run=run,
        eval_interval=eval_interval,
        temperature=temperature,
        epsilon=epsilon,
        depth=train_depth,
        batch_size=batch_size,
    )

    # save the model
    torch.save(model.state_dict(), "model.pth")

    # save the model to wandb
    wandb.save("model.pth")

    # Finish the wandb run
    run.finish()
