import torch
from torch import Tensor
import wandb
from evaluations import evaluate_model
from model import DecisionModel
from minimax import train_against_minimax


def loss_fn(
    probs: Tensor, outcome: int, player: int = 2, gamma: float = 0.95
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
    loss = torch.sum(discounted_rewards * probs)  # element-wise product

    # higher loss should indicate worse performance
    loss = -loss

    # normalize the loss
    loss = loss / num_moves

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
    depth = 2  # depth for minimax
    # initialize the model
    model = DecisionModel()

    # Initialize wandb
    run = wandb.init(project="connect_four")

    # log the model architecture
    wandb.watch(model, log="all", log_freq=10)
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "iterations": iterations,
            "eval_interval": eval_interval,
            "model_architecture": str(model),
            "temperature": temperature,
            "epsilon": epsilon,
            "depth": depth,
        }
    )

    # evaluate the untrained model
    print("Model evaluation before training:")
    initial_results = evaluate_model(
        model, num_games=eval_games, depth_for_minimax=eval_depth
    )
    print(initial_results)

    model = train_against_minimax(
        model,
        iterations=iterations,
        learning_rate=learning_rate,
        run=run,
        eval_interval=eval_interval,
        temperature=temperature,
        epsilon=epsilon,
        depth=depth,
    )

    # evaluate the trained model
    print("Model evaluation after training:")
    final_results = evaluate_model(
        model, num_games=eval_games, depth_for_minimax=eval_depth
    )
    print(final_results)

    # save the model
    torch.save(model.state_dict(), "model.pth")

    # save the model to wandb
    wandb.save("model.pth")

    # Log final comprehensive evaluation results to wandb
    for opponent, results in final_results.items():
        for outcome, count in results.items():
            wandb.log({f"final_{opponent}_{outcome}": count})

    # Finish the wandb run
    run.finish()
