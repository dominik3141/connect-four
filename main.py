import torch
import wandb
from model import DecisionModel
from minimax import train_against_minimax_supervised


if __name__ == "__main__":
    # HYPERPARAMETERS
    learning_rate = 0.005
    batches = 3
    eval_interval = 5
    eval_games = 50
    eval_depth = 1
    temperature = 1.0  # temperature for softmax
    epsilon = 0.0  # epsilon-greedy parameter
    gamma = 0.95
    depth_teacher = 1
    depth_opponent = 1
    batch_size = 8
    load_model = True
    save_model = False
    use_wandb = False

    # initialize the model
    model = DecisionModel()

    # load the weights from the previous run
    if load_model:
        model.load_state_dict(torch.load("model.pth"))

    # Initialize wandb
    if use_wandb:
        run = wandb.init(project="connect_four")

    # count the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # log the model architecture
    if use_wandb:
        wandb.watch(model, log="all", log_freq=batch_size)
        wandb.config.update(
            {
                "learning_rate": learning_rate,
                "iterations": batches,
                "eval_interval": eval_interval,
                "eval_games": eval_games,
                "eval_depth": eval_depth,
                "model_architecture": str(model),
                "temperature": temperature,
                "epsilon": epsilon,
                "gamma": gamma,
                "depth_teacher": depth_teacher,
                "depth_opponent": depth_opponent,
                "batch_size": batch_size,
                "load_model": load_model,
                "num_params": num_params,
            }
        )

    model = train_against_minimax_supervised(
        model,
        batches=batches,
        learning_rate=learning_rate,
        eval_interval=eval_interval,
        eval_games=eval_games,
        temperature=temperature,
        epsilon=epsilon,
        gamma=gamma,
        depth_teacher=depth_teacher,
        depth_opponent=depth_opponent,
        batch_size=batch_size,
    )

    if save_model:
        # save the model
        torch.save(model.state_dict(), "model.pth")

        if use_wandb:
            # save the model to wandb
            wandb.save("model.pth")

    # Finish the wandb run
    if use_wandb:
        run.finish()
