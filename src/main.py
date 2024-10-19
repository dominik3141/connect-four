import torch
import wandb
from model import DecisionModel
from self_play import train_using_self_play


if __name__ == "__main__":
    # HYPERPARAMETERS
    hyperparams = {
        "iterations": 1000,  # number of games to play
        "learning_rate": 0.001,
        "eval_interval": 50,
        "eval_games": 25,  # number of games to play in evaluation
        "eval_depth": 4,  # depth for minimax
        "temperature": 1.0,  # temperature for softmax
        "epsilon": 0.1,  # epsilon-greedy parameter
        "gamma": 0.95,
        "load_model": False,
        "save_model": False,
        "use_wandb": True,
        "save_prob": 0.001,  # probability of saving a game
    }

    # initialize the model
    model = DecisionModel()

    # load the weights from the previous run
    if hyperparams["load_model"]:
        model.load_state_dict(torch.load("model.pth"))

    # Initialize wandb
    if hyperparams["use_wandb"]:
        run = wandb.init(
            project="connect_four_self_play",
            save_code=True,
            settings=wandb.Settings(
                code_dir="src"
            ),  # only save the src directory (relative to the cwd of the terminal executing the script)
        )

    # count the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # log the model architecture
    if hyperparams["use_wandb"]:
        wandb.watch(model, log="all", log_freq=100)
        wandb.config.update(hyperparams)

    ###########################################################################
    # TRAINING
    ###########################################################################

    # train the model using self-play
    model = train_using_self_play(
        model,
        iterations=hyperparams["iterations"],
        learning_rate=hyperparams["learning_rate"],
        eval_interval=hyperparams["eval_interval"],
        temperature=hyperparams["temperature"],
        epsilon=hyperparams["epsilon"],
        eval_games=hyperparams["eval_games"],
        eval_depth=hyperparams["eval_depth"],
    )

    ###########################################################################

    if hyperparams["save_model"]:
        # save the model
        torch.save(model.state_dict(), "model.pth")

        if hyperparams["use_wandb"]:
            # save the model to wandb
            wandb.save("model.pth")

    # Finish the wandb run
    if hyperparams["use_wandb"]:
        run.finish()
