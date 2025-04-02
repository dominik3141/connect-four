import torch
import wandb
from model import DecisionModel, ValueModel
from self_play import train_using_self_play
import signal
import sys


if __name__ == "__main__":
    # HYPERPARAMETERS
    hyperparams = {
        "iterations": 100000000,  # number of games to play
        "learning_rate": 0.00001,
        "temperature": 1.0,  # temperature for softmax
        "epsilon": 0.5,  # epsilon-greedy parameter
        "discount_factor": 0.95,  # discount factor for value estimates
        "load_model": True,
        "save_model": True,
        "use_wandb": True,
    }

    # initialize the models
    policy_model = DecisionModel()
    value_model = ValueModel()

    # load the weights from the previous run
    if hyperparams["load_model"]:
        policy_model.load_state_dict(torch.load("policy_model.pth"))
        value_model.load_state_dict(torch.load("value_model.pth"))

    # Initialize wandb
    if hyperparams["use_wandb"]:
        run = wandb.init(
            project="connect_four_self_play",
            save_code=True,
            settings=wandb.Settings(code_dir="src"),
        )

    # count the number of parameters in the models
    policy_params = sum(p.numel() for p in policy_model.parameters())
    value_params = sum(p.numel() for p in value_model.parameters())
    print(f"Number of parameters in policy model: {policy_params}")
    print(f"Number of parameters in value model: {value_params}")

    # log the model architectures
    if hyperparams["use_wandb"]:
        wandb.watch(policy_model, log="all", log_freq=100)
        wandb.watch(value_model, log="all", log_freq=100)
        wandb.config.update(hyperparams)

    # Add signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Saving models and exiting...")
        if hyperparams["save_model"]:
            torch.save(policy_model.state_dict(), "policy_model.pth")
            torch.save(value_model.state_dict(), "value_model.pth")
            if hyperparams["use_wandb"]:
                wandb.save("policy_model.pth")
                wandb.save("value_model.pth")
        if hyperparams["use_wandb"]:
            run.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    ###########################################################################
    # TRAINING
    ###########################################################################

    try:
        # train the models using self-play
        train_using_self_play(
            policy_model,
            value_model,
            iterations=hyperparams["iterations"],
            learning_rate=hyperparams["learning_rate"],
            temperature=hyperparams["temperature"],
            epsilon=hyperparams["epsilon"],
            discount_factor=hyperparams["discount_factor"],
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving models and exiting...")

    ###########################################################################

    if hyperparams["save_model"]:
        # save the models
        torch.save(policy_model.state_dict(), "policy_model.pth")
        torch.save(value_model.state_dict(), "value_model.pth")

        if hyperparams["use_wandb"]:
            # save the models to wandb
            wandb.save("policy_model.pth")
            wandb.save("value_model.pth")

    # Finish the wandb run
    if hyperparams["use_wandb"]:
        run.finish()
