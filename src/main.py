import torch
import wandb
from model import DecisionModel, ValueModel
from self_play import train_using_self_play
import signal
import sys
import os


if __name__ == "__main__":
    # HYPERPARAMETERS
    hyperparams = {
        "iterations": 10000,  # Number of BATCHES to run
        "batch_size": 64,  # Games per batch
        "log_interval": 10,  # Log metrics every N batches
        "learning_rate": 0.0001,  # LR might need adjustment with batching
        "temperature": 1.5,  # Temperature for softmax exploration
        "epsilon": 0.15,  # Epsilon-greedy exploration
        "discount_factor": 0.97,  # Discount factor for value estimates
        "load_model": True,  # Load pre-existing model weights
        "save_model": True,  # Save model weights during/after training
        "use_wandb": True,  # Use Weights & Biases for logging
        "policy_model_path": "policy_model.pth",
        "value_model_path": "value_model.pth",
    }

    # initialize the models
    policy_model = DecisionModel()
    value_model = ValueModel()

    # load the weights from the previous run if file exists
    if hyperparams["load_model"]:
        if os.path.exists(hyperparams["policy_model_path"]):
            print(f"Loading policy model from {hyperparams['policy_model_path']}")
            policy_model.load_state_dict(torch.load(hyperparams["policy_model_path"]))
        else:
            print(
                f"Policy model file not found at {hyperparams['policy_model_path']}, starting fresh."
            )
        if os.path.exists(hyperparams["value_model_path"]):
            print(f"Loading value model from {hyperparams['value_model_path']}")
            value_model.load_state_dict(torch.load(hyperparams["value_model_path"]))
        else:
            print(
                f"Value model file not found at {hyperparams['value_model_path']}, starting fresh."
            )

    # Initialize wandb
    run = None  # Initialize run to None
    if hyperparams["use_wandb"]:
        run = wandb.init(
            project="connect_four_actor_critic",  # Updated project name?
            config=hyperparams,  # Log hyperparameters
            save_code=True,
            # settings=wandb.Settings(code_dir="src"), # Optional: configure code saving
        )
        # Log model architecture (optional, can be verbose)
        # wandb.watch(policy_model, log="all", log_freq=hyperparams["log_interval"] * hyperparams["batch_size"])
        # wandb.watch(value_model, log="all", log_freq=hyperparams["log_interval"] * hyperparams["batch_size"])

    # count the number of parameters in the models
    policy_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    value_params = sum(p.numel() for p in value_model.parameters() if p.requires_grad)
    print(f"Policy Model Parameters: {policy_params:,}")
    print(f"Value Model Parameters: {value_params:,}")
    if hyperparams["use_wandb"] and run:
        wandb.summary["policy_params"] = policy_params
        wandb.summary["value_params"] = value_params

    # Add signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Saving models and exiting...")
        if hyperparams["save_model"]:
            torch.save(policy_model.state_dict(), hyperparams["policy_model_path"])
            torch.save(value_model.state_dict(), hyperparams["value_model_path"])
            if hyperparams["use_wandb"] and run:
                # Save as artifacts
                wandb.save(hyperparams["policy_model_path"])
                wandb.save(hyperparams["value_model_path"])
                print("Models saved to W&B Artifacts.")
        if hyperparams["use_wandb"] and run:
            run.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    ###########################################################################
    # TRAINING
    ###########################################################################

    try:
        # train the models using self-play with batching
        train_using_self_play(
            policy_model,
            value_model,
            iterations=hyperparams["iterations"],
            batch_size=hyperparams["batch_size"],
            log_interval=hyperparams["log_interval"],
            learning_rate=hyperparams["learning_rate"],
            temperature=hyperparams["temperature"],
            epsilon=hyperparams["epsilon"],
            discount_factor=hyperparams["discount_factor"],
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (not Ctrl+C).")
        # Consider saving here too, although signal handler should catch Ctrl+C
    finally:
        # Ensure saving happens even if loop finishes normally or other exception occurs
        print("\nTraining finished or stopped.")
        if hyperparams["save_model"]:
            print("Saving final models...")
            torch.save(policy_model.state_dict(), hyperparams["policy_model_path"])
            torch.save(value_model.state_dict(), hyperparams["value_model_path"])
            print(
                f"Models saved locally to {hyperparams['policy_model_path']} and {hyperparams['value_model_path']}"
            )
            # Only save to wandb if the run is still active (wasn't finished by signal handler)
            if hyperparams["use_wandb"] and run and run.is_running:
                # Save as artifacts
                wandb.save(hyperparams["policy_model_path"])
                wandb.save(hyperparams["value_model_path"])
                print("Models saved to W&B Artifacts.")

        # Finish the wandb run if it hasn't been finished by signal handler
        if hyperparams["use_wandb"] and run and run.is_running:
            print("Finishing W&B run...")
            run.finish()

    print("Exiting.")
