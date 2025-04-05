import torch
import wandb
from model import ValueModel
from self_play import train_using_self_play
import signal
import sys
import os
import copy

# Flag to track if wandb run was finished by signal handler
_wandb_run_finished_by_handler = False

if __name__ == "__main__":
    # HYPERPARAMETERS
    hyperparams = {
        "iterations": 1000000,
        "batch_size": 64,
        "log_interval": 10,
        "learning_rate": 0.001,
        # --- Exploration Params for Value-Based Moves --- #
        "online_temperature": 0.5,  # Exploration temp for online value model
        "online_epsilon": 0.1,  # Epsilon-greedy for online value model
        "frozen_temperature": 0.5,  # Greedy temp for frozen value model
        "frozen_epsilon": 0.1,  # Epsilon-greedy for frozen value model
        # ----------------------------------------------- #
        "discount_factor": 0.95,
        # --- Training Infrastructure ---
        "load_model": True,  # Load pre-existing value model weights
        "save_model": True,  # Save value model locally
        "use_wandb": True,
        # --- Removed policy_model_path ---
        "value_model_path": "value_model.pth",  # Path for FROZEN value model
        # --- Removed online_policy_model_path ---
        "online_value_model_path": "online_value_model.pth",  # Path for ONLINE value model
        # --- Evaluation & Target Update ---
        "target_update_freq": 100,
        "eval_games": 1000,
        "win_rate_threshold": 0.55,
        "stacker_eval_games": 100,
        "force_replace_model": False,  # Now controls replacement of frozen value model
    }

    # Initialize the online value model
    # --- Removed policy_model = DecisionModel() ---
    value_model = ValueModel()

    # Initialize the target value models as copies of the online models
    # --- Removed target_policy_model = copy.deepcopy(policy_model) ---
    target_value_model = copy.deepcopy(value_model)

    # Freeze target network parameters
    # --- Removed target_policy_model loop ---
    for param in target_value_model.parameters():
        param.requires_grad = False

    # Load the weights from the previous run if file exists (loads into ONLINE value model)
    if hyperparams["load_model"]:
        # --- Removed policy model loading ---
        if os.path.exists(hyperparams["value_model_path"]):
            print(f"Loading online value model from {hyperparams['value_model_path']}")
            # Load into the active online model
            value_model.load_state_dict(torch.load(hyperparams["value_model_path"]))
        else:
            print(
                f"Frozen value model file not found at {hyperparams['value_model_path']}, starting fresh."
            )

        # Ensure target network starts with the same weights as the loaded online network
        # --- Removed target_policy_model load_state_dict ---
        target_value_model.load_state_dict(value_model.state_dict())
        print("Target value network initialized with loaded online network weights.")

    # Initialize wandb
    run = None
    if hyperparams["use_wandb"]:
        run = wandb.init(
            project="connect_four_value_based",  # Changed project name
            config=hyperparams,
            save_code=True,
        )
        # --- Removed policy_model watch ---
        wandb.watch(
            value_model,
            log="all",
            log_freq=hyperparams["log_interval"] * hyperparams["batch_size"],
        )

    # Count the number of parameters in the models
    # --- Removed policy_params ---
    value_params = sum(p.numel() for p in value_model.parameters() if p.requires_grad)
    # --- Removed Policy Model Parameter print ---
    print(f"Value Model Parameters: {value_params:,}")
    if hyperparams["use_wandb"] and run:
        # --- Removed policy_params summary ---
        wandb.summary["value_params"] = value_params

    # Add signal handler for graceful shutdown
    def signal_handler(sig, frame):
        global _wandb_run_finished_by_handler  # Use the global flag
        print("\nCtrl+C detected. Saving models and exiting...")
        if hyperparams["save_model"]:
            # Save FROZEN value model locally
            # --- Removed policy model saving ---
            torch.save(target_value_model.state_dict(), hyperparams["value_model_path"])
            print(
                f"Frozen value model saved locally to {hyperparams['value_model_path']}"
            )
            # Save ONLINE value model locally
            torch.save(value_model.state_dict(), hyperparams["online_value_model_path"])
            print(
                f"Online value model saved locally to {hyperparams['online_value_model_path']}"
            )
        else:
            print("Local saving skipped as save_model is False.")

        if hyperparams["use_wandb"] and run:
            # Upload value models as artifacts
            # --- Removed policy model artifact saving ---
            wandb.save(hyperparams["value_model_path"])
            wandb.save(hyperparams["online_value_model_path"])
            print("Frozen and Online value models saved to W&B Artifacts.")

            if not _wandb_run_finished_by_handler:  # Avoid finishing twice
                run.finish()
                _wandb_run_finished_by_handler = True
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    ###########################################################################
    # TRAINING
    ###########################################################################

    try:
        # train the models using self-play with target networks (Value-Based)
        train_using_self_play(
            # --- Removed policy models ---
            value_model,
            target_value_model,  # Pass target value model
            iterations=hyperparams["iterations"],
            batch_size=hyperparams["batch_size"],
            log_interval=hyperparams["log_interval"],
            learning_rate=hyperparams["learning_rate"],
            # --- Pass Value-Based Exploration Params --- #
            online_temperature=hyperparams["online_temperature"],
            online_epsilon=hyperparams["online_epsilon"],
            frozen_temperature=hyperparams["frozen_temperature"],
            frozen_epsilon=hyperparams["frozen_epsilon"],
            # -------------------------------------- #
            # --- Pass Other Params --- #
            discount_factor=hyperparams["discount_factor"],
            # --- Removed entropy_coefficient ---
            target_update_freq=hyperparams["target_update_freq"],
            eval_games=hyperparams["eval_games"],
            win_rate_threshold=hyperparams["win_rate_threshold"],
            stacker_eval_games=hyperparams["stacker_eval_games"],
            force_replace_model=hyperparams["force_replace_model"],
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (not Ctrl+C).")
    finally:
        print("\nTraining finished or stopped.")

        if hyperparams["save_model"]:
            print("Saving final frozen and online value models locally...")
            # --- Removed policy model saving ---
            torch.save(target_value_model.state_dict(), hyperparams["value_model_path"])
            print(
                f"Frozen value model saved locally to {hyperparams['value_model_path']}"
            )
            torch.save(value_model.state_dict(), hyperparams["online_value_model_path"])
            print(
                f"Online value model saved locally to {hyperparams['online_value_model_path']}"
            )
        else:
            print("Local saving skipped as save_model is False.")

        if hyperparams["use_wandb"] and run and not _wandb_run_finished_by_handler:
            print("Saving final frozen and online value models to W&B Artifacts...")
            # --- Removed policy model artifact saving ---
            wandb.save(hyperparams["value_model_path"])
            wandb.save(hyperparams["online_value_model_path"])
            print("Frozen and Online value models saved to W&B Artifacts.")

        if hyperparams["use_wandb"] and run and not _wandb_run_finished_by_handler:
            print("Finishing W&B run...")
            run.finish()

    print("Exiting.")
