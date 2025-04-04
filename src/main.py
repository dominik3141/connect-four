import torch
import wandb
from model import DecisionModel, ValueModel
from self_play import train_using_self_play
import signal
import sys
import os
import copy  # Add copy for deep copying models

# Flag to track if wandb run was finished by signal handler
_wandb_run_finished_by_handler = False

if __name__ == "__main__":
    # HYPERPARAMETERS
    hyperparams = {
        "iterations": 1000000,  # Number of BATCHES to run
        "batch_size": 64,  # Games per batch
        "log_interval": 10,  # Log metrics every N batches
        "learning_rate": 0.0001,  # LR might need adjustment with batching
        # --- Separate Exploration Params --- #
        "online_temperature": 2.0,  # Exploration temp for the learning model
        "online_epsilon": 0.2,  # Epsilon-greedy exploration for the learning model
        "frozen_temperature": 0.0,  # Near-greedy temp for the opponent model
        "frozen_epsilon": 0.0,  # No epsilon-greedy for the opponent model
        # ----------------------------------- #
        "discount_factor": 0.95,  # Discount factor for value estimates
        "entropy_coefficient": 0.00,  # Coefficient for entropy regularization bonus (set to 0 to disable)
        # --- Training Infrastructure ---
        "load_model": False,  # Load pre-existing model weights (loads into ONLINE, then copies to TARGET)
        "save_model": True,  # Save model locally (always saves to W&B even if False)
        "use_wandb": True,  # Use Weights & Biases for logging
        "policy_model_path": "policy_model.pth",  # Path for TARGET policy model
        "value_model_path": "value_model.pth",  # Path for TARGET value model
        "online_policy_model_path": "online_policy_model.pth",  # Path for ONLINE policy model
        "online_value_model_path": "online_value_model.pth",  # Path for ONLINE value model
        # --- Evaluation & Target Update ---
        "target_update_freq": 100,  # Update target networks every N batches
        "eval_games": 30,  # Games per evaluation playoff vs target
        "win_rate_threshold": 0.55,  # Online must win >55% vs target to update
        "stacker_eval_games": 100,  # Games for stacker evaluation
        "force_replace_model": False,  # Always replace frozen model if True
    }

    # Initialize the online models
    policy_model = DecisionModel()
    value_model = ValueModel()

    # Initialize the target models as copies of the online models
    target_policy_model = copy.deepcopy(policy_model)
    target_value_model = copy.deepcopy(value_model)

    # Freeze target network parameters (they are updated manually)
    for param in target_policy_model.parameters():
        param.requires_grad = False
    for param in target_value_model.parameters():
        param.requires_grad = False

    # load the weights from the previous run if file exists (loads into ONLINE models)
    # Note: The paths used here are the TARGET paths, as per previous logic.
    # This implies we are resuming training based on the last saved TARGET state.
    if hyperparams["load_model"]:
        if os.path.exists(hyperparams["policy_model_path"]):
            print(
                f"Loading online policy model from {hyperparams['policy_model_path']}"
            )
            policy_model.load_state_dict(torch.load(hyperparams["policy_model_path"]))
        else:
            print(
                f"Target policy model file not found at {hyperparams['policy_model_path']}, starting fresh."
            )
        if os.path.exists(hyperparams["value_model_path"]):
            print(f"Loading online value model from {hyperparams['value_model_path']}")
            value_model.load_state_dict(torch.load(hyperparams["value_model_path"]))
        else:
            print(
                f"Target value model file not found at {hyperparams['value_model_path']}, starting fresh."
            )

        # Ensure target networks start with the same weights as the loaded online networks
        target_policy_model.load_state_dict(policy_model.state_dict())
        target_value_model.load_state_dict(value_model.state_dict())
        print("Target networks initialized with loaded online network weights.")

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
        wandb.watch(
            policy_model,
            log="all",
            log_freq=hyperparams["log_interval"] * hyperparams["batch_size"],
        )
        wandb.watch(
            value_model,
            log="all",
            log_freq=hyperparams["log_interval"] * hyperparams["batch_size"],
        )

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
        # Save locally ONLY if save_model is True
        if hyperparams["save_model"]:
            # Save TARGET models locally
            torch.save(
                target_policy_model.state_dict(), hyperparams["policy_model_path"]
            )
            torch.save(target_value_model.state_dict(), hyperparams["value_model_path"])
            print(
                f"Target models saved locally to {hyperparams['policy_model_path']} and {hyperparams['value_model_path']}"
            )
            # Save ONLINE models locally
            torch.save(
                policy_model.state_dict(), hyperparams["online_policy_model_path"]
            )
            torch.save(value_model.state_dict(), hyperparams["online_value_model_path"])
            print(
                f"Online models saved locally to {hyperparams['online_policy_model_path']} and {hyperparams['online_value_model_path']}"
            )
        else:
            print("Local saving skipped as save_model is False.")

        # Save to W&B ONLY if use_wandb is True
        if hyperparams["use_wandb"] and run:
            # Upload TARGET models as artifacts
            wandb.save(hyperparams["policy_model_path"])
            wandb.save(hyperparams["value_model_path"])
            # Upload ONLINE models as artifacts
            wandb.save(hyperparams["online_policy_model_path"])
            wandb.save(hyperparams["online_value_model_path"])
            print("Target and Online models saved to W&B Artifacts.")

            # Finish W&B run
            global _wandb_run_finished_by_handler  # Declare we are modifying the global flag
            run.finish()
            _wandb_run_finished_by_handler = True  # Set the flag
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    ###########################################################################
    # TRAINING
    ###########################################################################

    try:
        # train the models using self-play with target networks
        train_using_self_play(
            policy_model,
            value_model,
            target_policy_model,
            target_value_model,
            iterations=hyperparams["iterations"],
            batch_size=hyperparams["batch_size"],
            log_interval=hyperparams["log_interval"],
            learning_rate=hyperparams["learning_rate"],
            # --- Pass Separate Exploration Params --- #
            online_temperature=hyperparams["online_temperature"],
            online_epsilon=hyperparams["online_epsilon"],
            frozen_temperature=hyperparams["frozen_temperature"],
            frozen_epsilon=hyperparams["frozen_epsilon"],
            # -------------------------------------- #
            # --- Pass Other Params --- #
            discount_factor=hyperparams["discount_factor"],
            entropy_coefficient=hyperparams["entropy_coefficient"],
            target_update_freq=hyperparams["target_update_freq"],
            eval_games=hyperparams["eval_games"],
            win_rate_threshold=hyperparams["win_rate_threshold"],
            stacker_eval_games=hyperparams["stacker_eval_games"],
            force_replace_model=hyperparams["force_replace_model"],
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (not Ctrl+C).")
        # Consider saving here too, although signal handler should catch Ctrl+C
    finally:
        # Ensure saving happens even if loop finishes normally or other exception occurs
        print("\nTraining finished or stopped.")

        # Save locally ONLY if save_model is True
        if hyperparams["save_model"]:
            print("Saving final target and online models locally...")
            # Save TARGET models locally
            torch.save(
                target_policy_model.state_dict(), hyperparams["policy_model_path"]
            )
            torch.save(target_value_model.state_dict(), hyperparams["value_model_path"])
            print(
                f"Target models saved locally to {hyperparams['policy_model_path']} and {hyperparams['value_model_path']}"
            )
            # Save ONLINE models locally
            torch.save(
                policy_model.state_dict(), hyperparams["online_policy_model_path"]
            )
            torch.save(value_model.state_dict(), hyperparams["online_value_model_path"])
            print(
                f"Online models saved locally to {hyperparams['online_policy_model_path']} and {hyperparams['online_value_model_path']}"
            )
        else:
            print("Local saving skipped as save_model is False.")

        # Save to W&B ONLY if use_wandb is True and run is still active
        if hyperparams["use_wandb"] and run and not _wandb_run_finished_by_handler:
            print("Saving final target and online models to W&B Artifacts...")
            # Upload TARGET models as artifacts
            wandb.save(hyperparams["policy_model_path"])
            wandb.save(hyperparams["value_model_path"])
            # Upload ONLINE models as artifacts
            wandb.save(hyperparams["online_policy_model_path"])
            wandb.save(hyperparams["online_value_model_path"])
            print("Target and Online models saved to W&B Artifacts.")

        # Finish the wandb run if it hasn't been finished by signal handler
        if hyperparams["use_wandb"] and run and not _wandb_run_finished_by_handler:
            print("Finishing W&B run...")
            run.finish()

    print("Exiting.")
