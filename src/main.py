import torch
import wandb
from .model import ValueModel
from .self_play import train_using_self_play
import signal
import os
import copy

# Flag to track if wandb run was finished by signal handler - REMOVED
# _wandb_run_finished_by_handler = False

if __name__ == "__main__":
    # HYPERPARAMETERS
    hyperparams = {
        "iterations": 1000000,
        "batch_size": 64,
        "log_interval": 10,
        "learning_rate": 0.001,
        # --- Exploration Params for Value-Based Moves --- #
        "online_temperature": 0.2,  # Exploration temp for online value model
        "online_epsilon": 0.0,  # Epsilon-greedy for online value model
        "frozen_temperature": 0.2,  # Greedy temp for frozen value model
        "frozen_epsilon": 0.0,  # Epsilon-greedy for frozen value model
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
            save_code=True,  # Keep saving code automatically
            # `settings=wandb.Settings(job_source="code")` might be needed depending on wandb version/setup
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
        # global _wandb_run_finished_by_handler - REMOVED
        print("\nCtrl+C detected. Initiating graceful shutdown...")
        # Simply raise KeyboardInterrupt to trigger the finally block
        raise KeyboardInterrupt
        # --- REMOVED OLD HANDLER LOGIC ---

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
            # Pass the wandb run object if available
            wandb_run=run,  # ADDED wandb_run
        )
    except KeyboardInterrupt:
        # This block now catches Ctrl+C as well via the signal handler raising it
        print("\nTraining interrupted by user (Ctrl+C or other).")
    finally:
        print("\nTraining finished or stopped. Proceeding with cleanup...")

        if hyperparams["save_model"]:
            print("Saving final frozen and online value models locally...")
            # Save FROZEN value model locally
            try:
                torch.save(
                    target_value_model.state_dict(), hyperparams["value_model_path"]
                )
                print(
                    f"Frozen value model saved locally to {hyperparams['value_model_path']}"
                )
            except Exception as e:
                print(f"Error saving frozen model locally: {e}")

            # Save ONLINE value model locally
            try:
                torch.save(
                    value_model.state_dict(), hyperparams["online_value_model_path"]
                )
                print(
                    f"Online value model saved locally to {hyperparams['online_value_model_path']}"
                )
            except Exception as e:
                print(f"Error saving online model locally: {e}")

            # Log models as W&B Artifacts if enabled and files exist
            if hyperparams["use_wandb"] and run:
                print("Logging final models as W&B Artifacts...")
                try:
                    # Create artifact for the frozen model if it exists
                    if os.path.exists(hyperparams["value_model_path"]):
                        frozen_artifact = wandb.Artifact(
                            "frozen_value_model",
                            type="model",
                            description="Final frozen value model state dict",
                            metadata=hyperparams,  # Optional: Add hyperparams for context
                        )
                        frozen_artifact.add_file(hyperparams["value_model_path"])
                        run.log_artifact(frozen_artifact)
                        print(f"Logged '{frozen_artifact.name}' artifact.")
                    else:
                        print(
                            f"Skipping frozen model artifact logging: File not found at {hyperparams['value_model_path']}"
                        )

                    # Create artifact for the online model if it exists
                    if os.path.exists(hyperparams["online_value_model_path"]):
                        online_artifact = wandb.Artifact(
                            "online_value_model",
                            type="model",
                            description="Final online value model state dict",
                            metadata=hyperparams,  # Optional: Add hyperparams for context
                        )
                        online_artifact.add_file(hyperparams["online_value_model_path"])
                        run.log_artifact(online_artifact)
                        print(f"Logged '{online_artifact.name}' artifact.")
                    else:
                        print(
                            f"Skipping online model artifact logging: File not found at {hyperparams['online_value_model_path']}"
                        )

                except Exception as e:
                    print(f"Error logging model artifacts to W&B: {e}")

        else:
            print("Local saving skipped as save_model is False.")

        # Finish the W&B run if it exists
        if hyperparams["use_wandb"] and run:
            print("Finishing W&B run...")
            run.finish()

    print("Exiting.")

# Ensure last line has newline if needed
