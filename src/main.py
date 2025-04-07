import mlx.core as mx
import mlx.nn as nn
import wandb
from .model import ValueModel
from .self_play import train_using_self_play
import signal
import os
import copy
from mlx.utils import tree_flatten

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
        # --- Updated paths for MLX weights ---
        "value_model_path": "frozen_value_model.safetensors",  # Path for FROZEN value model
        "online_value_model_path": "online_value_model.safetensors",  # Path for ONLINE value model
        # --- Evaluation & Target Update ---
        "target_update_freq": 100,
        "eval_games": 1000,
        "win_rate_threshold": 0.55,
        "stacker_eval_games": 100,
        "minimax_eval_games": 100,  # Number of games vs Minimax
        "minimax_eval_depths": [
            4,
            7,
            10,
        ],  # List of depths for Minimax evaluation
        "force_replace_model": False,  # Controls replacement of frozen value model
    }

    # Initialize the online value model (MLX)
    value_model = ValueModel()

    # Initialize the target value model (MLX)
    target_value_model = ValueModel()
    # Ensure target starts with same structure, weights will be copied/loaded later
    # No need to freeze params explicitly in MLX in the same way as PyTorch;
    # gradients won't be computed for it unless it's part of the grad calculation.

    # Load the weights from the previous run if file exists (loads into ONLINE model)
    loaded_weights = False
    if hyperparams["load_model"]:
        frozen_path = hyperparams["value_model_path"]
        if os.path.exists(frozen_path):
            print(f"Loading online value model from {frozen_path}")
            try:
                # Load weights into the active online model using MLX load_weights
                value_model.load_weights(frozen_path)
                mx.eval(value_model.parameters())  # Ensure weights are loaded
                loaded_weights = True
                print(
                    f"Successfully loaded weights into online model from {frozen_path}"
                )
            except Exception as e:
                print(f"Error loading weights from {frozen_path}: {e}. Starting fresh.")
        else:
            print(
                f"Frozen value model file not found at {frozen_path}, starting fresh."
            )

    # Ensure target network starts with the same weights as the online network
    # (either freshly initialized or loaded)
    target_value_model.update(value_model.parameters())  # Copy parameters using update
    mx.eval(target_value_model.parameters())  # Ensure update
    if loaded_weights:
        print("Target value network initialized with loaded online network weights.")
    else:
        print("Target value network initialized with fresh online network weights.")

    # Initialize wandb
    run = None
    if hyperparams["use_wandb"]:
        run = wandb.init(
            project="connect_four_mlx_value_based",  # Updated project name
            config=hyperparams,
            save_code=True,
        )
        # wandb.watch is typically PyTorch-specific. Monitoring gradients/params might need manual logging.
        print("wandb.watch is not used for MLX models.")
        # wandb.watch(
        #     value_model, # This likely won't work directly with MLX model
        #     log="all",
        #     log_freq=hyperparams["log_interval"] * hyperparams["batch_size"],
        # )

    # Count the number of parameters in the models using MLX properties
    # value_params = value_model.num_params # Incorrect - num_params is not a direct attribute
    # Correct MLX way: flatten the parameter tree and sum the size of leaves
    param_tree = value_model.parameters()
    param_list = tree_flatten(param_tree)
    value_params = sum(
        p.size for _, p in param_list
    )  # Sum sizes of arrays in flattened list

    print(f"Value Model Parameters (MLX): {value_params:,}")
    if hyperparams["use_wandb"] and run:
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
    # TRAINING (Using MLX models and functions)
    ###########################################################################

    try:
        # Train the models using self-play with target networks (MLX Value-Based)
        train_using_self_play(
            value_model=value_model,  # Pass MLX model
            frozen_value_model=target_value_model,  # Pass MLX model
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
            minimax_eval_games=hyperparams["minimax_eval_games"],
            minimax_eval_depths=hyperparams["minimax_eval_depths"],
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
            print("Saving final frozen and online value models (MLX weights)...")
            frozen_path = hyperparams["value_model_path"]
            online_path = hyperparams["online_value_model_path"]

            # Save FROZEN value model locally using MLX save_weights
            try:
                target_value_model.save_weights(frozen_path)
                print(f"Frozen value model saved locally to {frozen_path}")
            except Exception as e:
                print(f"Error saving frozen model locally: {e}")

            # Save ONLINE value model locally using MLX save_weights
            try:
                value_model.save_weights(online_path)
                print(f"Online value model saved locally to {online_path}")
            except Exception as e:
                print(f"Error saving online model locally: {e}")

            # Log models as W&B Artifacts if enabled and files exist
            if hyperparams["use_wandb"] and run:
                print("Logging final models as W&B Artifacts...")
                try:
                    # Create artifact for the frozen model (.safetensors)
                    if os.path.exists(frozen_path):
                        frozen_artifact = wandb.Artifact(
                            "frozen_value_model",  # Keep consistent name
                            type="model",
                            description="Final frozen MLX value model weights (.safetensors)",
                            metadata=hyperparams,
                        )
                        frozen_artifact.add_file(frozen_path)
                        run.log_artifact(frozen_artifact)
                        print(
                            f"Logged '{frozen_artifact.name}' artifact ({frozen_path})."
                        )
                    else:
                        print(
                            f"Skipping frozen model artifact logging: File not found at {frozen_path}"
                        )

                    # Create artifact for the online model (.safetensors)
                    if os.path.exists(online_path):
                        online_artifact = wandb.Artifact(
                            "online_value_model",  # Keep consistent name
                            type="model",
                            description="Final online MLX value model weights (.safetensors)",
                            metadata=hyperparams,
                        )
                        online_artifact.add_file(online_path)
                        run.log_artifact(online_artifact)
                        print(
                            f"Logged '{online_artifact.name}' artifact ({online_path})."
                        )
                    else:
                        print(
                            f"Skipping online model artifact logging: File not found at {online_path}"
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
