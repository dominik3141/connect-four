import argparse
import torch
import os
from src.model import ValueModel
from src.self_play import evaluate_models  # Assuming evaluate_models is in self_play
from typing import Optional


def load_model_weights(
    model: ValueModel, path: Optional[str], device: torch.device
) -> bool:
    """Loads model weights from a file if the path is valid."""
    if path and os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Successfully loaded model weights from {path}")
            return True
        except Exception as e:
            print(f"Error loading model weights from {path}: {e}")
            return False
    elif path:
        print(f"Model weights file not found at {path}. Using initialized model.")
        return False
    else:
        print("No path provided for model. Using initialized model.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two Connect Four Value Models against each other."
    )
    parser.add_argument(
        "--model1_path",
        type=str,
        required=True,
        help="Path to the weights file for the first model (considered 'online' for win rate calculation).",
    )
    parser.add_argument(
        "--model2_path",
        type=str,
        required=True,
        help="Path to the weights file for the second model (considered 'frozen' for win rate calculation).",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games to play for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on ('cpu' or 'cuda').",
    )

    args = parser.parse_args()

    # --- Device Setup ---
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Model Initialization ---
    model1 = ValueModel().to(device)
    model2 = ValueModel().to(device)

    # --- Load Weights ---
    loaded1 = load_model_weights(model1, args.model1_path, device)
    loaded2 = load_model_weights(model2, args.model2_path, device)

    if not loaded1 and not loaded2:
        print("Error: Failed to load weights for both models. Cannot proceed.")
        return
    elif not loaded1:
        print(
            "Warning: Failed to load weights for Model 1. It will use initial weights."
        )
    elif not loaded2:
        print(
            "Warning: Failed to load weights for Model 2. It will use initial weights."
        )

    # --- Set to Evaluation Mode ---
    model1.eval()
    model2.eval()

    # --- Run Evaluation ---
    # evaluate_models calculates the win rate of the *first* model passed (model1)
    # against the *second* model passed (model2).
    print(
        f"\nEvaluating Model 1 ({args.model1_path}) vs Model 2 ({args.model2_path}) for {args.num_games} games..."
    )
    win_rate_model1 = evaluate_models(
        online_value_model=model1,
        frozen_value_model=model2,
        num_games=args.num_games,
    )

    print("\n--- Evaluation Summary ---")
    print(f"Model 1 Path: {args.model1_path}")
    print(f"Model 2 Path: {args.model2_path}")
    print(f"Number of Games: {args.num_games}")
    print(f"Win Rate of Model 1 vs Model 2: {win_rate_model1:.2%}")
    print("------------------------")


if __name__ == "__main__":
    main()
