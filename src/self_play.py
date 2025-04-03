from model import DecisionModel, ValueModel, get_next_model_move
from engine import ConnectFour, make_move, is_in_terminal_state, is_legal, random_move
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional
from utils import safe_log_to_wandb, create_board_image
import wandb
import numpy as np
import math  # Keep math import if needed by other parts or future additions
import random


def board_to_tensor(board: ConnectFour) -> Tensor:
    """Convert a ConnectFour board to a tensor representation consistent with model expectations."""
    # Convert numpy array to tensor and transpose to match our expected format
    # The state is 6x7, we want 7x6
    tensor = torch.from_numpy(board.state).float()
    tensor = tensor.t()  # transpose to get 7x6

    # Player 1 is 1.0, Player 2 is 2.0, Empty is 0.0
    # No conversion needed as models expect this representation.

    # Make the tensor contiguous in memory
    return tensor.contiguous()


def play_against_self(
    policy_model: DecisionModel,
    value_model: ValueModel,
    target_value_model: ValueModel,  # Added target value model
    discount_factor: float,
    entropy_coefficient: float,  # Added entropy coefficient
    temperature: float = 1.0,
    epsilon: float = 0,
) -> Tuple[Tensor, Tensor, int, float, ConnectFour]:
    """
    Play a game where the model plays against itself using TD learning.
    Uses the online value_model for V(S_t) and the target_value_model for V(S_t+1).
    Includes entropy regularization in the policy loss.
    Returns the total policy loss, total value loss, game outcome, average policy entropy,
    and the final board state.
    """
    board = ConnectFour()
    # Use lists to accumulate tensors for loss terms and scalars for entropy
    policy_loss_terms: List[Tensor] = []
    value_loss_terms: List[Tensor] = []
    entropies: List[float] = []
    current_player = 1
    final_board = board

    while True:
        board_tensor = board_to_tensor(final_board)
        # Ensure requires_grad is true for value model input if needed downstream
        # board_tensor.requires_grad_(True) # Usually not needed if model params require grad

        # Value estimate BEFORE the move - Use ONLINE model
        old_value: Tensor = value_model(board_tensor, current_player)

        # Get move from ONLINE policy model (still uses exploration)
        move, prob, entropy = get_next_model_move(
            policy_model, final_board, temperature=temperature, epsilon=epsilon
        )
        entropies.append(
            entropy.item()
        )  # Store raw entropy for logging/avg calculation

        # Execute the move
        next_board = make_move(final_board, current_player, move)
        next_board_tensor = board_to_tensor(next_board)
        status = is_in_terminal_state(next_board)

        # Ensure next_value is always [1, 1] and on the correct device
        next_value = torch.zeros_like(old_value)  # Estimate of V(S_t+1, O)

        td_target = torch.zeros_like(old_value)  # Initialize target

        if status != 0:  # Game ended after this move
            # Player who just moved is current_player (P)
            if status == current_player:  # P Won
                td_target = torch.tensor([[1.0]], device=old_value.device)
            elif status == 3:  # Draw
                td_target = torch.tensor([[0.5]], device=old_value.device)
            else:  # P Lost (Opponent O Won)
                td_target = torch.tensor([[0.0]], device=old_value.device)
            # next_value remains [[0.0]] as game is over, not needed for target here
        else:
            # Game continues, estimate opponent's value in the next state V(S_t+1, O)
            next_player = 3 - current_player  # Opponent (O)
            # Estimate V(S_t+1, O) using the TARGET value network
            with torch.no_grad():  # Important: No gradients through target net
                next_value = target_value_model(next_board_tensor, next_player)

            # Target for V(S_t, P) is gamma * (1 - V_target(S_t+1, O))
            # Note: Reward R_t+1 is 0 here.
            td_target = discount_factor * (1.0 - next_value)

        # Calculate Value Loss term (e.g., MSE loss)
        # Loss pushes ONLINE value (old_value) towards the correctly calculated td_target
        value_loss_term = F.mse_loss(old_value, td_target)  # [1, 1] vs [1, 1]

        # Calculate TD Error (Advantage) for the Policy Model (Actor)
        # Advantage = Target - V(S_t) = Target - old_value
        # Use the same, correctly calculated td_target based on TARGET value net
        advantage = (td_target - old_value).detach()  # Detach: Critic guides Actor

        # Calculate Policy Loss term: -log_prob * Advantage - entropy_coeff * entropy
        epsilon_log = 1e-9
        # Ensure prob has grad_fn if needed downstream, otherwise may need recompute/clone
        # policy_loss_term = -torch.log(prob.clone() + epsilon_log) * advantage # Example if needed
        log_prob = torch.log(prob + epsilon_log)
        policy_loss_term = (
            -log_prob * advantage - entropy_coefficient * entropy
        )  # Add entropy bonus

        # Append loss tensors to lists
        policy_loss_terms.append(policy_loss_term)
        value_loss_terms.append(value_loss_term)

        final_board = next_board

        if status != 0:
            break  # Game ended

        # Prepare for next iteration
        current_player = 3 - current_player
        # old_value becomes the value estimate calculated in the next loop's start

    # Sum the losses from the lists
    total_policy_loss = (
        torch.stack(policy_loss_terms).sum() if policy_loss_terms else torch.tensor(0.0)
    )
    total_value_loss = (
        torch.stack(value_loss_terms).sum() if value_loss_terms else torch.tensor(0.0)
    )

    # Calculate average entropy for the game
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

    # Return final status, avg entropy, and the final board object
    return total_policy_loss, total_value_loss, status, avg_entropy, final_board


# Placeholder for evaluation function - Step 2
def evaluate_models(
    online_model: DecisionModel,
    target_model: DecisionModel,
    num_games: int = 20,  # Number of games for evaluation
) -> float:
    """
    Plays games between online and target models and returns the win rate of the online model.
    Uses greedy move selection (no exploration).
    """
    online_wins = 0
    target_wins = 0
    draws = 0

    print(f"Starting evaluation: {num_games} games...")

    for i in range(num_games):
        board = ConnectFour()
        current_player = 1
        # Assign models to players - alternate who starts
        model_p1 = online_model if i % 2 == 0 else target_model
        model_p2 = target_model if i % 2 == 0 else online_model

        while True:
            active_model = model_p1 if current_player == 1 else model_p2

            # Get greedy move (temperature=0 or very small, epsilon=0)
            move, _, _ = get_next_model_move(
                active_model,
                board,
                temperature=0.01,
                epsilon=0,  # Effectively greedy
            )

            board = make_move(board, current_player, move)
            status = is_in_terminal_state(board)

            if status != 0:
                # Determine winner based on who was P1/P2 in this specific game
                if status == 1:  # Player 1 won
                    if model_p1 == online_model:
                        online_wins += 1
                    else:
                        target_wins += 1
                elif status == 2:  # Player 2 won
                    if model_p2 == online_model:
                        online_wins += 1
                    else:
                        target_wins += 1
                elif status == 3:  # Draw
                    draws += 1
                break

            current_player = 3 - current_player

    total_played = online_wins + target_wins + draws
    if total_played == 0:
        return 0.0

    # Calculate win rate for the online model (excluding draws often)
    # win_rate = online_wins / (online_wins + target_wins) if (online_wins + target_wins) > 0 else 0.0
    # Let's use win rate including draws for now: online_wins / total_played
    win_rate = online_wins / total_played

    print(
        f"Evaluation Results: Online Wins: {online_wins}, Target Wins: {target_wins}, Draws: {draws}"
    )
    print(f"Online Model Win Rate: {win_rate:.2%}")
    return win_rate


def evaluate_vs_stacker(
    model: DecisionModel,
    num_games: int = 10,
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Evaluates the model's ability to defend against a simple "stacker" strategy.

    Plays games where one player ("Stacker") attempts to stack in a randomly chosen
    column for that game. The model plays the other player using greedy moves.

    Args:
        model: The DecisionModel (policy network) to evaluate.
        num_games: The number of games to play.

    Returns:
        A tuple containing:
        - The win rate of the model against the Stacker bot.
        - The final board state (numpy array) of the last game played, or None if num_games is 0.
    """
    model_wins = 0
    stacker_wins = 0
    draws = 0
    last_game_final_board_state: Optional[np.ndarray] = None

    if num_games <= 0:
        return 0.0, None

    print(
        f"Starting stacker evaluation: {num_games} games against Stacker targeting a random column each game..."
    )

    for i in range(num_games):
        board = ConnectFour()
        # --- Choose random target column for this game ---
        stacker_target_col_this_game = random.randint(0, 6)
        # -------------------------------------------------

        # Alternate who starts: Model (Player 1) or Stacker (Player 1)
        model_player = 1 if i % 2 == 0 else 2
        stacker_player = 3 - model_player
        current_player = 1
        game_board_state: Optional[np.ndarray] = (
            None  # To store the state before loop ends
        )

        while True:
            game_board_state = board.state.copy()  # Keep track of the current state

            if current_player == model_player:
                # Model's turn: Get greedy move
                move, _, _ = get_next_model_move(
                    model,
                    board,
                    temperature=0.01,  # Near-greedy
                    epsilon=0,
                )
            else:
                # Stacker's turn
                if is_legal(
                    board, stacker_target_col_this_game
                ):  # Use game-specific target column
                    move = stacker_target_col_this_game
                else:
                    # Fallback: If target column is full, play randomly
                    try:
                        move = random_move(board)
                    except ValueError:
                        # This happens if the board is full (stalemate)
                        status = 3  # Treat as stalemate immediately
                        break

            # Execute the move
            try:
                board = make_move(board, current_player, move)
            except ValueError as e:
                # Should be less likely now with the random_move check above
                print(f"Warning: Error making move during stacker eval: {e}")
                status = 3  # Assume draw if unexpected error
                break

            status = is_in_terminal_state(board)

            if status != 0:
                game_board_state = board.state.copy()  # Capture final state
                if status == model_player:
                    model_wins += 1
                elif status == stacker_player:
                    stacker_wins += 1
                elif status == 3:
                    draws += 1
                break

            current_player = 3 - current_player

        # Store the final board state of the last game
        if i == num_games - 1 and game_board_state is not None:
            last_game_final_board_state = game_board_state

    total_played = model_wins + stacker_wins + draws
    model_win_rate = model_wins / total_played if total_played > 0 else 0.0

    print(
        f"Stacker Evaluation Results: Model Wins: {model_wins}, Stacker Wins: {stacker_wins}, Draws: {draws}"
    )
    print(f"Model Win Rate vs Random Stacker: {model_win_rate:.2%}")
    return model_win_rate, last_game_final_board_state


# Modified to handle target networks, updates, evaluation, and entropy coefficient
def train_using_self_play(
    policy_model: DecisionModel,
    value_model: ValueModel,
    target_policy_model: DecisionModel,  # Added target models
    target_value_model: ValueModel,
    iterations: int = 1000,
    batch_size: int = 32,
    log_interval: int = 10,
    learning_rate: float = 0.001,
    temperature: float = 1.0,
    epsilon: float = 0.1,
    discount_factor: float = 0.95,
    entropy_coefficient: float = 0.01,  # Added entropy coefficient
    target_update_freq: int = 10,  # How often to potentially update target
    eval_games: int = 20,  # Games per evaluation
    win_rate_threshold: float = 0.55,  # Online must win >55% to update target
    stacker_eval_games: int = 20,  # <<< Add new arg
) -> None:
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=learning_rate)

    # Ensure target models are in eval mode (important if using dropout/batchnorm)
    target_policy_model.eval()
    target_value_model.eval()

    # Running metrics, reset every log_interval batches
    running_value_loss = 0.0
    running_policy_loss = 0.0
    running_entropy = 0.0
    total_games_played = 0
    last_final_board_state: Optional[np.ndarray] = None

    print(f"Starting training for {iterations} batches of size {batch_size}...")
    print(f"Target networks evaluated every {target_update_freq} batches.")
    print(f"Target network updated if online model win rate > {win_rate_threshold:.1%}")
    print(f"Using entropy regularization coefficient: {entropy_coefficient}")
    print(f"Stacker evaluation uses {stacker_eval_games} games.")

    for i in range(iterations):
        # Set online models to train mode
        policy_model.train()
        value_model.train()

        # Accumulators for the current batch
        batch_policy_loss = torch.tensor(0.0, requires_grad=True)
        batch_value_loss = torch.tensor(0.0, requires_grad=True)
        batch_entropy_sum = 0.0
        batch_games = 0

        # --- Play Batch (Self-Play for Training Data) ---
        for game_idx in range(batch_size):
            # Pass the target_value_model and entropy_coefficient
            total_policy_loss, total_value_loss, status, avg_entropy, final_board = (
                play_against_self(
                    policy_model,  # Online policy generates moves
                    value_model,  # Online value estimates V(S_t)
                    target_value_model,  # Target value estimates V(S_t+1)
                    discount_factor=discount_factor,
                    entropy_coefficient=entropy_coefficient,  # Pass coefficient
                    temperature=temperature,
                    epsilon=epsilon,
                )
            )

            # Accumulate losses for the batch
            batch_policy_loss = batch_policy_loss + total_policy_loss
            batch_value_loss = batch_value_loss + total_value_loss
            batch_entropy_sum += avg_entropy
            batch_games += 1
            total_games_played += 1

            # Store the state of the very last board played in the batch
            if game_idx == batch_size - 1:
                last_final_board_state = final_board.state.copy()

        if batch_games == 0:
            print(f"Warning: Batch {i + 1} completed with 0 games.")
            continue  # Skip update if no games were played

        # --- Update Online Models ---
        avg_batch_policy_loss = batch_policy_loss / batch_games
        avg_batch_value_loss = batch_value_loss / batch_games

        # Zero gradients before backpropagation for the batch
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()

        # Backpropagate average losses
        if avg_batch_policy_loss.requires_grad:
            # Retain graph if value loss needs it, otherwise set to False is safer
            avg_batch_policy_loss.backward(
                retain_graph=False
            )  # Check if retain_graph is needed
        else:
            print(f"Warning: Batch {i + 1} Policy loss has no grad.")

        if avg_batch_value_loss.requires_grad:
            avg_batch_value_loss.backward()
        else:
            print(f"Warning: Batch {i + 1} Value loss has no grad.")

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)

        # Step optimizers
        optimizer_policy.step()
        optimizer_value.step()

        # --- Evaluate and Potentially Update Target Networks ---
        if (i + 1) % target_update_freq == 0:
            # Put online model in eval mode for fair comparison
            policy_model.eval()
            # Evaluate online policy against target policy
            online_win_rate = evaluate_models(
                policy_model, target_policy_model, num_games=eval_games
            )
            # Log the win rate regardless of update
            safe_log_to_wandb(
                {"evaluation/online_vs_target_win_rate": online_win_rate}, step=i + 1
            )

            # --- Evaluate online policy against Stacker bot --- # <<< NEW EVALUATION CALL
            stacker_win_rate, last_stacker_game_board = evaluate_vs_stacker(
                policy_model,
                num_games=stacker_eval_games,
            )
            stacker_log_data = {
                "evaluation/online_vs_stacker_win_rate": stacker_win_rate
            }
            # Log the final board image from the stacker evaluation
            if last_stacker_game_board is not None and wandb.run is not None:
                stacker_board_image_np = create_board_image(last_stacker_game_board)
                stacker_log_data["evaluation/final_stacker_eval_board"] = wandb.Image(
                    stacker_board_image_np,
                    caption=f"Final Stacker Eval Board - Batch {i + 1}",
                )
            safe_log_to_wandb(stacker_log_data, step=i + 1)
            # --- End Stacker Evaluation --- #

            if online_win_rate > win_rate_threshold:
                print(
                    f"Online model passed evaluation (Win Rate vs Target: {online_win_rate:.2%}). Updating target networks."
                )
                target_policy_model.load_state_dict(policy_model.state_dict())
                target_value_model.load_state_dict(value_model.state_dict())
                # Log 1 for successful update
                safe_log_to_wandb({"evaluation/target_network_updated": 1}, step=i + 1)
            else:
                print(
                    f"Online model did not pass evaluation (Win Rate vs Target: {online_win_rate:.2%}). Target networks remain unchanged."
                )
                # Log 0 for no update
                safe_log_to_wandb({"evaluation/target_network_updated": 0}, step=i + 1)

        # --- Logging ---
        running_policy_loss += avg_batch_policy_loss.item()
        running_value_loss += avg_batch_value_loss.item()
        running_entropy += batch_entropy_sum / batch_games

        if (i + 1) % log_interval == 0:
            avg_policy_loss = running_policy_loss / log_interval
            avg_value_loss = running_value_loss / log_interval
            avg_entropy_log = running_entropy / log_interval

            print(f"\nBatch {i + 1}/{iterations} (Total Games: {total_games_played})")
            print(
                f"  Avg Value Loss (last {log_interval} batches): {avg_value_loss:.4f}"
            )
            print(
                f"  Avg Policy Loss (last {log_interval} batches): {avg_policy_loss:.4f}"
            )
            print(
                f"  Avg Policy Entropy (last {log_interval} batches): {avg_entropy_log:.4f}"
            )

            # Prepare log data dictionary
            log_data = {
                "training/batch_value_loss": avg_value_loss,  # Added training/ prefix
                "training/batch_policy_loss": avg_policy_loss,  # Added training/ prefix
                "training/batch_policy_entropy": avg_entropy_log,  # Added training/ prefix
                "training/total_games_played": total_games_played,  # Added training/ prefix
            }

            # Log the image of the last self-play board state from the batch
            if last_final_board_state is not None and wandb.run is not None:
                board_image_np = create_board_image(last_final_board_state)
                log_data["training/final_selfplay_board"] = (
                    wandb.Image(  # Added training/ prefix
                        board_image_np, caption=f"Final Self-Play Board - Batch {i + 1}"
                    )
                )
                last_final_board_state = None

            safe_log_to_wandb(log_data, step=i + 1)

            # Reset running metrics for the next interval
            running_value_loss = 0.0
            running_policy_loss = 0.0
            running_entropy = 0.0


def _score(status: int, player: int) -> int:
    """
    Calculate the score of a game.
    """
    if status == player:  # win
        return 1
    elif status == 3:  # stalemate
        return 0.5
    elif status == 0:  # game has not ended
        raise ValueError("Game has not ended")
    else:
        return 0
