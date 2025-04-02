from model import DecisionModel, ValueModel, get_next_model_move
from engine import ConnectFour, make_move, is_in_terminal_state
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional
from utils import safe_log_to_wandb, create_board_image
import wandb
import numpy as np


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
    discount_factor: float,
    temperature: float = 1.0,
    epsilon: float = 0,
) -> Tuple[Tensor, Tensor, int, float, ConnectFour]:
    """
    Play a game where the model plays against itself using TD learning for both actor and critic.
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

        # Value estimate BEFORE the move
        old_value: Tensor = value_model(board_tensor, current_player)

        # Get move, probability, and entropy
        move, prob, entropy = get_next_model_move(
            policy_model, final_board, temperature=temperature, epsilon=epsilon
        )
        entropies.append(entropy.item())

        # Execute the move
        next_board = make_move(final_board, current_player, move)
        next_board_tensor = board_to_tensor(next_board)
        status = is_in_terminal_state(next_board)

        # Determine reward and next state value, ensuring correct shape [1, 1]
        reward_val = 0.0
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
            # next_value estimates V(S_t+1, O)
            next_value = value_model(next_board_tensor, next_player).detach()
            # Target for V(S_t, P) is gamma * (1 - V(S_t+1, O))
            # Note: Reward R_t+1 is 0 here.
            td_target = discount_factor * (1.0 - next_value)

        # Calculate Value Loss term (e.g., MSE loss)
        # Loss pushes old_value (V(S_t, P)) towards the correctly calculated td_target
        value_loss_term = F.mse_loss(old_value, td_target)  # [1, 1] vs [1, 1]

        # Calculate TD Error (Advantage) for the Policy Model (Actor)
        # Advantage = Target - V(S_t)
        # Use the same, correctly calculated td_target
        advantage = (td_target - old_value).detach()  # Detach: Critic guides Actor

        # Calculate Policy Loss term: -log_prob * Advantage
        epsilon_log = 1e-9
        policy_loss_term = -torch.log(prob + epsilon_log) * advantage

        # Append loss tensors to lists
        policy_loss_terms.append(policy_loss_term)
        value_loss_terms.append(value_loss_term)

        final_board = next_board

        if status != 0:
            break  # Game ended

        # Prepare for next iteration
        current_player = 3 - current_player
        # old_value becomes the value estimate calculated in the next loop's start

    # Return the accumulated losses for the entire game
    # The status is from the perspective of the *board*, not the player who just moved
    # final_status = is_in_terminal_state(final_board) # Use status from loop end

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


def train_using_self_play(
    policy_model: DecisionModel,
    value_model: ValueModel,
    iterations: int = 1000,  # Now refers to batches
    batch_size: int = 32,  # Number of games per batch
    log_interval: int = 10,  # Log every N batches
    learning_rate: float = 0.001,  # Adjusted LR might be needed
    temperature: float = 1.0,
    epsilon: float = 0.1,  # Slightly higher default epsilon?
    discount_factor: float = 0.95,
) -> None:
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=learning_rate)

    # Running metrics, reset every log_interval batches
    running_value_loss = 0.0
    running_policy_loss = 0.0
    running_entropy = 0.0
    total_games_played = 0
    last_final_board_state: Optional[np.ndarray] = None

    print(f"Starting training for {iterations} batches of size {batch_size}...")

    for i in range(iterations):
        # Accumulators for the current batch
        batch_policy_loss = torch.tensor(0.0, requires_grad=True)
        batch_value_loss = torch.tensor(0.0, requires_grad=True)
        batch_entropy_sum = 0.0
        batch_games = 0

        # --- Play Batch ---
        for game_idx in range(batch_size):
            total_policy_loss, total_value_loss, status, avg_entropy, final_board = (
                play_against_self(
                    policy_model,
                    value_model,
                    discount_factor=discount_factor,
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

        # --- Update Models ---
        avg_batch_policy_loss = batch_policy_loss / batch_games
        avg_batch_value_loss = batch_value_loss / batch_games

        # Zero gradients before backpropagation for the batch
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()

        # Backpropagate average losses
        if avg_batch_policy_loss.requires_grad:
            avg_batch_policy_loss.backward()
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
                "batch_value_loss": avg_value_loss,
                "batch_policy_loss": avg_policy_loss,
                "batch_policy_entropy": avg_entropy_log,
                "total_games_played": total_games_played,
            }

            # Log the image of the last board state from the batch
            if last_final_board_state is not None and wandb.run is not None:
                board_image_np = create_board_image(last_final_board_state)
                log_data["final_board_state"] = wandb.Image(
                    board_image_np, caption=f"Final Board - Batch {i + 1}"
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
