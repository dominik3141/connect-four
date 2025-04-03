from model import DecisionModel, ValueModel, get_next_model_move
from engine import ConnectFour, make_move, is_in_terminal_state, is_legal, random_move
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional
from utils import safe_log_to_wandb, create_board_image
import wandb
import numpy as np
import random
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


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
    online_policy_model: DecisionModel,
    online_value_model: ValueModel,
    frozen_policy_model: DecisionModel,
    frozen_value_model: ValueModel,
    discount_factor: float,
    entropy_coefficient: float,
    temperature: float = 1.0,
    epsilon: float = 0,
) -> Tuple[Tensor, Tensor, int, float, ConnectFour, List[Tensor]]:
    """
    Plays a game where the online model plays against a frozen version of itself.
    Losses (policy and value) are calculated *only* for the steps taken by the online player.
    Value estimates for logging are collected for *all* steps using the online model,
    always evaluated from the perspective of the player designated as 'online' for this game.

    Returns the total policy loss (for online player), total value loss (for online player),
    game outcome, average policy entropy (for online player), the final board state,
    and a list of value estimates made by the online model (from the online player's perspective)
    for *every* state encountered.
    """
    board = ConnectFour()
    online_policy_loss_terms: List[Tensor] = []
    online_value_loss_terms: List[Tensor] = []
    online_entropies: List[float] = []
    # This list stores estimates from ALL turns, always from the online player's perspective.
    online_perspective_value_estimates_game: List[Tensor] = []
    final_board = board

    # Randomly assign online/frozen models to players 1 and 2
    online_player_id = random.choice([1, 2])

    current_player = 1

    while True:
        board_tensor = board_to_tensor(final_board)
        is_online_turn = current_player == online_player_id

        # --- Calculate Value Estimate for Logging (Every Turn, Online Perspective) ---
        # Use the ONLINE value model to estimate the value for the ONLINE player,
        # regardless of whose actual turn it is. Detach for logging.
        value_estimate_for_log = online_value_model(
            board_tensor,
            online_player_id,  # <<< Use online_player_id here
        ).detach()
        online_perspective_value_estimates_game.append(value_estimate_for_log)

        # --- Determine Acting Models ---
        acting_policy_model = (
            online_policy_model if is_online_turn else frozen_policy_model
        )

        # --- Value Estimate for Loss Calculation (Only Online Turn, Current Player Perspective) ---
        # This MUST be from the perspective of the current_player when it's the online turn,
        # because the TD target is calculated relative to this player's action.
        online_value_estimate_t: Optional[Tensor] = None
        if is_online_turn:
            # This calculation needs gradients for the backward pass.
            # It correctly uses current_player (which equals online_player_id in this block).
            online_value_estimate_t = online_value_model(board_tensor, current_player)

        # --- Select and Execute Move ---
        move, prob, entropy = get_next_model_move(
            acting_policy_model,
            final_board,
            temperature=temperature,
            epsilon=epsilon,
        )

        next_board = make_move(final_board, current_player, move)
        next_board_tensor = board_to_tensor(next_board)
        status = is_in_terminal_state(next_board)

        # --- Calculate Target Value (TD Target for Online Player's Move) ---
        # Target is only needed if it was the online player's turn.
        td_target: Optional[Tensor] = None
        if is_online_turn and online_value_estimate_t is not None:
            if status != 0:  # Game ended
                # Reward calculation remains the same (outcome for the player who just moved)
                if status == current_player:
                    reward = 1.0
                elif status == 3:
                    reward = 0.0
                else:
                    reward = -1.0
                td_target = torch.tensor(
                    [[reward]], device=online_value_estimate_t.device
                )
            else:  # Game continues
                next_player = 3 - current_player  # Opponent
                with torch.no_grad():
                    frozen_next_value_opp = frozen_value_model(
                        next_board_tensor, next_player
                    )
                # Target V(S_t, P) = gamma * (-V_frozen(S_{t+1}, O)) still holds
                td_target = discount_factor * (-frozen_next_value_opp)

        # --- Calculate Losses (Only if Online Player Moved) ---
        if (
            is_online_turn
            and online_value_estimate_t is not None
            and td_target is not None
        ):
            # Value Loss uses the estimate calculated for the current (online) player
            value_loss_term = F.mse_loss(online_value_estimate_t, td_target)
            online_value_loss_terms.append(value_loss_term)

            # Policy Loss Advantage also uses the estimate for the current (online) player
            advantage = (td_target - online_value_estimate_t).detach()
            epsilon_log = 1e-9
            log_prob = torch.log(prob + epsilon_log)
            policy_loss_term = -log_prob * advantage - entropy_coefficient * entropy
            online_policy_loss_terms.append(policy_loss_term)

            online_entropies.append(entropy.item())

        # --- Update State and Player ---
        final_board = next_board
        if status != 0:
            break
        current_player = 3 - current_player

    # --- Sum Losses ---
    total_policy_loss = (
        torch.stack(online_policy_loss_terms).sum()
        if online_policy_loss_terms
        else torch.tensor(0.0)
    )
    total_value_loss = (
        torch.stack(online_value_loss_terms).sum()
        if online_value_loss_terms
        else torch.tensor(0.0)
    )

    # --- Calculate Avg Entropy ---
    avg_online_entropy = (
        sum(online_entropies) / len(online_entropies) if online_entropies else 0.0
    )

    # Return the list containing estimates from the online player's perspective for all states
    return (
        total_policy_loss,
        total_value_loss,
        status,
        avg_online_entropy,
        final_board,
        online_perspective_value_estimates_game,  # <<< Return the new list
    )


def evaluate_models(
    online_model: DecisionModel,
    frozen_model: DecisionModel,
    num_games: int = 20,
) -> float:
    """
    Plays games between online and frozen models and returns the win rate of the online model.
    Uses greedy move selection (no exploration).
    """
    online_wins = 0
    frozen_wins = 0
    draws = 0

    print(f"Starting evaluation: {num_games} games...")

    for i in range(num_games):
        board = ConnectFour()
        current_player = 1
        # Assign models to players - alternate who starts
        model_p1 = online_model if i % 2 == 0 else frozen_model
        model_p2 = frozen_model if i % 2 == 0 else online_model

        while True:
            active_model = model_p1 if current_player == 1 else model_p2

            # Get greedy move (temperature=0 or very small, epsilon=0)
            move, _, _ = get_next_model_move(
                active_model,
                board,
                temperature=0.01,
                epsilon=0,
            )

            board = make_move(board, current_player, move)
            status = is_in_terminal_state(board)

            if status != 0:
                # Determine winner based on who was P1/P2 in this specific game
                if status == 1:  # Player 1 won
                    if model_p1 == online_model:
                        online_wins += 1
                    else:
                        frozen_wins += 1
                elif status == 2:  # Player 2 won
                    if model_p2 == online_model:
                        online_wins += 1
                    else:
                        frozen_wins += 1
                elif status == 3:  # Draw
                    draws += 1
                break

            current_player = 3 - current_player

    total_played = online_wins + frozen_wins + draws
    if total_played == 0:
        return 0.0

    win_rate = online_wins / total_played

    print(
        f"Evaluation Results: Online Wins: {online_wins}, Frozen Wins: {frozen_wins}, Draws: {draws}"
    )
    print(f"Online Model Win Rate vs Frozen: {win_rate:.2%}")
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


# Modified to handle frozen networks, updates, evaluation, and entropy coefficient
def train_using_self_play(
    policy_model: DecisionModel,
    value_model: ValueModel,
    frozen_policy_model: DecisionModel,
    frozen_value_model: ValueModel,
    iterations: int = 1000,
    batch_size: int = 32,
    log_interval: int = 10,
    learning_rate: float = 0.001,
    temperature: float = 1.0,
    epsilon: float = 0.1,
    discount_factor: float = 0.95,
    entropy_coefficient: float = 0.01,
    target_update_freq: int = 10,
    eval_games: int = 20,
    win_rate_threshold: float = 0.55,
    stacker_eval_games: int = 20,
) -> None:
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=learning_rate)

    frozen_policy_model.eval()
    frozen_value_model.eval()

    running_value_loss = 0.0
    running_policy_loss = 0.0
    running_entropy = 0.0
    # This list collects estimates from the online player's perspective for all turns.
    running_online_perspective_values: List[float] = []
    total_games_played = 0
    last_final_board_state: Optional[np.ndarray] = None

    print(f"Starting training for {iterations} batches of size {batch_size}...")
    print(f"Frozen networks evaluated every {target_update_freq} batches.")
    print(f"Frozen network updated if online model win rate > {win_rate_threshold:.1%}")
    print(f"Using entropy regularization coefficient: {entropy_coefficient}")
    print(f"Stacker evaluation uses {stacker_eval_games} games.")

    for i in range(iterations):
        policy_model.train()
        value_model.train()

        batch_policy_loss = torch.tensor(0.0, requires_grad=True)
        batch_value_loss = torch.tensor(0.0, requires_grad=True)
        batch_entropy_sum = 0.0
        batch_games = 0
        # Store online-perspective values from the last game for the plot
        batch_last_game_online_perspective_values: Optional[List[float]] = None

        # --- Play Batch (Online vs Frozen) ---
        for game_idx in range(batch_size):
            (
                total_policy_loss,
                total_value_loss,
                status,
                avg_entropy,
                final_board,
                # Receive the list containing online-perspective estimates from all turns
                online_perspective_estimates,
            ) = play_against_self(
                online_policy_model=policy_model,
                online_value_model=value_model,
                frozen_policy_model=frozen_policy_model,
                frozen_value_model=frozen_value_model,
                discount_factor=discount_factor,
                entropy_coefficient=entropy_coefficient,
                temperature=temperature,
                epsilon=epsilon,
            )

            batch_policy_loss = batch_policy_loss + total_policy_loss
            batch_value_loss = batch_value_loss + total_value_loss
            batch_entropy_sum += avg_entropy
            batch_games += 1
            total_games_played += 1

            # Collect online-perspective estimates from all turns
            game_online_perspective_values = [
                v.item() for v in online_perspective_estimates
            ]
            running_online_perspective_values.extend(game_online_perspective_values)

            if game_idx == batch_size - 1:
                last_final_board_state = final_board.state.copy()
                # Store values from the last game for the plot
                batch_last_game_online_perspective_values = (
                    game_online_perspective_values
                )

        if batch_games == 0:
            print(f"Warning: Batch {i + 1} completed with 0 games.")
            continue

        # --- Update Online Models ---
        avg_batch_policy_loss = batch_policy_loss / batch_games
        avg_batch_value_loss = batch_value_loss / batch_games

        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()

        if avg_batch_policy_loss.requires_grad:
            avg_batch_policy_loss.backward(retain_graph=False)
        else:
            if avg_batch_policy_loss.item() != 0.0:
                print(
                    f"Warning: Batch {i + 1} Policy loss is non-zero but has no grad."
                )

        if avg_batch_value_loss.requires_grad:
            avg_batch_value_loss.backward()
        else:
            if avg_batch_value_loss.item() != 0.0:
                print(f"Warning: Batch {i + 1} Value loss is non-zero but has no grad.")

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)

        optimizer_policy.step()
        optimizer_value.step()

        # --- Evaluate and Potentially Update Frozen Networks ---
        if (i + 1) % target_update_freq == 0:
            policy_model.eval()
            online_win_rate = evaluate_models(
                policy_model, frozen_policy_model, num_games=eval_games
            )
            safe_log_to_wandb(
                {"evaluation/online_vs_frozen_win_rate": online_win_rate}, step=i + 1
            )

            stacker_win_rate, last_stacker_game_board = evaluate_vs_stacker(
                policy_model,
                num_games=stacker_eval_games,
            )
            stacker_log_data = {
                "evaluation/online_vs_stacker_win_rate": stacker_win_rate
            }
            if last_stacker_game_board is not None and wandb.run is not None:
                stacker_board_image_np = create_board_image(last_stacker_game_board)
                stacker_log_data["evaluation/final_stacker_eval_board"] = wandb.Image(
                    stacker_board_image_np,
                    caption=f"Final Stacker Eval Board - Batch {i + 1}",
                )
            safe_log_to_wandb(stacker_log_data, step=i + 1)

            if online_win_rate > win_rate_threshold:
                print(
                    f"Online model passed evaluation (Win Rate vs Frozen: {online_win_rate:.2%}). Updating frozen networks."
                )
                frozen_policy_model.load_state_dict(policy_model.state_dict())
                frozen_value_model.load_state_dict(value_model.state_dict())
                safe_log_to_wandb({"evaluation/frozen_network_updated": 1}, step=i + 1)
            else:
                print(
                    f"Online model did not pass evaluation (Win Rate vs Frozen: {online_win_rate:.2%}). Frozen networks remain unchanged."
                )
                safe_log_to_wandb({"evaluation/frozen_network_updated": 0}, step=i + 1)

        # --- Logging ---
        running_policy_loss += avg_batch_policy_loss.item()
        running_value_loss += avg_batch_value_loss.item()
        running_entropy += batch_entropy_sum / batch_games

        if (i + 1) % log_interval == 0:
            avg_policy_loss = running_policy_loss / log_interval
            avg_value_loss = running_value_loss / log_interval
            avg_entropy_log = running_entropy / log_interval

            value_std_dev = 0.0
            value_histogram = None
            # Use the list containing online-perspective estimates for histogram and std dev
            if running_online_perspective_values:
                value_std_dev = (
                    torch.tensor(running_online_perspective_values).std().item()
                )
                if wandb.run is not None:
                    # Histogram reflects online player's perspective on ALL states visited
                    value_histogram = wandb.Histogram(running_online_perspective_values)

            print(f"\nBatch {i + 1}/{iterations} (Total Games: {total_games_played})")
            print(
                f"  Avg Value Loss (Online Moves, last {log_interval} batches): {avg_value_loss:.4f}"
            )  # Clarified Value Loss source
            print(
                f"  Avg Policy Loss (Online Moves, last {log_interval} batches): {avg_policy_loss:.4f}"
            )  # Clarified Policy Loss source
            print(
                f"  Avg Policy Entropy (Online Moves, last {log_interval} batches): {avg_entropy_log:.4f}"
            )  # Clarified Entropy source
            # Update print statement description
            print(
                f"  Std Dev Value Pred (Online Perspective, All Turns, last {log_interval} batches): {value_std_dev:.4f}"
            )

            log_data = {
                "training/batch_value_loss": avg_value_loss,  # Loss calculated for online player's moves
                "training/batch_policy_loss": avg_policy_loss,  # Loss calculated for online player's moves
                "training/batch_policy_entropy": avg_entropy_log,  # Entropy calculated for online player's moves
                # Update wandb key description
                "training/online_perspective_value_std_dev": value_std_dev,
                "training/total_games_played": total_games_played,
            }

            if value_histogram:
                # Update wandb key description
                log_data["training/online_perspective_value_distribution"] = (
                    value_histogram
                )

            # Use the list containing online-perspective values from the last game for the plot
            if batch_last_game_online_perspective_values and wandb.run is not None:
                if len(batch_last_game_online_perspective_values) > 1:
                    fig = None
                    try:
                        turns = list(
                            range(len(batch_last_game_online_perspective_values))
                        )
                        values = batch_last_game_online_perspective_values

                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(turns, values, marker="o", linestyle="-")
                        ax.set_xlabel("Game Step Index")
                        # Update axis label
                        ax.set_ylabel("Predicted Value (Online Player Perspective)")
                        # Update title
                        ax.set_title(
                            f"Last Game Value Progression (Online Persp.) - Batch {i + 1}"
                        )
                        ax.set_ylim(-1.05, 1.05)
                        ax.grid(True)

                        # Update wandb key description
                        log_data["training/last_game_value_progression_plot"] = fig
                        plt.close(fig)

                    except Exception as e:
                        print(f"Warning: Failed to create value progression plot: {e}")
                        if fig:
                            plt.close(fig)

            # ... (logging final board remains the same) ...
            if last_final_board_state is not None and wandb.run is not None:
                board_image_np = create_board_image(last_final_board_state)
                log_data["training/final_online_vs_frozen_board"] = wandb.Image(
                    board_image_np,
                    caption=f"Final Online vs Frozen Board - Batch {i + 1}",
                )
                last_final_board_state = None  # Clear it after logging

            safe_log_to_wandb(log_data, step=i + 1)

            # Reset running metrics for the next interval
            running_value_loss = 0.0
            running_policy_loss = 0.0
            running_entropy = 0.0
            # Reset the list collecting online-perspective predictions
            running_online_perspective_values = []
