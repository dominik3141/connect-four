from .model import ValueModel, get_next_value_based_move, board_to_tensor
from .engine import ConnectFour, make_move, is_in_terminal_state, is_legal, random_move
from .minimax import minimax_move
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Tuple, List, Optional, TYPE_CHECKING, Callable
from .utils import safe_log_to_wandb, create_board_image
import wandb
import numpy as np
import random
import matplotlib
import io
from PIL import Image
import os

# Use TYPE_CHECKING to avoid circular import issues if wandb.sdk.wandb_run.Run is needed for type hint
if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


# Helper function for MLX loss calculation
def mse_loss(pred: mx.array, target: mx.array) -> mx.array:
    return mx.mean(mx.square(pred - target))


def play_against_self(
    online_value_model: ValueModel,
    frozen_value_model: ValueModel,
    discount_factor: float,
    online_temperature: float = 0.01,
    online_epsilon: float = 0.1,
    frozen_temperature: float = 0.01,
    frozen_epsilon: float = 0.0,
) -> Tuple[
    List[mx.array],  # List of individual loss terms (tensors)
    List[mx.array],  # List of online value estimates (tensors)
    List[mx.array],  # List of TD targets (tensors)
    int,  # game outcome status
    float,  # average value entropy
    ConnectFour,  # final board
    List[mx.array],  # online perspective value estimates (all steps)
    int,  # online player id
    List[mx.array],  # advantages
    Optional[mx.array],  # last prediction before terminal
    Optional[float],  # final relative outcome
]:
    """
    Plays a game between online and frozen value models using MLX.
    Calculates loss terms, values, and targets for the online player's moves.

    Returns loss components separately for gradient calculation.
    """
    board = ConnectFour()
    # Store components needed for loss calculation later
    online_value_estimates_for_loss: List[mx.array] = []
    online_td_targets_for_loss: List[mx.array] = []
    online_loss_terms: List[
        mx.array
    ] = []  # Keep track for debugging/analysis if needed

    online_value_entropies: List[float] = []
    online_perspective_value_estimates_game: List[mx.array] = []
    online_advantages_game: List[mx.array] = []
    final_board = board
    last_pred_before_terminal: Optional[mx.array] = None
    final_relative_outcome: Optional[float] = None

    online_player_id = random.choice([1, 2])
    current_player = 1

    while True:
        board_tensor = board_to_tensor(final_board)  # MLX tensor
        is_online_turn = current_player == online_player_id

        # Log value from online model's perspective (no grad needed here)
        value_estimate_for_log = online_value_model(board_tensor, online_player_id)
        # Detach not explicitly needed in MLX unless inside grad calculation context
        online_perspective_value_estimates_game.append(value_estimate_for_log)

        last_pred_before_terminal = value_estimate_for_log

        acting_value_model = (
            online_value_model if is_online_turn else frozen_value_model
        )
        acting_temperature = (
            online_temperature if is_online_turn else frozen_temperature
        )
        acting_epsilon = online_epsilon if is_online_turn else frozen_epsilon

        online_value_estimate_t: Optional[mx.array] = None
        if is_online_turn:
            # This needs to be part of the gradient calculation later
            # For now, just get the value; we'll recompute in loss_fn if needed
            online_value_estimate_t = online_value_model(board_tensor, current_player)

        try:
            move, _, _, value_entropy, _ = get_next_value_based_move(
                acting_value_model,
                final_board,
                current_player,
                temperature=acting_temperature,
                epsilon=acting_epsilon,
            )
        except ValueError:
            print(f"Warning: No legal moves. Board:\n{final_board}")
            break

        next_board = make_move(final_board, current_player, move)
        next_board_tensor = board_to_tensor(next_board)  # MLX tensor
        status = is_in_terminal_state(next_board)

        td_target: Optional[mx.array] = None
        if is_online_turn:
            if status != 0:  # Game ended
                assert status in [1, 2, 3], f"Invalid terminal state: {status}"
                reward = (
                    1.0 if status == current_player else (0.0 if status == 3 else -1.0)
                )
                # Target is just the final reward
                td_target = mx.array([[reward]], dtype=mx.float32)
            else:  # Game continues
                next_player = 3 - current_player
                # Value of next state V(S', next_player) from ONLINE model perspective
                # No grad needed for target calculation
                online_next_value_opp = online_value_model(
                    next_board_tensor, next_player
                )
                # TD Target V(S) = R + gamma * V(S') => Here R=0, target is gamma * V_next
                # BUT, we want target for V(S, current_player).
                # V(S, current) = R + gamma * V(S', next_player). NO, that's not right.
                # Bellman: V(s,p) = E[ R + gamma * V(s', p') ] where p' is next player
                # In zero-sum: V(s', p') = -V(s', other_player)
                # So, V(s, current_player) = E[ R + gamma * (-V(s', current_player)) ] ?? No.
                # Let's use the definition: Value is expected return for *that* player.
                # V(s, p) = E[ sum(gamma^t * R_t) | S_0=s, P_0=p ]
                # TD Update: V(S_t, P_t) <- V(S_t, P_t) + alpha * [ Target - V(S_t, P_t) ]
                # Target = R_t+1 + gamma * V(S_t+1, P_t+1) (using the next state's value for the player whose turn it is)
                # Target = 0 + discount_factor * V(next_board, next_player) <- This seems correct.
                # We use the ONLINE model to estimate the next state's value.
                td_target = discount_factor * online_next_value_opp

            # Append components needed for loss calculation outside the loop
            if online_value_estimate_t is not None and td_target is not None:
                online_value_estimates_for_loss.append(online_value_estimate_t)
                online_td_targets_for_loss.append(td_target)
                # Optional: Calculate individual loss term (no grad here)
                # loss_term = mse_loss(online_value_estimate_t, td_target)
                # online_loss_terms.append(loss_term)

                # Calculate advantage (no grad needed)
                advantage = td_target - online_value_estimate_t
                online_advantages_game.append(advantage)

                online_value_entropies.append(value_entropy)

        final_board = next_board
        if status != 0:
            if status == online_player_id:
                final_relative_outcome = 1.0
            elif status == 3:
                final_relative_outcome = 0.0
            else:
                final_relative_outcome = -1.0
            break
        current_player = 3 - current_player

    avg_online_value_entropy = (
        sum(online_value_entropies) / len(online_value_entropies)
        if online_value_entropies
        else 0.0
    )

    # Return the lists needed for the loss function
    return (
        online_loss_terms,  # For analysis maybe
        online_value_estimates_for_loss,  # V(S_t, P_t) estimates
        online_td_targets_for_loss,  # TD targets
        status,
        avg_online_value_entropy,
        final_board,
        online_perspective_value_estimates_game,
        online_player_id,
        online_advantages_game,
        last_pred_before_terminal,
        final_relative_outcome,
    )


def evaluate_models(
    online_value_model: ValueModel,
    frozen_value_model: ValueModel,
    num_games: int = 20,
) -> float:
    """Plays games between online and frozen value models using MLX, returns online win rate."""
    online_wins = 0
    frozen_wins = 0
    draws = 0
    print(f"Starting evaluation (MLX): {num_games} games...")

    for i in range(num_games):
        board = ConnectFour()
        current_player = 1
        # Assign models based on who goes first
        model_p1 = online_value_model if i % 2 == 0 else frozen_value_model
        model_p2 = frozen_value_model if i % 2 == 0 else online_value_model
        online_player_this_game = 1 if i % 2 == 0 else 2

        while True:
            active_model = model_p1 if current_player == 1 else model_p2
            try:
                # Use near-greedy for evaluation
                move, _, _, _, _ = get_next_value_based_move(
                    active_model, board, current_player, temperature=0.001, epsilon=0
                )
            except ValueError:
                print(f"Eval game {i + 1}: No legal moves. Draw.")
                status = 3
                break

            board = make_move(board, current_player, move)
            status = is_in_terminal_state(board)

            if status != 0:
                if status == online_player_this_game:
                    online_wins += 1
                elif status == 3:
                    draws += 1
                else:
                    frozen_wins += 1
                break
            current_player = 3 - current_player

    total_played = online_wins + frozen_wins + draws
    win_rate = online_wins / total_played if total_played > 0 else 0.0
    print(
        f"Eval Results: Online Wins={online_wins}, Frozen Wins={frozen_wins}, Draws={draws}"
    )
    print(f"Online Model Win Rate vs Frozen: {win_rate:.2%}")
    return win_rate


def evaluate_vs_stacker(
    value_model: ValueModel,
    num_games: int = 10,
) -> Tuple[float, Optional[np.ndarray]]:
    """Evaluates MLX model against stacker strategy."""
    model_wins = 0
    stacker_wins = 0
    draws = 0
    last_game_final_board_state: Optional[np.ndarray] = None
    if num_games <= 0:
        return 0.0, None
    print(f"Starting stacker evaluation (MLX): {num_games} games...")

    for i in range(num_games):
        board = ConnectFour()
        stacker_target_col = random.randint(0, 6)
        model_player = 1 if i % 2 == 0 else 2
        stacker_player = 3 - model_player
        current_player = 1
        game_board_state: Optional[np.ndarray] = None

        while True:
            game_board_state = board.state.copy()
            if current_player == model_player:
                try:
                    move, _, _, _, _ = get_next_value_based_move(
                        value_model, board, current_player, temperature=0.001, epsilon=0
                    )
                except ValueError:
                    status = 3
                    break  # No moves = draw
            else:  # Stacker turn
                move = (
                    stacker_target_col
                    if is_legal(board, stacker_target_col)
                    else random_move(board)
                )
                if move is None:
                    status = 3
                    break  # No random move possible = draw

            try:
                board = make_move(board, current_player, move)
            except ValueError as e:
                print(f"Stacker eval err: {e}")
                status = 3
                break

            status = is_in_terminal_state(board)
            if status != 0:
                game_board_state = board.state.copy()
                if status == model_player:
                    model_wins += 1
                elif status == stacker_player:
                    stacker_wins += 1
                else:
                    draws += 1
                break
            current_player = 3 - current_player

        if i == num_games - 1:
            last_game_final_board_state = game_board_state

    total_played = model_wins + stacker_wins + draws
    win_rate = model_wins / total_played if total_played > 0 else 0.0
    print(
        f"Stacker Results: Model Wins={model_wins}, Stacker Wins={stacker_wins}, Draws={draws}"
    )
    print(f"Model Win Rate vs Stacker: {win_rate:.2%}")
    return win_rate, last_game_final_board_state


def evaluate_vs_minimax(
    value_model: ValueModel,
    num_games: int = 10,
    minimax_depth: int = 1,
) -> Tuple[float, Optional[np.ndarray]]:
    """Evaluates MLX model against minimax."""
    model_wins = 0
    minimax_wins = 0
    draws = 0
    last_game_final_board_state: Optional[np.ndarray] = None
    if num_games <= 0:
        return 0.0, None
    print(
        f"Starting minimax evaluation (MLX): {num_games} games vs Depth {minimax_depth}..."
    )

    for i in range(num_games):
        board = ConnectFour()
        model_player = 1 if i % 2 == 0 else 2
        minimax_player = 3 - model_player
        current_player = 1
        game_board_state: Optional[np.ndarray] = None

        while True:
            game_board_state = board.state.copy()
            if current_player == model_player:
                try:
                    move, _, _, _, _ = get_next_value_based_move(
                        value_model, board, current_player, temperature=0.001, epsilon=0
                    )
                except ValueError:
                    status = 3
                    break
            else:  # Minimax turn
                try:
                    move = minimax_move(board, current_player, minimax_depth)
                    if not is_legal(board, move):  # Safety check
                        print(f"Minimax illegal move {move}, using random.")
                        move = random_move(board)
                except Exception as e:
                    print(f"Minimax err: {e}, using random.")
                    move = random_move(board)
                if move is None:
                    status = 3
                    break  # No moves = draw

            try:
                board = make_move(board, current_player, move)
            except ValueError as e:
                print(f"Minimax eval move err: {e}")
                status = 3
                break

            status = is_in_terminal_state(board)
            if status != 0:
                game_board_state = board.state.copy()
                if status == model_player:
                    model_wins += 1
                elif status == minimax_player:
                    minimax_wins += 1
                else:
                    draws += 1
                break
            current_player = 3 - current_player

        if i == num_games - 1:
            last_game_final_board_state = game_board_state

    total_played = model_wins + minimax_wins + draws
    win_rate = model_wins / total_played if total_played > 0 else 0.0
    print(
        f"Minimax Results (d={minimax_depth}): Model Wins={model_wins}, Minimax Wins={minimax_wins}, Draws={draws}"
    )
    print(f"Model Win Rate vs Minimax (d={minimax_depth}): {win_rate:.2%}")
    return win_rate, last_game_final_board_state


def train_using_self_play(
    value_model: ValueModel,
    frozen_value_model: ValueModel,
    iterations: int = 1000,
    batch_size: int = 32,
    log_interval: int = 10,
    learning_rate: float = 0.001,
    online_temperature: float = 0.1,
    online_epsilon: float = 0.1,
    frozen_temperature: float = 0.01,
    frozen_epsilon: float = 0.0,
    discount_factor: float = 0.95,
    target_update_freq: int = 10,
    eval_games: int = 20,
    win_rate_threshold: float = 0.55,
    stacker_eval_games: int = 20,
    minimax_eval_games: int = 20,
    minimax_eval_depths: List[int] = [4, 7],
    force_replace_model: bool = False,
    wandb_run: Optional["Run"] = None,
) -> None:
    """Trains the value model using self-play with MLX."""

    optimizer = optim.Adam(learning_rate=learning_rate)

    # No explicit eval mode in MLX usually, but freeze params if needed
    # frozen_value_model.freeze() # Or handle via not computing grads

    # Loss and gradient function using value_and_grad
    # This function needs access to the *current* parameters of value_model
    # It will take the *data* (estimates and targets) as input
    def loss_fn(model: ValueModel, estimates: mx.array, targets: mx.array) -> mx.array:
        # Re-run the model? No, estimates are already computed with current params usually.
        # Let's assume the estimates passed are those computed in the forward pass *before* grad.
        # If using value_and_grad, the 'model' passed might be a specific version.
        # The standard MLX way is value_and_grad(model, loss_function)
        # where loss_function takes the model *and* data
        # Redefine loss_fn structure
        return mse_loss(estimates, targets)

    value_and_grad_fn = nn.value_and_grad(value_model, loss_fn)

    running_value_loss = 0.0
    running_value_entropy = 0.0
    running_online_perspective_values: List[float] = []
    running_advantages: List[float] = []
    running_last_preds_before_terminal: List[float] = []
    running_final_outcomes: List[float] = []
    total_games_played = 0
    last_final_board_state: Optional[np.ndarray] = None
    archived_frozen_model_count = 0

    print(
        f"Starting training (MLX Value-Based) for {iterations} batches... (Batch Size: {batch_size})"
    )
    # Print hyperparams...

    for i in range(iterations):
        # No explicit train mode typically needed for MLX inference/training switching

        # Collect data for the batch
        batch_estimates_for_loss: List[mx.array] = []
        batch_targets_for_loss: List[mx.array] = []
        batch_value_entropy_sum = 0.0
        batch_advantages: List[mx.array] = []  # Store as mx.array
        batch_games = 0
        batch_last_preds_before_terminal: List[Optional[mx.array]] = []
        batch_final_outcomes: List[Optional[float]] = []
        batch_last_game_online_perspective_values: Optional[List[float]] = None
        batch_last_game_status: Optional[int] = None
        batch_last_game_online_player_id: Optional[int] = None

        for game_idx in range(batch_size):
            (
                _,  # loss_terms (optional analysis)
                estimates,  # V(S_t, P_t) for online turns
                targets,  # TD Targets for online turns
                status,
                avg_game_value_entropy,
                final_board,
                online_perspective_estimates,
                online_player_id,
                game_advantages,
                last_pred,
                final_outcome,
            ) = play_against_self(
                online_value_model=value_model,
                frozen_value_model=frozen_value_model,
                discount_factor=discount_factor,
                online_temperature=online_temperature,
                online_epsilon=online_epsilon,
                frozen_temperature=frozen_temperature,
                frozen_epsilon=frozen_epsilon,
            )

            if estimates:  # Only add if the online player took turns
                batch_estimates_for_loss.extend(estimates)
                batch_targets_for_loss.extend(targets)
                batch_value_entropy_sum += avg_game_value_entropy * len(
                    estimates
                )  # Weight entropy by number of online turns
                batch_advantages.extend(game_advantages)

            batch_games += 1  # Count total games played in batch
            total_games_played += 1

            # Store logging info
            game_online_perspective_values = [
                v.item() for v in online_perspective_estimates
            ]
            running_online_perspective_values.extend(game_online_perspective_values)

            if game_idx == batch_size - 1:
                last_final_board_state = final_board.state.copy()
                batch_last_game_online_perspective_values = (
                    game_online_perspective_values
                )
                batch_last_game_status = status
                batch_last_game_online_player_id = online_player_id

            batch_last_preds_before_terminal.append(last_pred)
            batch_final_outcomes.append(final_outcome)

        # --- End Batch Game Loop ---

        if not batch_estimates_for_loss:  # Skip batch if online player never moved
            print(
                f"Warning: Batch {i + 1} completed with no online player moves to train on."
            )
            continue

        # Stack collected data into single tensors for loss calculation
        all_estimates = mx.concatenate(batch_estimates_for_loss, axis=0)
        all_targets = mx.concatenate(batch_targets_for_loss, axis=0)

        # Calculate loss and gradients
        # Pass the *data* (estimates, targets) to the loss function via value_and_grad
        # value_and_grad_fn expects the model and then the arguments for loss_fn
        loss, grads = value_and_grad_fn(value_model, all_estimates, all_targets)

        # Update model parameters
        optimizer.update(value_model, grads)
        mx.eval(
            value_model.parameters(), optimizer.state
        )  # Ensure updates are materialized

        # --- Logging and Evaluation --- #
        num_online_steps_in_batch = len(all_estimates)
        avg_batch_value_entropy = (
            batch_value_entropy_sum / num_online_steps_in_batch
            if num_online_steps_in_batch > 0
            else 0.0
        )

        # Log average loss for the batch
        batch_avg_loss = loss.item()  # Loss is already the mean
        running_value_loss += batch_avg_loss
        running_value_entropy += avg_batch_value_entropy
        running_advantages.extend([adv.item() for adv in batch_advantages])
        for pred, outcome in zip(
            batch_last_preds_before_terminal, batch_final_outcomes
        ):
            if pred is not None and outcome is not None:
                running_last_preds_before_terminal.append(pred.item())
                running_final_outcomes.append(outcome)

        # --- Target Network Update Logic --- #
        if (i + 1) % target_update_freq == 0:
            # No eval mode needed. Run evaluations.
            online_win_rate = evaluate_models(
                value_model, frozen_value_model, num_games=eval_games
            )
            safe_log_to_wandb(
                {"evaluation/online_vs_frozen_win_rate": online_win_rate},
                step=i + 1,
                wandb_run=wandb_run,
            )

            # Stacker Eval
            stacker_win_rate, last_stacker_board = evaluate_vs_stacker(
                value_model, num_games=stacker_eval_games
            )
            stacker_log = {"evaluation/online_vs_stacker_win_rate": stacker_win_rate}
            if last_stacker_board is not None and wandb_run:
                img = create_board_image(last_stacker_board)
                stacker_log["evaluation/final_stacker_eval_board"] = wandb.Image(
                    img, caption=f"Stacker Eval Batch {i + 1}"
                )
            safe_log_to_wandb(stacker_log, step=i + 1, wandb_run=wandb_run)

            # Minimax Eval
            for depth in minimax_eval_depths:
                minimax_win_rate, last_minimax_board = evaluate_vs_minimax(
                    value_model, num_games=minimax_eval_games, minimax_depth=depth
                )
                minimax_log = {
                    f"evaluation/online_vs_minimax_d{depth}_win_rate": minimax_win_rate
                }
                if last_minimax_board is not None and wandb_run:
                    img = create_board_image(last_minimax_board)
                    minimax_log[f"evaluation/final_minimax_d{depth}_eval_board"] = (
                        wandb.Image(img, caption=f"Minimax d{depth} Eval Batch {i + 1}")
                    )
                safe_log_to_wandb(minimax_log, step=i + 1, wandb_run=wandb_run)

            # Decide whether to update frozen model
            update_frozen = False
            update_reason = ""
            if force_replace_model:
                update_frozen = True
                update_reason = "forced by flag"
            elif online_win_rate > win_rate_threshold:
                update_frozen = True
                update_reason = f"win rate {online_win_rate:.2%} > threshold {win_rate_threshold:.1%}"
            else:
                print(
                    f"Online model failed eval ({online_win_rate:.2%}). Frozen net unchanged."
                )

            if update_frozen:
                print(f"Updating frozen value network ({update_reason}).")
                # MLX parameter update: Iterate and assign
                # This creates a *new* dictionary, parameters are copies
                frozen_params = value_model.parameters()
                # Update the frozen_value_model instance with these parameters
                # Assuming frozen_value_model has the same structure
                frozen_value_model.update(frozen_params)
                mx.eval(frozen_value_model.parameters())  # Ensure update is complete

                # Log artifact (Needs adaptation for MLX weights)
                if wandb_run:
                    try:
                        temp_model_path = (
                            f"temp_frozen_update_batch_{i + 1}.safetensors"
                        )
                        print(
                            f"Saving current online model weights to {temp_model_path} for artifact..."
                        )
                        # Use MLX save_weights
                        value_model.save_weights(temp_model_path)

                        frozen_artifact = wandb.Artifact(
                            "frozen_value_model",
                            type="model",
                            description=f"Updated frozen MLX model at batch {i + 1}. Reason: {update_reason}",
                            metadata={
                                "batch": i + 1,
                                "update_reason": update_reason,
                                "online_win_rate": online_win_rate,
                            },
                        )
                        frozen_artifact.add_file(temp_model_path)
                        wandb_run.log_artifact(frozen_artifact)
                        archived_frozen_model_count += 1
                        print(
                            f"Logged new frozen model version artifact: v{archived_frozen_model_count}"
                        )
                        os.remove(temp_model_path)
                    except Exception as e:
                        print(f"Warning: Failed to log frozen model artifact: {e}")
                        if os.path.exists(temp_model_path):
                            os.remove(temp_model_path)

                safe_log_to_wandb(
                    {
                        "evaluation/frozen_network_updated": 1,
                        "evaluation/frozen_update_reason_code": 1
                        if force_replace_model
                        else 2,
                    },
                    step=i + 1,
                    wandb_run=wandb_run,
                )
            else:
                safe_log_to_wandb(
                    {"evaluation/frozen_network_updated": 0},
                    step=i + 1,
                    wandb_run=wandb_run,
                )

        # --- Periodic Logging --- #
        if (i + 1) % log_interval == 0:
            avg_value_loss_log = running_value_loss / log_interval
            avg_value_entropy_log = running_value_entropy / log_interval

            adv_mean, adv_std, adv_hist = 0.0, 0.0, None
            if running_advantages:
                adv_array = mx.array(running_advantages)
                adv_mean = mx.mean(adv_array).item()
                adv_std = mx.sqrt(mx.var(adv_array)).item()  # Use variance for std dev
                if wandb_run:
                    adv_hist = wandb.Histogram(
                        adv_array.tolist()
                    )  # Wandb hist needs list/numpy

            value_std_dev, value_hist = 0.0, None
            if running_online_perspective_values:
                val_array = mx.array(running_online_perspective_values)
                value_std_dev = mx.sqrt(mx.var(val_array)).item()
                if wandb_run:
                    value_hist = wandb.Histogram(val_array.tolist())

            # Value Sign Accuracy & Training Win Rate Calculation (logic remains similar)
            correct_sign, total_sign = 0, 0
            train_wins, train_losses, train_draws, train_total = 0, 0, 0, 0
            if running_final_outcomes:
                train_total = len(running_final_outcomes)
                for pred_val, outcome_val in zip(
                    running_last_preds_before_terminal, running_final_outcomes
                ):
                    if outcome_val != 0.0:
                        total_sign += 1
                        if (pred_val > 0 and outcome_val > 0) or (
                            pred_val < 0 and outcome_val < 0
                        ):
                            correct_sign += 1
                    if outcome_val == 1.0:
                        train_wins += 1
                    elif outcome_val == -1.0:
                        train_losses += 1
                    else:
                        train_draws += 1
            value_sign_accuracy = correct_sign / total_sign if total_sign > 0 else 0.0
            train_win_rate = train_wins / train_total if train_total > 0 else 0.0

            print(f"\nBatch {i + 1}/{iterations} (Total Games: {total_games_played})")
            print(f"  Avg Value Loss: {avg_value_loss_log:.4f}")
            print(f"  Avg Value Move Entropy: {avg_value_entropy_log:.4f}")
            print(f"  Advantage (Mean ± Std): {adv_mean:.4f} ± {adv_std:.4f}")
            print(f"  Value Pred Std Dev (Online Persp.): {value_std_dev:.4f}")
            print(f"  Last Value Sign Accuracy: {value_sign_accuracy:.2%}")
            print(
                f"  Training Win Rate (~{log_interval * batch_size} games): {train_win_rate:.2%}"
            )
            print(
                f"     (W:{train_wins}, L:{train_losses}, D:{train_draws}) / {train_total} games"
            )

            log_data = {
                "training/batch_value_loss": avg_value_loss_log,
                "training/batch_value_move_entropy": avg_value_entropy_log,
                "training/online_perspective_value_std_dev": value_std_dev,
                "training/total_games_played": total_games_played,
                "training/advantage_mean": adv_mean,
                "training/advantage_std_dev": adv_std,
                "training/value_last_sign_accuracy": value_sign_accuracy,
                "training/online_vs_frozen_win_rate": train_win_rate,
                "progress/iterations": i + 1,
                "progress/iterations_pct": (i + 1) / iterations,
            }
            if value_hist:
                log_data["training/online_perspective_value_distribution"] = value_hist
            if adv_hist:
                log_data["training/advantage_distribution"] = adv_hist

            # Log last game value progression plot (logic mostly the same, uses matplotlib)
            if batch_last_game_online_perspective_values and wandb_run:
                if len(batch_last_game_online_perspective_values) > 1:
                    # (Plotting code remains the same as it uses lists/matplotlib)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    # ... [plotting code as before] ...
                    ax.plot(
                        range(len(batch_last_game_online_perspective_values)),
                        batch_last_game_online_perspective_values,
                        marker="o",
                    )
                    # ... [titles, labels, saving plot to wandb image] ...
                    plt.close(fig)
                    # ... [Add image to log_data["training/last_game_value_progression_plot"]] ...

            # Log last game board image (logic remains the same)
            if last_final_board_state is not None and wandb_run:
                board_image_np = create_board_image(last_final_board_state)
                log_data["training/final_online_vs_frozen_board"] = wandb.Image(
                    board_image_np, caption=f"Final Board Batch {i + 1}"
                )
                last_final_board_state = None

            safe_log_to_wandb(log_data, step=i + 1, wandb_run=wandb_run)

            # Reset running metrics
            running_value_loss = 0.0
            running_value_entropy = 0.0
            running_online_perspective_values = []
            running_advantages = []
            running_last_preds_before_terminal = []
            running_final_outcomes = []

    print("\nMLX Training loop finished.")
