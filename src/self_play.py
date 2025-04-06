from .model import ValueModel, get_next_value_based_move, board_to_tensor
from .engine import ConnectFour, make_move, is_in_terminal_state, is_legal, random_move
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, TYPE_CHECKING
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


def play_against_self(
    online_value_model: ValueModel,
    frozen_value_model: ValueModel,
    discount_factor: float,
    online_temperature: float = 0.01,
    online_epsilon: float = 0.1,
    frozen_temperature: float = 0.01,
    frozen_epsilon: float = 0.0,
) -> Tuple[
    Tensor,
    int,
    float,
    ConnectFour,
    List[Tensor],
    int,
    List[Tensor],
    Optional[Tensor],
    Optional[float],
]:
    """
    Plays a game where the online value model plays against a frozen version.
    Value loss is calculated only for the steps taken by the online player.
    Value estimates for logging are collected for *all* steps using the online model,
    always evaluated from the perspective of the player designated as 'online'.
    TD Target uses the ONLINE value model for the next state value V(S_t+1).
    Move selection is based on evaluating V(S', P) for the current player P
    in the state S' resulting from each possible action.

    Returns:
        total value loss (for online player),
        game outcome status (1, 2, or 3),
        average VALUE selection entropy (for online player),
        the final board state,
        a list of value estimates made by the online model (from the online player's perspective) for *every* state encountered,
        the online player ID,
        a list of advantage values calculated during the online player's turns,
        Value prediction (Tensor) made by the online model for the online player for the state *before* the terminal state.
        Final game outcome relative to the online player (+1 win, -1 loss, 0 draw).
    """
    board = ConnectFour()
    online_value_loss_terms: List[Tensor] = []
    online_value_entropies: List[float] = []
    online_perspective_value_estimates_game: List[Tensor] = []
    online_advantages_game: List[Tensor] = []
    final_board = board
    last_pred_before_terminal: Optional[Tensor] = None
    final_relative_outcome: Optional[float] = None

    online_player_id = random.choice([1, 2])
    current_player = 1

    while True:
        board_tensor = board_to_tensor(final_board)
        is_online_turn = current_player == online_player_id

        value_estimate_for_log = online_value_model(
            board_tensor,
            online_player_id,
        ).detach()
        online_perspective_value_estimates_game.append(value_estimate_for_log)

        last_pred_before_terminal = value_estimate_for_log

        acting_value_model = (
            online_value_model if is_online_turn else frozen_value_model
        )
        acting_temperature = (
            online_temperature if is_online_turn else frozen_temperature
        )
        acting_epsilon = online_epsilon if is_online_turn else frozen_epsilon

        online_value_estimate_t: Optional[Tensor] = None
        if is_online_turn:
            online_value_estimate_t = online_value_model(board_tensor, current_player)

        try:
            move, move_prob, all_probs, value_entropy, _ = get_next_value_based_move(
                acting_value_model,
                final_board,
                current_player,
                temperature=acting_temperature,
                epsilon=acting_epsilon,
            )
        except ValueError:
            print(
                f"Warning: get_next_value_based_move called with no legal moves. Board state:\n{final_board}"
            )
            break

        next_board = make_move(final_board, current_player, move)
        next_board_tensor = board_to_tensor(next_board)
        status = is_in_terminal_state(next_board)

        td_target: Optional[Tensor] = None
        if is_online_turn and online_value_estimate_t is not None:
            if status != 0:  # Game ended
                assert status in [1, 2, 3], f"Invalid terminal state: {status}"
                if status == current_player:
                    reward = 1.0
                elif status == 3:
                    reward = 0.0
                else:
                    reward = -1.0
                td_target = torch.tensor(
                    [[reward]],
                    device=online_value_estimate_t.device,
                    dtype=online_value_estimate_t.dtype,
                )
            else:  # Game continues
                next_player = 3 - current_player
                with torch.no_grad():
                    online_next_value_opp = online_value_model(
                        next_board_tensor, next_player
                    )
                td_target = discount_factor * (-online_next_value_opp)

        if (
            is_online_turn
            and online_value_estimate_t is not None
            and td_target is not None
        ):
            value_loss_term = F.mse_loss(online_value_estimate_t, td_target)
            online_value_loss_terms.append(value_loss_term)

            advantage = (td_target - online_value_estimate_t).detach()
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

    total_value_loss = (
        torch.stack(online_value_loss_terms).sum()
        if online_value_loss_terms
        else torch.tensor(0.0)
    )

    avg_online_value_entropy = (
        sum(online_value_entropies) / len(online_value_entropies)
        if online_value_entropies
        else 0.0
    )

    return (
        total_value_loss,
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
    """
    Plays games between online and frozen value models and returns the win rate of the online model.
    Uses near-greedy move selection based on value evaluation.
    """
    online_wins = 0
    frozen_wins = 0
    draws = 0

    print(f"Starting evaluation: {num_games} games...")

    for i in range(num_games):
        board = ConnectFour()
        current_player = 1
        model_p1 = online_value_model if i % 2 == 0 else frozen_value_model
        model_p2 = frozen_value_model if i % 2 == 0 else online_value_model
        online_player_this_game = 1 if i % 2 == 0 else 2

        while True:
            active_model = model_p1 if current_player == 1 else model_p2

            try:
                move, _, _, _, _ = get_next_value_based_move(
                    active_model,
                    board,
                    current_player,
                    temperature=0.001,
                    epsilon=0,
                )
            except ValueError:
                print(
                    f"Warning: Evaluation game {i + 1} ended unexpectedly (no legal moves). Treating as draw."
                )
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

    # Calculate win rate *after* all games are played
    total_played = online_wins + frozen_wins + draws
    if total_played == 0:
        win_rate = 0.0  # Avoid division by zero if num_games was 0
    else:
        # Win rate is based on games that didn't end unexpectedly early (before a win/loss/draw)
        # Note: The denominator is total_played (W+L+D), not num_games, in case some games errored out.
        # If you want win rate purely out of non-draw games: online_wins / (online_wins + frozen_wins)
        win_rate = online_wins / total_played

    print(
        f"Evaluation Results: Online Wins: {online_wins}, Frozen Wins: {frozen_wins}, Draws: {draws}"
    )
    print(f"Online Model Win Rate vs Frozen: {win_rate:.2%}")
    return win_rate  # Return the final win rate


def evaluate_vs_stacker(
    value_model: ValueModel,
    num_games: int = 10,
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Evaluates the model's ability to defend against a simple "stacker" strategy.
    Model uses near-greedy value-based moves.
    """
    model_wins = 0
    stacker_wins = 0
    draws = 0
    last_game_final_board_state: Optional[np.ndarray] = None

    if num_games <= 0:
        return 0.0, None

    print(f"Starting stacker evaluation: {num_games} games against Stacker...")

    for i in range(num_games):
        board = ConnectFour()
        stacker_target_col_this_game = random.randint(0, 6)
        model_player = 1 if i % 2 == 0 else 2
        stacker_player = 3 - model_player
        current_player = 1
        game_board_state: Optional[np.ndarray] = None

        while True:
            game_board_state = board.state.copy()

            if current_player == model_player:
                try:
                    move, _, _, _, _ = get_next_value_based_move(
                        value_model,
                        board,
                        current_player,
                        temperature=0.001,
                        epsilon=0,
                    )
                except ValueError:
                    print(
                        f"Warning: Stacker eval game {i + 1} - Model had no legal moves. Treating as draw."
                    )
                    status = 3
                    break
            else:
                if is_legal(board, stacker_target_col_this_game):
                    move = stacker_target_col_this_game
                else:
                    try:
                        move = random_move(board)
                    except ValueError:
                        status = 3
                        break

            try:
                board = make_move(board, current_player, move)
            except ValueError as e:
                print(f"Warning: Error making move during stacker eval: {e}")
                status = 3
                break

            status = is_in_terminal_state(board)

            if status != 0:
                game_board_state = board.state.copy()
                if status == model_player:
                    model_wins += 1
                elif status == stacker_player:
                    stacker_wins += 1
                elif status == 3:
                    draws += 1
                break

            current_player = 3 - current_player

        if i == num_games - 1 and game_board_state is not None:
            last_game_final_board_state = game_board_state

    total_played = model_wins + stacker_wins + draws
    model_win_rate = model_wins / total_played if total_played > 0 else 0.0

    print(
        f"Stacker Evaluation Results: Model Wins: {model_wins}, Stacker Wins: {stacker_wins}, Draws: {draws}"
    )
    print(f"Model Win Rate vs Random Stacker: {model_win_rate:.2%}")
    return model_win_rate, last_game_final_board_state


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
    force_replace_model: bool = False,
    wandb_run: Optional["Run"] = None,
) -> None:
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=learning_rate)

    frozen_value_model.eval()

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
        f"Starting training (Value-Based) for {iterations} batches of size {batch_size}..."
    )
    print(
        f"Online Temp: {online_temperature:.3f}, Online Epsilon: {online_epsilon:.3f}"
    )
    print(
        f"Frozen Temp: {frozen_temperature:.3f}, Frozen Epsilon: {frozen_epsilon:.3f}"
    )
    print(f"Frozen networks evaluated every {target_update_freq} batches.")
    print(f"Frozen network updated if online model win rate > {win_rate_threshold:.1%}")
    print(f"Stacker evaluation uses {stacker_eval_games} games.")

    for i in range(iterations):
        value_model.train()

        batch_value_loss = torch.tensor(0.0, requires_grad=True)
        batch_value_entropy_sum = 0.0
        batch_advantages: List[Tensor] = []
        batch_games = 0
        batch_last_preds_before_terminal: List[Optional[Tensor]] = []
        batch_final_outcomes: List[Optional[float]] = []
        batch_last_game_online_perspective_values: Optional[List[float]] = None
        batch_last_game_status: Optional[int] = None
        batch_last_game_online_player_id: Optional[int] = None

        for game_idx in range(batch_size):
            (
                total_value_loss,
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

            batch_value_loss = batch_value_loss + total_value_loss
            batch_value_entropy_sum = batch_value_entropy_sum + avg_game_value_entropy
            batch_advantages.extend(game_advantages)
            batch_games += 1
            total_games_played += 1

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

        if batch_games == 0:
            print(f"Warning: Batch {i + 1} completed with 0 games.")
            continue

        avg_batch_value_entropy = (
            batch_value_entropy_sum / batch_games if batch_games > 0 else 0.0
        )

        avg_batch_value_loss = batch_value_loss / batch_games

        optimizer_value.zero_grad()

        if avg_batch_value_loss.requires_grad:
            avg_batch_value_loss.backward()
        else:
            if avg_batch_value_loss.item() != 0.0:
                print(f"Warning: Batch {i + 1} Value loss is non-zero but has no grad.")

        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)

        optimizer_value.step()

        if (i + 1) % target_update_freq == 0:
            value_model.eval()
            online_win_rate = evaluate_models(
                value_model, frozen_value_model, num_games=eval_games
            )
            safe_log_to_wandb(
                {"evaluation/online_vs_frozen_win_rate": online_win_rate},
                step=i + 1,
                wandb_run=wandb_run,
            )

            stacker_win_rate, last_stacker_game_board = evaluate_vs_stacker(
                value_model,
                num_games=stacker_eval_games,
            )
            stacker_log_data = {
                "evaluation/online_vs_stacker_win_rate": stacker_win_rate
            }
            if last_stacker_game_board is not None and wandb_run is not None:
                stacker_board_image_np = create_board_image(last_stacker_game_board)
                stacker_log_data["evaluation/final_stacker_eval_board"] = wandb.Image(
                    stacker_board_image_np,
                    caption=f"Final Stacker Eval Board - Batch {i + 1}",
                )
            safe_log_to_wandb(stacker_log_data, step=i + 1, wandb_run=wandb_run)

            update_frozen = False
            update_reason = ""
            if force_replace_model:
                update_frozen = True
                update_reason = "forced by flag"
                print(
                    "Forcing update of frozen value network (force_replace_model=True)."
                )
            elif online_win_rate > win_rate_threshold:
                update_frozen = True
                update_reason = f"win rate {online_win_rate:.2%} > threshold {win_rate_threshold:.1%}"
                print(
                    f"Online model passed evaluation ({update_reason}). Updating frozen value network."
                )
            else:
                print(
                    f"Online model did not pass evaluation (Win Rate vs Frozen: {online_win_rate:.2%}). Frozen value network remains unchanged."
                )

            if update_frozen:
                # Save the ONLINE model's state dict (which will become the new frozen model)
                # Log it as a new version of the 'frozen_value_model' artifact
                if wandb_run is not None:
                    try:
                        # Define a temporary path for the state dict
                        temp_model_path = f"temp_frozen_update_batch_{i + 1}.pth"
                        print(
                            f"Preparing to update frozen model. Saving current online model state to {temp_model_path}..."
                        )
                        torch.save(value_model.state_dict(), temp_model_path)

                        # --- Log as new version of 'frozen_value_model' ---
                        frozen_model_artifact = wandb.Artifact(
                            "frozen_value_model",  # Consistent artifact name for versioning
                            type="model",
                            description=f"Updated frozen value model at batch {i + 1}. Replaced because: {update_reason}",
                            metadata={
                                "batch": i + 1,
                                "update_reason": update_reason,
                                "online_win_rate": online_win_rate,
                            },
                        )
                        frozen_model_artifact.add_file(temp_model_path)
                        wandb_run.log_artifact(
                            frozen_model_artifact
                        )  # Creates frozen_value_model:vX
                        archived_frozen_model_count += (
                            1  # Counter is now for versions of the main artifact
                        )
                        print(
                            f"Logged new frozen model version: {frozen_model_artifact.name}:v{archived_frozen_model_count}"
                        )
                        # Remove the temporary file after logging
                        os.remove(temp_model_path)
                        # --- End Artifact Logging ---

                    except Exception as e:
                        print(
                            f"Warning: Failed to log new version of frozen model artifact: {e}"
                        )
                        # Clean up temp file even if logging failed
                        if os.path.exists(temp_model_path):
                            try:
                                os.remove(temp_model_path)
                            except OSError as rm_err:
                                print(
                                    f"Warning: Failed to remove temporary model file {temp_model_path}: {rm_err}"
                                )

                # Load the online model's state into the frozen model instance
                frozen_value_model.load_state_dict(value_model.state_dict())
                # Log update success metric
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

        running_value_loss += avg_batch_value_loss.item()
        running_value_entropy += avg_batch_value_entropy
        running_advantages.extend([adv.item() for adv in batch_advantages])
        for pred, outcome in zip(
            batch_last_preds_before_terminal, batch_final_outcomes
        ):
            if pred is not None and outcome is not None:
                running_last_preds_before_terminal.append(pred.item())
                running_final_outcomes.append(outcome)

        if (i + 1) % log_interval == 0:
            avg_value_loss = running_value_loss / log_interval
            avg_value_entropy_log = running_value_entropy / log_interval

            adv_mean = 0.0
            adv_std = 0.0
            adv_hist = None
            if running_advantages:
                adv_tensor = torch.tensor(running_advantages)
                adv_mean = adv_tensor.mean().item()
                adv_std = adv_tensor.std().item()
                if wandb_run is not None:
                    adv_hist = wandb.Histogram(adv_tensor)

            value_std_dev = 0.0
            value_histogram = None
            if running_online_perspective_values:
                value_std_dev = (
                    torch.tensor(running_online_perspective_values).std().item()
                )
                if wandb_run is not None:
                    value_histogram = wandb.Histogram(running_online_perspective_values)

            correct_sign_predictions = 0
            total_sign_predictions = 0
            value_sign_accuracy = 0.0
            training_wins = 0
            training_losses = 0
            training_draws = 0
            training_total_games = 0
            training_win_rate = 0.0
            if running_final_outcomes:
                training_total_games = len(running_final_outcomes)
                for pred, outcome in zip(
                    running_last_preds_before_terminal, running_final_outcomes
                ):
                    if outcome != 0.0:
                        total_sign_predictions += 1
                        if (pred > 0 and outcome > 0) or (pred < 0 and outcome < 0):
                            correct_sign_predictions += 1
                    if outcome == 1.0:
                        training_wins += 1
                    elif outcome == -1.0:
                        training_losses += 1
                    else:
                        training_draws += 1
                if total_sign_predictions > 0:
                    value_sign_accuracy = (
                        correct_sign_predictions / total_sign_predictions
                    )
                if training_total_games > 0:
                    training_win_rate = training_wins / training_total_games

            print(f"\nBatch {i + 1}/{iterations} (Total Games: {total_games_played})")
            print(f"  Avg Value Loss: {avg_value_loss:.4f}")
            print(f"  Avg Value Move Entropy: {avg_value_entropy_log:.4f}")
            print(f"  Avg Advantage: {adv_mean:.4f}")
            print(f"  Std Dev Advantage: {adv_std:.4f}")
            print(f"  Std Dev Value Pred (Online Persp.): {value_std_dev:.4f}")
            print(
                f"  Last Value Sign Accuracy (vs Win/Loss): {value_sign_accuracy:.2%}"
            )
            print(
                f"  Training Win Rate (Online vs Frozen, last ~{log_interval * batch_size} games): {training_win_rate:.2%}"
            )
            print(
                f"     (W: {training_wins}, L: {training_losses}, D: {training_draws}) over {training_total_games} games"
            )

            log_data = {
                "training/batch_value_loss": avg_value_loss,
                "training/batch_value_move_entropy": avg_value_entropy_log,
                "training/online_perspective_value_std_dev": value_std_dev,
                "training/total_games_played": total_games_played,
                "training/advantage_mean": adv_mean,
                "training/advantage_std_dev": adv_std,
                "training/value_last_sign_accuracy": value_sign_accuracy,
                "training/online_vs_frozen_win_rate": training_win_rate,
                "progress/iterations": i + 1,
                "progress/iterations_pct": (i + 1) / iterations,
            }

            if value_histogram:
                log_data["training/online_perspective_value_distribution"] = (
                    value_histogram
                )
            if adv_hist:
                log_data["training/advantage_distribution"] = adv_hist

            if batch_last_game_online_perspective_values and wandb_run is not None:
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
                        ax.set_ylabel("Predicted Value (Online Player Perspective)")
                        outcome_str = "Outcome: Unknown"
                        if (
                            batch_last_game_status is not None
                            and batch_last_game_online_player_id is not None
                        ):
                            if (
                                batch_last_game_status
                                == batch_last_game_online_player_id
                            ):
                                outcome_str = "Outcome: Online Player Won"
                            elif batch_last_game_status == 3:
                                outcome_str = "Outcome: Draw"
                            else:
                                outcome_str = "Outcome: Online Player Lost"
                        ax.set_title(
                            f"Last Game Value Progression (Online Persp.) - Batch {i + 1}"
                        )
                        ax.set_ylim(-1.05, 1.05)
                        ax.grid(True)

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        img = Image.open(buf)
                        log_data["training/last_game_value_progression_plot"] = (
                            wandb.Image(
                                img,
                                caption=outcome_str,
                            )
                        )
                        plt.close(fig)
                        buf.close()

                    except Exception as e:
                        print(f"Warning: Failed to create value progression plot: {e}")
                        if fig is not None:
                            plt.close(fig)
                        if "buf" in locals() and not buf.closed:
                            buf.close()

            if last_final_board_state is not None and wandb_run is not None:
                board_image_np = create_board_image(last_final_board_state)
                log_data["training/final_online_vs_frozen_board"] = wandb.Image(
                    board_image_np,
                    caption=f"Final Online vs Frozen Board - Batch {i + 1}",
                )
                last_final_board_state = None

            safe_log_to_wandb(log_data, step=i + 1, wandb_run=wandb_run)

            running_value_loss = 0.0
            running_value_entropy = 0.0
            running_online_perspective_values = []
            running_advantages = []
            running_last_preds_before_terminal = []
            running_final_outcomes = []

    print("\nTraining loop finished.")
