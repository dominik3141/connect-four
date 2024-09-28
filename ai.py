from engine import ConnectFour
import engine
import torch
from torch import Tensor
from typing import Tuple
import wandb
from evaluations import evaluate_model_comprehensive
from model import DecisionModel, get_next_move


def loss_fn(
    probs: Tensor, outcome: int, player: int = 2, gamma: float = 0.95
) -> Tensor:
    if outcome == 3:  # Draw
        reward = -2.0
    elif outcome == player:  # Player wins
        reward = 4.0
    else:  # Player loses
        reward = -5.0

    num_moves = len(probs)
    discount_factors = torch.tensor([gamma**i for i in range(num_moves)])
    discounted_rewards = discount_factors * reward
    loss = torch.sum(discounted_rewards * probs)  # element-wise product

    # higher loss should indicate worse performance
    loss = -loss

    # normalize the loss
    loss = loss / num_moves

    return loss


def play_against_self(
    model: DecisionModel, temperature: float = 1.0, epsilon: float = 0
) -> Tuple[Tensor, Tensor, int]:
    """
    Play a game where the model plays against itself.
    Returns the move probabilities for both players and the game outcome.
    """
    board = ConnectFour()
    ai1_move_probs = []
    ai2_move_probs = []
    current_player = 1

    while True:
        move, prob = get_next_move(
            model, board, temperature=temperature, epsilon=epsilon
        )
        board = engine.make_move(board, current_player, move)

        if current_player == 1:
            ai1_move_probs.append(prob)
        else:
            ai2_move_probs.append(prob)

        if engine.is_in_terminal_state(board) != 0:
            break

        current_player = 3 - current_player  # Switch player

    status = engine.is_in_terminal_state(board)

    # Reverse the move probabilities for correct discounting
    ai1_move_probs.reverse()
    ai2_move_probs.reverse()

    ai1_move_probs_tensor = torch.stack(ai1_move_probs)
    ai2_move_probs_tensor = torch.stack(ai2_move_probs)

    return ai1_move_probs_tensor, ai2_move_probs_tensor, status


def self_play(
    model: DecisionModel,
    iterations: int = 100,
    learning_rate: float = 0.01,
    run=None,
    eval_interval: int = 100,
    temperature: float = 1.0,
    epsilon: float = 0,
) -> DecisionModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(iterations):
        optimizer.zero_grad()

        ai1_move_probs, ai2_move_probs, status = play_against_self(
            model, temperature=temperature, epsilon=epsilon
        )

        loss2 = loss_fn(ai2_move_probs, status, player=2)
        loss = loss2
        loss.backward()
        optimizer.step()

        # Evaluate the model every eval_interval iterations
        if run and (i + 1) % eval_interval == 0:
            eval_results = evaluate_model_comprehensive(model, num_games=10)
            total_games = sum(
                sum(results.values()) for results in eval_results.values()
            )
            total_wins = sum(results["wins"] for results in eval_results.values())
            win_rate = total_wins / total_games
            run.log({"win_rate": win_rate, "iteration": i + 1})
            for opponent, results in eval_results.items():
                for outcome, count in results.items():
                    run.log({f"{opponent}_{outcome}": count, "iteration": i + 1})

    return model


if __name__ == "__main__":
    # HYPERPARAMETERS
    learning_rate = 0.01
    iterations = 5000
    eval_interval = 50
    temperature = 1.0
    epsilon = 0.75

    # initialize the model
    model = DecisionModel()

    # Initialize wandb
    run = wandb.init(project="connect_four")

    # log the model architecture
    wandb.watch(model, log="all", log_freq=10)
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "iterations": iterations,
            "eval_interval": eval_interval,
            "model_architecture": str(model),
            "temperature": temperature,
            "epsilon": epsilon,
        }
    )

    # evaluate the untrained model
    print("Model evaluation before training:")
    initial_results = evaluate_model_comprehensive(model)
    print(initial_results)

    # train the model using self-play
    model = self_play(
        model,
        iterations=iterations,
        learning_rate=learning_rate,
        run=run,
        eval_interval=eval_interval,
        temperature=temperature,
        epsilon=epsilon,
    )

    # evaluate the trained model
    print("Model evaluation after training:")
    final_results = evaluate_model_comprehensive(model)
    print(final_results)

    # save the model
    torch.save(model.state_dict(), "model_self_play.pth")

    # save the model to wandb
    wandb.save("model_self_play.pth")

    # Log final comprehensive evaluation results to wandb
    for opponent, results in final_results.items():
        for outcome, count in results.items():
            wandb.log({f"final_{opponent}_{outcome}": count})

    # Finish the wandb run
    run.finish()
