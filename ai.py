from engine import ConnectFour
import engine
import torch
from torch import Tensor
from typing import Tuple
import wandb
from evaluations import evaluate_model, minimax_move, log_evaluation_results
from model import DecisionModel, get_next_model_move


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


def play_against_minimax(
    model: DecisionModel, temperature: float = 1.0, epsilon: float = 0
) -> Tuple[Tensor, int]:
    """
    Play a game where the model plays against the minimax opponent.
    Returns the move probabilities for the model and the game outcome.
    """
    board = ConnectFour()
    ai_move_probs = []
    current_player = 1

    while True:
        if current_player == 1:
            move = minimax_move(board)
        else:
            move, prob = get_next_model_move(
                model, board, temperature=temperature, epsilon=epsilon
            )
            ai_move_probs.append(prob)

        board = engine.make_move(board, current_player, move)

        if engine.is_in_terminal_state(board) != 0:
            break

        current_player = 3 - current_player  # Switch player

    status = engine.is_in_terminal_state(board)

    # Reverse the move probabilities for correct discounting
    ai_move_probs.reverse()
    ai_move_probs_tensor = torch.stack(ai_move_probs)

    return ai_move_probs_tensor, status


def train_against_minimax(
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

        ai_move_probs, status = play_against_minimax(
            model, temperature=temperature, epsilon=epsilon
        )

        loss = loss_fn(ai_move_probs, status, player=2)  # always train as player 2
        loss.backward()
        optimizer.step()

        if i % eval_interval == 0:
            eval_results = evaluate_model(model)
            log_evaluation_results(run, eval_results, i)

    return model


if __name__ == "__main__":
    # HYPERPARAMETERS
    learning_rate = 0.0025
    iterations = 9000
    eval_interval = 200
    temperature = 1.0
    epsilon = 0.0

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
    initial_results = evaluate_model(model)
    print(initial_results)

    model = train_against_minimax(
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
    final_results = evaluate_model(model)
    print(final_results)

    # save the model
    torch.save(model.state_dict(), "model.pth")

    # save the model to wandb
    wandb.save("model.pth")

    # Log final comprehensive evaluation results to wandb
    for opponent, results in final_results.items():
        for outcome, count in results.items():
            wandb.log({f"final_{opponent}_{outcome}": count})

    # Finish the wandb run
    run.finish()
