from engine import ConnectFour, Move
import engine
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple
from torch.distributions import Categorical
import wandb


class DecisionModel(nn.Module):
    def __init__(self):
        super(DecisionModel, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(7 * 6, 128),  # Input layer (7 columns, 6 rows)
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, x):
        return self.lin(x)


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


def get_next_move(
    model: DecisionModel, board: ConnectFour, temperature: float = 1.0
) -> Tuple[Move, Tensor]:
    """
    Return the move and the probability of the move, ensuring only legal moves are selected.
    """
    state_tensor = torch.Tensor(board.state).view(7 * 6).float()
    logits = model(state_tensor)

    # Create a mask for legal moves, explicitly converting to int
    legal_moves = torch.Tensor([int(engine.is_legal(board, move)) for move in range(7)])

    # Set logits of illegal moves to a large negative number
    masked_logits = torch.where(legal_moves == 1, logits, torch.tensor(-1e9))

    # add some noise to the logits
    if temperature != 1.0:
        masked_logits = masked_logits / temperature

    probs = F.softmax(masked_logits, dim=-1)

    distribution = Categorical(probs)
    move = distribution.sample()
    probability = probs[move]

    return move, probability


def play_against_self(
    model: DecisionModel, temperature: float = 1.0
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
        move, prob = get_next_move(model, board, temperature=temperature)
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
) -> DecisionModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(iterations):
        optimizer.zero_grad()

        ai1_move_probs, ai2_move_probs, status = play_against_self(
            model, temperature=temperature
        )

        # Compute losses for both players
        loss1 = loss_fn(ai1_move_probs, status, player=1)
        loss2 = loss_fn(ai2_move_probs, status, player=2)

        # add the losses
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        # Evaluate the model every eval_interval iterations
        if run and (i + 1) % eval_interval == 0:
            win_rate = evaluate_model(model)
            run.log({"win_rate": win_rate, "iteration": i + 1})

    return model


def play_against_random_player(model: DecisionModel) -> Tuple[Tensor, int]:
    board = ConnectFour()

    i = 0
    ai_move_probs = []
    while True:
        i += 1
        move = engine.random_move(board)
        board = engine.make_move(board, 1, move)

        if engine.is_in_terminal_state(board) != 0:
            break

        move, prob = get_next_move(model, board)
        board = engine.make_move(board, 2, move)
        ai_move_probs.append(prob)

        if engine.is_in_terminal_state(board) != 0:
            break

    status = engine.is_in_terminal_state(board)

    ai_move_probs.reverse()
    ai_move_probs_tensor = torch.stack(ai_move_probs)

    return ai_move_probs_tensor, status


def train_model(
    iterations: int, learning_rate: float = 0.01, model: DecisionModel = None
) -> DecisionModel:
    if model is None:
        model = DecisionModel()

    # for now without batching
    for i in range(iterations):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()

        ai_move_probs_tensor, status = play_against_random_player(model)
        loss = loss_fn(ai_move_probs_tensor, status)

        loss.backward()
        optimizer.step()

    return model


def evaluate_model(model: DecisionModel, iterations: int = 100) -> float:
    # return the ratio of wins for the model
    wins = 0
    for i in range(iterations):
        _, status = play_against_random_player(model)
        if status == 2:
            wins += 1
    return wins / iterations


if __name__ == "__main__":
    # HYPERPARAMETERS
    learning_rate = 0.02
    iterations = 4000
    eval_interval = 100
    temperature = 8.0
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
        }
    )

    # evaluate the untrained model
    print(f"Model evaluation before training: {evaluate_model(model)}")

    # train the model using self-play
    model = self_play(
        model,
        iterations=iterations,
        learning_rate=learning_rate,
        run=run,
        eval_interval=eval_interval,
        temperature=temperature,
    )

    # evaluate the trained model
    print(f"Model evaluation after training: {evaluate_model(model)}")

    # save the model
    torch.save(model.state_dict(), "model_self_play.pth")

    # Finish the wandb run
    run.finish()
