from engine import ConnectFour, Move
import engine
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple
from torch.distributions import Categorical
import wandb
import random


class DecisionModel(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super(DecisionModel, self).__init__()

        self.embedding = nn.Linear(2 * 7 * 6, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to shape (2, 7, 6)
        x = x.view(7, 6)
        player1_board = (x == 1).float().view(-1)
        player2_board = (x == 2).float().view(-1)
        x = torch.cat([player1_board, player2_board], dim=0)  # Shape: (2 * 7 * 6)

        x = self.embedding(x)  # Shape: (d_model,)
        x = x.unsqueeze(0)  # Shape: (1, d_model)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Shape: (d_model,)
        return self.fc_out(x)


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
    model: DecisionModel,
    board: ConnectFour,
    temperature: float = 1.0,
    epsilon: float = 0,
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

    # choose a random move with probability epsilon
    if random.random() < epsilon:
        move = random.randint(0, 6)
    else:
        move = distribution.sample()

    # make sure the move is legal
    if not engine.is_legal(board, move):
        return get_next_move(model, board, temperature, epsilon)

    probability = probs[move]

    return move, probability


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
    learning_rate = 0.01
    iterations = 5000
    eval_interval = 50
    temperature = 1.0
    epsilon = 0.75

    # initialize the model
    model = DecisionModel()

    # load the weights from file
    # model.load_state_dict(torch.load("model_self_play.pth"))

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
    print(f"Model evaluation before training: {evaluate_model(model)}")

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
    print(f"Model evaluation after training: {evaluate_model(model)}")

    # save the model
    torch.save(model.state_dict(), "model_self_play.pth")

    # save the model to wandb
    wandb.save("model_self_play.pth")

    # Finish the wandb run
    run.finish()
