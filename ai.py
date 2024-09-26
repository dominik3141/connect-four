from engine import ConnectFour, Move
import engine
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Dict, Tuple
from torch.distributions import Categorical


class DecisionModel(nn.Module):
    def __init__(self):
        super(DecisionModel, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(7 * 6, 128),  # Input layer (7 columns, 6 rows)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(
                64, 7
            ),  # Output layer (7 columns) (log_props for each possible move)
        )

    def forward(self, x):
        return self.lin(x)


def loss_fn(probs: Tensor, outcome: int, gamma: float = 0.95) -> Tensor:
    """
    probs is of shape [num_moves] (as it contains only the probability of the selected move)
    """
    dict_of_reward: Dict[int, float] = {
        1: -5.0,  # Player 1 won (undesirable)I
        2: 4.0,  # Player 2 won (AI won, desirable)
        3: -2.0,  # Stalemate (slightly undesirable)
    }

    # calculate the reward
    reward = dict_of_reward[outcome]

    # we calculate the loss in a very vectorized way
    num_moves = len(probs)
    discount_factors = torch.tensor([gamma**i for i in range(num_moves)])
    discounted_rewards = discount_factors * gamma * reward
    loss = torch.sum(discounted_rewards * probs)  # element-wise product

    # higher loss should indicate worse performance
    loss = -loss

    # we also want to normalize the loss
    loss = loss / num_moves

    return loss


def get_next_move(model: DecisionModel, board: ConnectFour) -> Tuple[Move, Tensor]:
    """
    Return the move and the probability of the move, ensuring only legal moves are selected.
    """
    state_tensor = torch.Tensor(board.state).view(7 * 6).float()
    logits = model(state_tensor)

    # Create a mask for legal moves, explicitly converting to int
    legal_moves = torch.Tensor([int(engine.is_legal(board, move)) for move in range(7)])

    # Set logits of illegal moves to a large negative number
    masked_logits = torch.where(legal_moves == 1, logits, torch.tensor(-1e9))

    probs = F.softmax(masked_logits, dim=-1)

    distribution = Categorical(probs)
    move = distribution.sample()
    probability = probs[move]

    return move, probability


def self_play(model: DecisionModel) -> DecisionModel:
    pass


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
    model = DecisionModel()

    # evaluate the untrained model
    print(f"Model evaluation before training: {evaluate_model(model)}")

    # train the model
    model = train_model(500, model=model)

    # evaluate the trained model
    print(f"Model evaluation after training: {evaluate_model(model)}")
