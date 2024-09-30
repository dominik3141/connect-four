import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from engine import ConnectFour, Move, is_legal
import random
from typing import Tuple
from torch import Tensor


class DecisionModel(nn.Module):
    def __init__(self):
        super(DecisionModel, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(7 * 6 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to shape (7, 6)
        x = x.view(7, 6)
        player1_board = (x == 1).float().view(-1)
        player2_board = (x == 2).float().view(-1)
        x = torch.cat([player1_board, player2_board], dim=0)  # Shape: (7 * 6 * 2)
        return self.lin(x)


def get_next_model_move(
    model: DecisionModel,
    board: ConnectFour,
    temperature: float = 1.0,
    epsilon: float = 0,
) -> Tuple[Move, Tensor]:
    """
    Return the move and the probability of the move, ensuring only legal moves are selected.
    """
    while True:
        state_tensor = torch.Tensor(board.state).view(7 * 6).float()
        logits = model(state_tensor)

        # Create a mask for legal moves, explicitly converting to int
        legal_moves = torch.Tensor([int(is_legal(board, move)) for move in range(7)])

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

        # make sure the move is legal, if not, try again
        if not is_legal(board, move):
            continue

        probability = probs[move]

        return move, probability
