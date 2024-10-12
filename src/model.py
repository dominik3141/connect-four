import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from engine import ConnectFour, Move, is_legal
import random
from typing import Tuple
from torch import Tensor


class DecisionModel(nn.Module):
    """
    A model that takes a board state and the player and returns a move.
    """

    def __init__(self):
        super(DecisionModel, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(7 * 6 * 2 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to shape (batch_size, 7, 6)
        x = x.view(-1, 7, 6)

        player1_board = (x == 1).float().view(-1, 7 * 6)
        player2_board = (x == 2).float().view(-1, 7 * 6)
        x = torch.cat(
            [player1_board, player2_board], dim=-1
        )  # Shape: (batch_size, 7 * 6 * 2)

        # Calculate next player for each board in the batch
        next_player = who_is_next(x)

        x = torch.cat([x, next_player.unsqueeze(1)], dim=-1)

        return self.lin(x)


def who_is_next(board: torch.Tensor) -> torch.Tensor:
    """
    Return the player who is next to move for each board in the batch.
    """
    # Sum along the last dimension and check if even
    is_player1_next = (board.sum(dim=-1) % 2 == 0).float()
    # Convert to 1 for player 1, 2 for player 2
    return is_player1_next + (1 - is_player1_next) * 2


def get_next_model_move(
    model: DecisionModel,
    board: ConnectFour,
    temperature: float = 1.0,
    epsilon: float = 0,  # epsilon-greedy parameter
) -> Tuple[Move, Tensor]:
    """
    Return the move and the probability of the move, ensuring only legal moves are selected.
    """
    while True:
        state_tensor = torch.Tensor(board.state).view(1, 7 * 6).float()
        logits = model(state_tensor)

        # Create a mask for legal moves, explicitly converting to int
        legal_moves = torch.Tensor([int(is_legal(board, move)) for move in range(7)])

        # Set logits of illegal moves to a large negative number
        masked_logits = torch.where(legal_moves == 1, logits, torch.tensor(-1e9))

        # add some noise to the logits
        if temperature != 1.0:
            masked_logits = masked_logits / temperature

        probs = F.softmax(masked_logits, dim=-1).view(-1)

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
