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
) -> Tuple[Move, Tensor, Tensor]:
    """
    Return the move, the probability of the move, and the policy distribution's entropy,
    ensuring only legal moves are selected.
    """
    # The loop for retrying illegal moves due to epsilon-greedy needs adjustment
    # Let's calculate probs once and only sample/retry if needed

    state_tensor = board_to_tensor(board).view(1, 7 * 6).float()

    logits = model(state_tensor)

    # Create a mask for legal moves
    legal_moves_mask = torch.tensor(
        [is_legal(board, move) for move in range(7)], dtype=torch.bool
    )

    # If no legal moves, something is wrong (shouldn't happen in Connect 4 unless board full)
    if not legal_moves_mask.any():
        # Handle this case, maybe raise error or return a special value
        # For now, let's pretend it doesn't happen in a valid game state
        # However, returning dummy values to avoid crashing
        print("Error: No legal moves found!")
        return 0, torch.tensor(0.0), torch.tensor(0.0)

    # Set logits of illegal moves to a large negative number
    masked_logits = torch.where(legal_moves_mask, logits, torch.tensor(-float("inf")))

    # Apply temperature scaling
    if temperature != 1.0 and temperature > 0:
        masked_logits = masked_logits / temperature
    elif temperature <= 0:
        print("Warning: Temperature is non-positive, using 1.0")

    # Calculate probabilities for legal moves
    probs = F.softmax(masked_logits, dim=-1).view(-1)

    # Create distribution only over legal moves if necessary, or handle sampling carefully
    distribution = Categorical(probs=probs)

    # Calculate entropy before potential epsilon-greedy move
    entropy = distribution.entropy()

    # Epsilon-greedy action selection
    if random.random() < epsilon:
        # Choose a random *legal* move
        legal_indices = torch.where(legal_moves_mask)[0]
        # Check if legal_indices is empty (should be handled above)
        if len(legal_indices) == 0:
            # This case should ideally be unreachable if the check above works
            print("Error: No legal moves for random choice!")
            return 0, torch.tensor(0.0), torch.tensor(0.0)
        move_idx = random.choice(legal_indices).item()
        move = Move(move_idx)
        probability = probs[move]
    else:
        # Sample from the distribution
        move_tensor = distribution.sample()
        move = Move(move_tensor.item())
        probability = probs[move]

    return move, probability, entropy


class ValueModel(nn.Module):
    """
    A model that takes a board state and a player (1 or 2) and returns that player's probability of winning.
    Output is between 0 and 1, where values closer to 1 indicate a higher probability
    of the specified player winning the game. It always evaluates from Player 1's perspective internally.
    """

    def __init__(self):
        super(ValueModel, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(7 * 6 * 2 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Output probability between 0 and 1 (for player 1)
        )

    def forward(self, x: torch.Tensor, player: int) -> torch.Tensor:
        # Convert input to shape (batch_size, 7, 6)
        original_x = x.view(-1, 7, 6)

        # Create separate binary planes for each player, always from P1's perspective
        player1_board = (original_x == 1).float().view(-1, 7 * 6)
        player2_board = (original_x == 2).float().view(-1, 7 * 6)
        processed_x = torch.cat([player1_board, player2_board], dim=-1)

        # Add next player information (who is next on the original board)
        # Note: 'who_is_next' expects the concatenated board representation
        next_player = who_is_next(processed_x)
        processed_x = torch.cat([processed_x, next_player.unsqueeze(1)], dim=-1)

        # Calculate the probability of player 1 winning
        p1_win_prob = self.lin(processed_x)

        # If the requested player is player 2, return 1 - p1_win_prob
        if player == 2:
            return 1.0 - p1_win_prob
        else:  # player == 1
            return p1_win_prob


def board_to_tensor(board: ConnectFour) -> Tensor:
    """Convert a ConnectFour board to a tensor representation consistent with model expectations."""
    tensor = torch.from_numpy(board.state).float()
    tensor = tensor.t()  # transpose to get 7x6
    return tensor.contiguous()
