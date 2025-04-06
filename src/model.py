import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .engine import ConnectFour, Move, is_legal, make_move
import random
from typing import Tuple, List, Optional
from torch import Tensor


def who_is_next(board: torch.Tensor) -> torch.Tensor:
    """
    Return the player who is next to move for each board in the batch.
    Expects input shape (batch_size, 7 * 6 * 2) from concatenated P1/P2 boards.
    """
    # Input 'board' here is already processed (P1 planes + P2 planes)
    # Count non-zero entries (pieces) across both player planes
    num_pieces = (board != 0).sum(dim=-1)
    # Player 1 moves on even turns (0, 2, 4...), Player 2 on odd turns (1, 3, 5...)
    is_player1_next = (num_pieces % 2 == 0).float()
    # Convert to 1 for player 1, 2 for player 2
    return is_player1_next + (1 - is_player1_next) * 2


def board_to_tensor(board: ConnectFour) -> Tensor:
    """Convert a ConnectFour board to a tensor representation consistent with model expectations."""
    # Convert numpy array to tensor
    tensor = torch.from_numpy(board.state).float()
    # Transpose is NO LONGER needed if models expect 6x7 directly?
    # Let's check ValueModel input processing. It uses view(-1, 7, 6) which implies input should be 7x6.
    # So, the transpose IS needed here.
    tensor = tensor.t()  # transpose to get 7x6
    return tensor.contiguous()


def get_next_value_based_move(
    value_model: "ValueModel",  # Forward reference ValueModel
    board: ConnectFour,
    current_player: int,
    temperature: float = 0.01,
    epsilon: float = 0.0,
) -> Tuple[Move, Tensor, Tensor, float, Tensor]:
    """
    Selects the next move based on the ValueModel's evaluation of subsequent states.

    Evaluates all legal moves, calculates the expected value V(S', P) for the *current player*
    in the state S' resulting from each move, and chooses based on these values using
    temperature-scaled softmax sampling or epsilon-greedy exploration.

    Returns:
        - The chosen move (Move).
        - The probability of choosing that move under the actual policy (using input temperature/epsilon) (Tensor).
        - The probabilities of all legal moves under the actual policy (Tensor).
        - The entropy of the actual move selection distribution (float).
        - The probability of choosing that move under a temp=1 policy (Tensor).
    """
    legal_moves: List[Move] = [col for col in range(7) if is_legal(board, col)]

    if not legal_moves:
        # Should only happen in a full board (stalemate) situation handled before calling this
        raise ValueError("No legal moves available to evaluate.")

    next_state_values = []
    device = next(value_model.parameters()).device  # Get model device

    # Evaluate the value of each possible next state for the *current* player
    with torch.no_grad():
        for move in legal_moves:
            next_board = make_move(board, current_player, move)
            next_board_tensor = board_to_tensor(next_board).to(device)
            # Get value V(S', P) where S' is next state, P is current player
            value = value_model(next_board_tensor, current_player)
            next_state_values.append(value.item())  # Store scalar value

    values_tensor = torch.tensor(next_state_values, device=device)

    # --- Calculate Temp=1 probabilities (for reporting/analysis) ---
    temp_1_probs = F.softmax(values_tensor, dim=0)
    temp_1_move_prob: Optional[Tensor] = None  # Initialize

    # --- Determine chosen move and actual selection probabilities ---
    chosen_move: Move
    move_prob: Tensor
    probs: Tensor
    entropy: float
    chosen_move_idx: int  # Index within legal_moves

    if temperature <= 0:  # Greedy selection
        chosen_move_idx = torch.argmax(values_tensor).item()
        chosen_move = legal_moves[chosen_move_idx]
        probs = torch.zeros(len(legal_moves), device=device)
        probs[chosen_move_idx] = 1.0
        move_prob = torch.tensor(1.0, device=device)
        entropy = 0.0
    else:
        # Apply actual temperature
        scaled_values = values_tensor / temperature
        probs = F.softmax(scaled_values, dim=0)
        distribution = Categorical(probs=probs)
        entropy = distribution.entropy().item()

        if random.random() < epsilon:
            chosen_move = random.choice(legal_moves)
            chosen_move_idx = legal_moves.index(chosen_move)
            move_prob = probs[chosen_move_idx]
        else:
            chosen_move_idx = distribution.sample().item()
            chosen_move = legal_moves[chosen_move_idx]
            move_prob = probs[chosen_move_idx]

    # --- Get the Temp=1 probability for the *actual* chosen move ---
    temp_1_move_prob = temp_1_probs[chosen_move_idx]

    # Map actual probabilities back to all 7 columns
    full_probs = torch.zeros(7, device=device)
    for i, move in enumerate(legal_moves):
        full_probs[move] = probs[i]

    # Ensure temp_1_move_prob is not None before returning
    if temp_1_move_prob is None:
        # This should be unreachable if legal_moves is not empty
        raise RuntimeError("temp_1_move_prob was not assigned.")

    return chosen_move, move_prob, full_probs, entropy, temp_1_move_prob


class ValueModel(nn.Module):
    """
    A model that takes a board state and a player (1 or 2) and returns that player's expected value
    for the game outcome, using rewards {Win: 1, Draw: 0, Loss: -1}. Output is bounded to [-1, 1] via tanh.
    """

    def __init__(self):
        super(ValueModel, self).__init__()
        # Input: 7x6 board state -> flattened 42 features.
        # Represent as two planes (player 1 presence, player 2 presence) -> 42 * 2 = 84 features.
        # Add one feature for the current player whose turn it is. Total: 85 features.
        self.lin = nn.Sequential(
            nn.Linear(7 * 6 * 2 + 1, 512),  # Input size confirmed: 84 + 1 = 85
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, player: int) -> torch.Tensor:
        """
        Evaluates the board state `x` from the perspective of `player`.

        Args:
            x: The board state tensor, expected shape (N, 7, 6) or (7, 6).
               Represents the board with 0=empty, 1=P1, 2=P2.
            player: The player (1 or 2) whose perspective the value should be calculated from.

        Returns:
            A tensor representing the expected value V(x, player), bounded between -1 and 1.
        """
        # Ensure input has batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension: (1, 7, 6)

        # Create separate binary planes for each player
        player1_board = (x == 1).float().view(x.size(0), -1)  # (N, 42)
        player2_board = (x == 2).float().view(x.size(0), -1)  # (N, 42)
        processed_x = torch.cat([player1_board, player2_board], dim=-1)  # (N, 84)

        # Determine the player whose turn it *actually* is on the board state `x`
        # Count pieces: P1 moves on even total pieces, P2 on odd
        num_pieces = (x != 0).view(x.size(0), -1).sum(dim=1)  # (N,)
        is_p1_turn = (num_pieces % 2 == 0).float()  # (N,)
        current_turn_player_indicator = (
            is_p1_turn * 1.0 + (1 - is_p1_turn) * 2.0
        )  # (N,), values are 1.0 or 2.0

        # Add the current turn player indicator as a feature
        processed_x = torch.cat(
            [processed_x, current_turn_player_indicator.unsqueeze(1)], dim=-1
        )  # (N, 85)

        # Calculate the value using the network. This value is implicitly V(x, 1)
        # because the network architecture doesn't explicitly use the 'player' argument yet.
        # Let's assume the network learns V(x, current_turn_player).
        value_from_model = self.lin(processed_x)  # (N, 1)

        # Now, adjust the sign based on the 'player' argument requested by the caller.
        # We want V(x, player). The model gives V(x, current_turn_player).
        # If player == current_turn_player, return value_from_model.
        # If player != current_turn_player, return -value_from_model (zero-sum assumption).
        player_tensor = torch.full_like(
            current_turn_player_indicator, float(player)
        )  # (N,)
        sign_multiplier = torch.where(
            player_tensor == current_turn_player_indicator, 1.0, -1.0
        )  # (N,)

        final_value = value_from_model * sign_multiplier.unsqueeze(1)  # (N, 1)

        return final_value
