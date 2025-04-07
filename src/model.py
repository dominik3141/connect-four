import mlx.core as mx
import mlx.nn as nn
import random  # Use standard library random for epsilon-greedy choice
from .engine import ConnectFour, Move, is_legal, make_move
from typing import Tuple, List, Optional


def board_to_tensor(board: ConnectFour) -> mx.array:
    """Convert a ConnectFour board to an mlx array representation."""
    # Convert numpy array to mlx array
    tensor = mx.array(board.state, dtype=mx.float32)
    # Models expect 7x6 (Channels-first convention isn't typical here)
    tensor = mx.transpose(tensor, (1, 0))  # transpose to get 7x6
    return tensor  # No contiguous needed for MLX


def get_next_value_based_move(
    value_model: "ValueModel",
    board: ConnectFour,
    current_player: int,
    temperature: float = 0.01,
    epsilon: float = 0.0,
) -> Tuple[Move, mx.array, mx.array, float, mx.array]:
    """
    Selects the next move based on the ValueModel's evaluation using MLX.

    Evaluates legal moves, calculates V(S', P) for the current player in the
    resulting state S', and chooses using temperature-scaled sampling or epsilon-greedy.

    Returns:
        - The chosen move (Move).
        - The probability of choosing that move under the actual policy (mx.array scalar).
        - The probabilities of all legal moves under the actual policy (mx.array).
        - The entropy of the actual move selection distribution (float).
        - The probability of choosing that move under a temp=1 policy (mx.array scalar).
    """
    legal_moves: List[Move] = [col for col in range(7) if is_legal(board, col)]

    if not legal_moves:
        raise ValueError("No legal moves available to evaluate.")

    next_state_values_list = []  # Use list to collect scalars/arrays before converting

    # Evaluate the value of each possible next state for the *current* player
    # MLX evaluation is often done outside explicit no_grad contexts
    for move in legal_moves:
        next_board = make_move(board, current_player, move)
        # Pass player explicitly now, model handles perspective
        value = value_model(board_to_tensor(next_board), current_player)
        next_state_values_list.append(value.item())  # Get scalar value

    values_tensor = mx.array(next_state_values_list)

    # --- Calculate Temp=1 probabilities ---
    # Use mx.softmax
    temp_1_probs = mx.softmax(values_tensor, axis=0)
    temp_1_move_prob: Optional[mx.array] = None

    # --- Determine chosen move and actual selection probabilities ---
    chosen_move: Move
    move_prob: mx.array
    probs: mx.array
    entropy: float
    chosen_move_idx: int

    if temperature <= 0:  # Greedy selection
        chosen_move_idx = mx.argmax(values_tensor).item()
        chosen_move = legal_moves[chosen_move_idx]
        # Create probability distribution for greedy choice
        probs = mx.zeros_like(values_tensor)
        # MLX array update needs care, direct indexing might not work as expected for assignment
        # Let's create it correctly:
        one_hot = mx.zeros_like(values_tensor)
        one_hot[chosen_move_idx] = 1.0
        probs = one_hot
        move_prob = mx.array(1.0)
        entropy = 0.0  # Entropy of a deterministic distribution is 0
    else:
        # Apply actual temperature
        scaled_values = values_tensor / temperature
        probs = mx.softmax(scaled_values, axis=0)  # Softmax probabilities

        # Calculate entropy using MLX operations
        # entropy = - sum(p * log(p))
        # Add small epsilon to prevent log(0)
        log_probs = mx.log(probs + 1e-9)
        entropy = -mx.sum(probs * log_probs).item()

        # Epsilon-greedy or sampling
        use_random_move = random.random() < epsilon  # Standard random float generation
        if use_random_move:
            # Epsilon step: Choose a random legal move
            chosen_move_idx = random.randint(
                0, len(legal_moves) - 1
            )  # Use standard random for index, no .item() needed
            chosen_move = legal_moves[chosen_move_idx]
            # The probability of *this specific* move under the softmax policy
            move_prob = probs[chosen_move_idx]
        else:
            # Sampling step: Sample from the softmax distribution
            # Use mlx.random.categorical
            # Ensure probs are on CPU for numpy conversion if needed by categorical, or use mx native
            # mlx.random.categorical expects log-probabilities or logits usually. Let's use logits.
            # We already have probs, let's sample using indices and probabilities
            # Probs might sum slightly off 1.0 due to float precision, normalize
            probs_normalized = probs / mx.sum(probs)
            # Generate random number and find corresponding bin
            # Alternatively, use mx.random.categorical directly if it accepts probs (check docs)
            # Assuming mx.random.categorical works like torch.multinomial:
            # It expects logits (unnormalized log probs). scaled_values are logits.
            chosen_move_idx_tensor = mx.random.categorical(
                scaled_values
            )  # Use mx.random.categorical
            chosen_move_idx = chosen_move_idx_tensor.item()
            chosen_move = legal_moves[chosen_move_idx]
            move_prob = probs[chosen_move_idx]  # Prob of the sampled move

    # --- Get the Temp=1 probability for the *actual* chosen move ---
    temp_1_move_prob = temp_1_probs[chosen_move_idx]  # Indexing is fine here

    # Map actual probabilities back to all 7 columns
    full_probs = mx.zeros(7)
    # Create indices array and update in one go if possible, or loop
    indices = mx.array(legal_moves)
    # mx.scatter might be the way, but a loop is simpler for now
    current_full_probs_list = full_probs.tolist()
    for i, move in enumerate(legal_moves):
        current_full_probs_list[move] = probs[i].item()
    full_probs = mx.array(current_full_probs_list)

    if temp_1_move_prob is None:
        raise RuntimeError("temp_1_move_prob was not assigned.")

    # Return mx.arrays where appropriate
    return chosen_move, move_prob, full_probs, entropy, temp_1_move_prob


# --- Updated ValueModel using MLX ---
class ValueModel(nn.Module):
    """
    MLX model: board state + player -> expected value [-1, 1].
    Input: (N, 7, 6) board tensor, player ID (1 or 2).
    """

    def __init__(self):
        super().__init__()
        # Input: 7x6 board -> two 7x6 planes -> flattened 84 features + 1 player feature = 85
        self.lin = nn.Sequential(
            nn.Linear(7 * 6 * 2 + 1, 512),  # Input size 85
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # Keep Tanh for [-1, 1] output
        )

    def __call__(self, x: mx.array, player: int) -> mx.array:
        """
        Evaluates board state `x` for `player` using MLX.

        Args:
            x: Board state tensor (N, 7, 6) or (7, 6). 0=empty, 1=P1, 2=P2.
            player: Player (1 or 2) perspective.

        Returns:
            Expected value V(x, player) as mx.array (N, 1).
        """
        # Ensure batch dimension
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=0)  # (1, 7, 6)

        # Create player planes
        player1_board = mx.reshape(
            (x == 1).astype(mx.float32), (x.shape[0], -1)
        )  # (N, 42)
        player2_board = mx.reshape(
            (x == 2).astype(mx.float32), (x.shape[0], -1)
        )  # (N, 42)
        processed_x = mx.concatenate([player1_board, player2_board], axis=-1)  # (N, 84)

        # Determine current turn player indicator (as in PyTorch version)
        num_pieces = mx.sum(mx.reshape(x != 0, (x.shape[0], -1)), axis=1)  # (N,)
        is_p1_turn = (num_pieces % 2 == 0).astype(mx.float32)  # (N,)
        current_turn_player_indicator = (
            is_p1_turn * 1.0 + (1.0 - is_p1_turn) * 2.0
        )  # (N,) values 1.0 or 2.0

        # Add indicator as feature
        processed_x = mx.concatenate(
            [processed_x, mx.expand_dims(current_turn_player_indicator, axis=1)],
            axis=-1,
        )  # (N, 85)

        # Get value from network V(x, current_turn_player)
        value_from_model = self.lin(processed_x)  # (N, 1)

        # Adjust sign based on requested 'player' perspective (zero-sum)
        # player_tensor = mx.full_like(current_turn_player_indicator, float(player)) # Incorrect: mx.full_like doesn't exist
        # Create tensor with the same shape/dtype as current_turn_player_indicator, filled with player value
        player_tensor = mx.full(
            current_turn_player_indicator.shape,
            float(player),
            dtype=current_turn_player_indicator.dtype,
        )  # (N,)

        # Use mx.where for conditional sign multiplier
        sign_multiplier = mx.where(
            player_tensor == current_turn_player_indicator,
            mx.array(1.0),
            mx.array(-1.0),
        )  # (N,)

        final_value = value_from_model * mx.expand_dims(
            sign_multiplier, axis=1
        )  # (N, 1)

        return final_value
