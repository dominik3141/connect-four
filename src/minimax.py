from .engine import ConnectFour, is_in_terminal_state, make_move, is_legal
from typing import List, Dict
from .utils import SavedGame
import random
import numpy as np
import time
import ctypes
import os

# Load the shared library
# Ensure the path is correct for your system.
# If the library is in the root 'libs' directory relative to the workspace:
lib_path = "libs/libconnect4.so"
# If the library is relative to the src directory:
# lib_path = "../libs/libconnect4.so"

# Check if the library file exists before trying to load
if not os.path.exists(lib_path):
    # Try alternative path if running from src directory
    alt_lib_path = "../libs/libconnect4.so"
    if os.path.exists(alt_lib_path):
        lib_path = alt_lib_path
    else:
        raise FileNotFoundError(
            f"Shared library not found at {lib_path} or {alt_lib_path}"
        )

lib = ctypes.CDLL(lib_path)


# Define the argument and return types for the C function
lib.minimax_move.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # Board state (flattened)
    ctypes.c_int,  # Player
    ctypes.c_int,  # Depth
]
lib.minimax_move.restype = ctypes.c_int  # The best move as an integer


def minimax_move(board: ConnectFour, player: int, depth: int = 1) -> int:
    """Calculates the best move using the C minimax implementation."""
    # Flatten the board state to pass it to the C function
    # Ensure the array is C-contiguous and has the correct type (int32)
    flat_state = np.ascontiguousarray(board.state.flatten(), dtype=np.int32)
    flat_state_p = flat_state.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Call the C function
    best_move = lib.minimax_move(flat_state_p, player, depth)

    return best_move


def minimax_games(
    num_games: int,
    depth_player1: int = 1,
    depth_player2: int = 1,
    save_prob: float = 0.0,
):
    """
    Plays games between two minimax players and optionally saves them.
    """
    wins_p1 = 0
    wins_p2 = 0
    draws = 0
    saved_games_count = 0

    for i in range(num_games):
        board = ConnectFour()
        game_states = [board.state.tolist()]
        current_player = 1
        start_time = time.time()

        while True:
            depth = depth_player1 if current_player == 1 else depth_player2
            move = minimax_move(board, player=current_player, depth=depth)

            try:
                board = make_move(board, current_player, move)
                game_states.append(board.state.tolist())
            except ValueError:
                print(
                    f"Game {i + 1}: Minimax player {current_player} chose illegal move {move}. Ending game as error/draw."
                )
                # Treat as draw? Or handle differently? For now, break.
                status = 3  # Indicate draw due to error
                break

            status = is_in_terminal_state(board)
            if status != 0:
                break

            current_player = 3 - current_player

        end_time = time.time()
        duration = end_time - start_time

        if status == 1:
            wins_p1 += 1
            winner = 1
        elif status == 2:
            wins_p2 += 1
            winner = 2
        else:
            draws += 1
            winner = 3  # 3 for draw

        print(
            f"Game {i + 1}: P1(d={depth_player1}) vs P2(d={depth_player2}) -> Winner: {('P1' if winner == 1 else ('P2' if winner == 2 else 'Draw'))}, Duration: {duration:.2f}s"
        )

        if random.random() < save_prob:
            game_data = SavedGame(
                game_states=game_states,
                winner=winner,
                depth_player1=depth_player1,
                depth_player2=depth_player2,
            )
            game_data.save_to_file()
            saved_games_count += 1

    print("\n--- Minimax vs Minimax Results ---")
    print(f"Games Played: {num_games}")
    print(
        f"Player 1 (Depth {depth_player1}) Wins: {wins_p1} ({(wins_p1 / num_games):.1%})"
    )
    print(
        f"Player 2 (Depth {depth_player2}) Wins: {wins_p2} ({(wins_p2 / num_games):.1%})"
    )
    print(f"Draws: {draws} ({(draws / num_games):.1%})")
    if save_prob > 0:
        print(f"Games Saved: {saved_games_count}")


def benchmark_minimax(num_games: int, test_depths: List[int]) -> Dict[int, float]:
    """Benchmarks minimax move calculation time for different depths."""
    results = {}
    board = ConnectFour()  # Use a sample board state

    # Maybe add a few random moves to make it non-empty?
    for _ in range(6):
        try:
            move = random.choice([c for c in range(7) if is_legal(board, c)])
            player = 1 if _ % 2 == 0 else 2
            board = make_move(board, player, move)
        except IndexError:  # No legal moves left
            break

    print("\n--- Minimax Benchmarking ---")
    print(f"Using board state:\n{board.state}")
    print(f"Running {num_games} trials per depth...")

    for depth in test_depths:
        total_time = 0
        print(f"Testing Depth: {depth}")
        start_trial = time.time()
        for i in range(num_games):
            # Alternate player to test both perspectives if relevant
            player = 1 if i % 2 == 0 else 2
            start_time = time.time()
            _ = minimax_move(board, player=player, depth=depth)  # Call the function
            end_time = time.time()
            total_time += end_time - start_time
        end_trial = time.time()
        avg_time = total_time / num_games
        results[depth] = avg_time
        print(f"  Avg time per move: {avg_time:.6f} seconds")
        print(f"  Total trial time: {end_trial - start_trial:.2f} seconds")

    print("--- Benchmark Complete ---")
    return results


# Example usage
if __name__ == "__main__":
    # Initialize a sample board with a winning row for Player 1
    sample_state = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [2, 2, 1, 2, 0, 0, 0],
        ]
    )

    board = ConnectFour(sample_state)
    print("Board State:")
    print(board)

    best_move = minimax_move(board, player=1, depth=7)
    print(f"Best Move: {best_move}")
