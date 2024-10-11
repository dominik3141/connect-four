from src.create_trainingdata import play_game
from src.game_database import encode_game, decode_game
import numpy as np


def test_serialization():
    """Test that the encoding and decoding of a game is lossless."""

    # play ten games
    for game_num in range(10):
        game = play_game(1)

        encoded_game = encode_game(game)
        decoded_game = decode_game(encoded_game)

        try:
            assert np.array_equal(game, decoded_game)
        except AssertionError:
            print(f"Assertion failed for game {game_num}")
            print("Original game:")
            print(np.array(game))
            print("\nDecoded game:")
            print(np.array(decoded_game))
            print("\nDifferences:")
            print(np.array(game) - np.array(decoded_game))
            print("\nEncoded game:")
            print(encoded_game)
            raise  # Re-raise the assertion error after printing debug info
