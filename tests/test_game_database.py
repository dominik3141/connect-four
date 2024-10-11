from create_trainingdata import play_game
from game_database import encode_game, decode_game
import numpy as np


def test_serialization():
    """Test that the encoding and decoding of a game is lossless."""

    # play ten games
    for _ in range(10):
        game = play_game(1)

        encoded_game = encode_game(game)
        decoded_game = decode_game(encoded_game)

        assert np.array_equal(game[0], decoded_game[0])
