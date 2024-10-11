"""
We want to let two minimax player play against each other and save their games to a database so
that we do not have to create games for supervised learning on-the-fly each time we need them.

A game is a list of board states.
"""

from typing import List, Tuple
import sqlite3
import hashlib
import datetime
import os

BoardState = List[
    List[int]
]  # a board state is simply a 2D list of integers, can be accessed as board[row][column]
Game = List[BoardState]  # a game is a list of board states
GameCoin = Tuple[
    int, int
]  # a game coin is a tuple of two bits (which we have to treat as integers for python reasons)
EncodedBoard = List[GameCoin]  # an encoded board is a list of game coins

DATABASE_FILE = "games.sqlite"


def initialize_database() -> None:
    """
    Initialize the database if it doesn't exist. Does nothing if the database file is already present.
    """
    if os.path.exists(DATABASE_FILE):
        return

    with sqlite3.connect(DATABASE_FILE) as connection:
        cursor = connection.cursor()

        # create the games table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT NOT NULL UNIQUE,
                game BLOB NOT NULL
            )
            """
        )

        # create the game metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS game_metadata (
                game_id INTEGER PRIMARY KEY,
                first_saved TIMESTAMP NOT NULL,
                found_count INTEGER NOT NULL DEFAULT 1,
                last_found TIMESTAMP NOT NULL,
                game_length INTEGER NOT NULL,
                FOREIGN KEY (game_id) REFERENCES games (id)
            )
            """
        )


def save_game(game: Game) -> None:
    """
    Save a game to the database.
    """
    game_hash = calculate_hash(game)
    encoded_game = encode_game(game)
    current_time = datetime.datetime.now()
    game_length = len(game)

    with sqlite3.connect(DATABASE_FILE) as connection:
        cursor = connection.cursor()

        try:
            cursor.execute(
                "INSERT INTO games (hash, game) VALUES (?, ?)",
                (game_hash, encoded_game),
            )
            game_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO game_metadata (game_id, first_saved, last_found, game_length) VALUES (?, ?, ?, ?)",
                (game_id, current_time, current_time, game_length),
            )
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM games WHERE hash = ?", (game_hash,))
            game_id = cursor.fetchone()[0]
            cursor.execute(
                """
                UPDATE game_metadata 
                SET found_count = found_count + 1, last_found = ? 
                WHERE game_id = ?
                """,
                (current_time, game_id),
            )

        connection.commit()


def calculate_hash(game: Game) -> str:
    """
    We only want to store unique games, so we need to hash them.
    """
    return hashlib.sha256(encode_game(game)).hexdigest()


def cell_to_game_coin(cell: int) -> GameCoin:
    """Converts a cell value to a GameCoin."""
    if cell == 0:
        return (0, 0)  # Empty
    elif cell == 1:
        return (0, 1)  # Player 1
    elif cell == 2:
        return (1, 0)  # Player 2
    else:
        raise ValueError("Invalid cell value. Must be 0, 1, or 2.")


def game_coin_to_cell(coin: GameCoin) -> int:
    """Converts a GameCoin back to a cell value."""
    if coin == (0, 0):
        return 0  # Empty
    elif coin == (0, 1):
        return 1  # Player 1
    elif coin == (1, 0):
        return 2  # Player 2
    else:
        raise ValueError("Invalid GameCoin. Must be (0, 0), (0, 1), or (1, 0).")


def encode_board_state(board: BoardState) -> bytes:
    """Encodes a BoardState to bytes."""
    encoded: EncodedBoard = [cell_to_game_coin(cell) for row in board for cell in row]
    bit_list = [
        bit for coin in encoded for bit in coin
    ]  # Flatten the list of GameCoins
    # Convert the list of bits to bytes
    return bits_to_bytes(bit_list)


def decode_board_state(encoded_bytes: bytes) -> BoardState:
    """Decodes bytes back to a BoardState."""
    bit_list = bytes_to_bits(encoded_bytes, 84)  # 84 bits for a board state
    encoded: EncodedBoard = [
        (bit_list[i], bit_list[i + 1]) for i in range(0, len(bit_list), 2)
    ]
    # Convert back to a 2D board state
    return [
        [game_coin_to_cell(encoded[row * 7 + col]) for col in range(7)]
        for row in range(6)
    ]


def bits_to_bytes(bits: List[int]) -> bytes:
    """Converts a list of bits (0s and 1s) to a bytes object using little-endian bit order."""
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= bits[i + j] << j  # Set bit j (LSB first)
        byte_array.append(byte)
    return bytes(byte_array)


def bytes_to_bits(byte_data: bytes, num_bits: int) -> List[int]:
    """Converts a bytes object back to a list of bits using little-endian bit order."""
    bits = []
    for byte in byte_data:
        for i in range(8):  # Iterate from LSB to MSB
            bits.append((byte >> i) & 1)
    return bits[:num_bits]  # Trim to the desired number of bits


def encode_game(game: Game) -> bytes:
    """Encodes a list of BoardStates (a game) to bytes."""
    return b"".join(encode_board_state(board) for board in game)


def decode_game(encoded_bytes: bytes) -> Game:
    """Decodes bytes back to a list of BoardStates (a game)."""
    board_size_in_bytes = 11  # 7*6*2 / 8 = 10.5
    return [
        decode_board_state(encoded_bytes[i : i + board_size_in_bytes])
        for i in range(0, len(encoded_bytes), board_size_in_bytes)
    ]
