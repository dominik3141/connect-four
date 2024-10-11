"""
Let two minimax agents play against each other and save the games.
"""

from minimax import minimax_move
from engine import ConnectFour
from game_database import initialize_database, save_game
from game_database import Game
from engine import make_move, is_in_terminal_state


def play_game(depth: int) -> Game:
    # initialize the game
    board = ConnectFour()
    game: Game = []

    # play the game
    while True:
        for player in [1, 2]:
            move = minimax_move(board, player, depth)

            # make the move
            board = make_move(board, player, move)
            game.append(board.state)

            # check if the game is in a terminal state
            terminal_state = is_in_terminal_state(board)

            if terminal_state != 0:
                return game


if __name__ == "__main__":
    NUM_GAMES = 100
    DEPTH = 3

    # initialize the database
    initialize_database()

    for _ in range(NUM_GAMES):
        game = play_game(DEPTH)
        save_game(game)
