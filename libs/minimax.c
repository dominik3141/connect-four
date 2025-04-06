#include <stdio.h>
#include <limits.h>
#include <stdlib.h> // For malloc/free if we needed dynamic allocation (not needed here)

#define ROWS 6
#define COLS 7
#define BOARD_SIZE (ROWS * COLS)

typedef enum
{
    EMPTY = 0,
    PLAYER1 = 1,
    PLAYER2 = 2
} Player;

// Function prototypes now include the board parameter
int isValidMove(int board[ROWS][COLS], int col);
void makeMove(int board[ROWS][COLS], int col, Player player);
void undoMove(int board[ROWS][COLS], int col);
int getWinner(int board[ROWS][COLS]);
int isBoardFull(int board[ROWS][COLS]);
int isTerminalState(int board[ROWS][COLS]);
int minimax(int board[ROWS][COLS], int depth, int maximizingPlayer);
int findBestMove(int board[ROWS][COLS], int depth, Player player);

// Check if a move is valid (column not full)
int isValidMove(int board[ROWS][COLS], int col)
{
    // Check column index validity
    if (col < 0 || col >= COLS) {
        return 0; // Invalid column index
    }
    // Check if the top cell in the column is empty
    return board[0][col] == EMPTY;
}

// Make a move in the specified column for the player
void makeMove(int board[ROWS][COLS], int col, Player player)
{
    // Find the lowest empty row in the column
    for (int r = ROWS - 1; r >= 0; r--)
    {
        if (board[r][col] == EMPTY)
        {
            board[r][col] = player;
            return; // Move made
        }
    }
    // Should not happen if isValidMove was checked, but good practice
}

// Undo the last move made in the specified column
void undoMove(int board[ROWS][COLS], int col)
{
    // Find the highest non-empty cell in the column
    for (int r = 0; r < ROWS; r++)
    {
        if (board[r][col] != EMPTY)
        {
            board[r][col] = EMPTY; // Set it back to empty
            return; // Move undone
        }
    }
    // Should not happen if the column wasn't empty
}

// Check if the board is completely full
int isBoardFull(int board[ROWS][COLS])
{
    // Check if the top row has any empty cells
    for (int c = 0; c < COLS; c++)
    {
        if (board[0][c] == EMPTY)
            return 0; // Found an empty cell, not full
    }
    return 1; // Top row is full, board is full
}

// Get the winner (1 for Player1, 2 for Player2, 0 for no winner yet)
int getWinner(int board[ROWS][COLS])
{
    // Check horizontal wins
    for (int r = 0; r < ROWS; r++)
    {
        for (int c = 0; c < COLS - 3; c++)
        {
            if (board[r][c] != EMPTY &&
                board[r][c] == board[r][c + 1] &&
                board[r][c] == board[r][c + 2] &&
                board[r][c] == board[r][c + 3])
                return board[r][c];
        }
    }

    // Check vertical wins
    for (int c = 0; c < COLS; c++)
    {
        for (int r = 0; r < ROWS - 3; r++)
        {
            if (board[r][c] != EMPTY &&
                board[r][c] == board[r + 1][c] &&
                board[r][c] == board[r + 2][c] &&
                board[r][c] == board[r + 3][c])
                return board[r][c];
        }
    }

    // Check positive diagonal wins (top-left to bottom-right)
    for (int r = 0; r < ROWS - 3; r++)
    {
        for (int c = 0; c < COLS - 3; c++)
        {
            if (board[r][c] != EMPTY &&
                board[r][c] == board[r + 1][c + 1] &&
                board[r][c] == board[r + 2][c + 2] &&
                board[r][c] == board[r + 3][c + 3])
                return board[r][c];
        }
    }

    // Check negative diagonal wins (bottom-left to top-right)
    for (int r = 3; r < ROWS; r++)
    {
        for (int c = 0; c < COLS - 3; c++)
        {
            if (board[r][c] != EMPTY &&
                board[r][c] == board[r - 1][c + 1] &&
                board[r][c] == board[r - 2][c + 2] &&
                board[r][c] == board[r - 3][c + 3])
                return board[r][c];
        }
    }

    return 0; // No winner yet
}

// Check if the game has reached a terminal state (win or draw)
int isTerminalState(int board[ROWS][COLS])
{
    return getWinner(board) != 0 || isBoardFull(board);
}

// Minimax algorithm implementation
// Returns score: +1 for P1 win, -1 for P2 win, 0 for draw/ongoing at depth limit
int minimax(int board[ROWS][COLS], int depth, int maximizingPlayer) // maximizingPlayer is 1 if P1, 0 if P2
{
    // Base case: check terminal state or depth limit
    if (depth == 0 || isTerminalState(board))
    {
        int winner = getWinner(board);
        if (winner == PLAYER1)
            return 100 + depth; // Favor faster wins
        if (winner == PLAYER2)
            return -100 - depth; // Favor faster wins (more negative is better for P2)
        return 0; // Draw or non-terminal at depth limit
    }

    // Recursive step: explore possible moves
    if (maximizingPlayer) // Player 1 (Maximizer)
    {
        int maxEval = INT_MIN;
        for (int col = 0; col < COLS; col++)
        {
            if (isValidMove(board, col))
            {
                makeMove(board, col, PLAYER1);
                int eval = minimax(board, depth - 1, 0); // Next player is minimizing (P2)
                undoMove(board, col);
                if (eval > maxEval)
                    maxEval = eval;
            }
        }
        // If no valid moves possible (shouldn't happen unless board full), return 0
        return (maxEval == INT_MIN) ? 0 : maxEval;
    }
    else // Player 2 (Minimizer)
    {
        int minEval = INT_MAX;
        for (int col = 0; col < COLS; col++)
        {
            if (isValidMove(board, col))
            {
                makeMove(board, col, PLAYER2);
                int eval = minimax(board, depth - 1, 1); // Next player is maximizing (P1)
                undoMove(board, col);
                if (eval < minEval)
                    minEval = eval;
            }
        }
        // If no valid moves possible, return 0
        return (minEval == INT_MAX) ? 0 : minEval;
    }
}

// Finds the best move for the given player using minimax
int findBestMove(int board[ROWS][COLS], int depth, Player player)
{
    int bestScore = (player == PLAYER1) ? INT_MIN : INT_MAX;
    int bestMove = -1;

    // Iterate through all possible columns
    for (int col = 0; col < COLS; col++)
    {
        if (isValidMove(board, col))
        {
            // Try the move
            makeMove(board, col, player);

            // Evaluate the move using minimax for the opponent
            // Note: The depth passed to minimax is depth-1 because we just made a move.
            // The 'maximizingPlayer' argument depends on who the *next* player is.
            // If current player is P1 (maximizer), the next is P2 (minimizer, so 0).
            // If current player is P2 (minimizer), the next is P1 (maximizer, so 1).
            int score = minimax(board, depth - 1, (player == PLAYER1) ? 0 : 1);

            // Undo the move
            undoMove(board, col);

            // Update best move if this move is better
            if (player == PLAYER1)
            {
                if (score > bestScore)
                {
                    bestScore = score;
                    bestMove = col;
                }
            }
            else // Player 2
            {
                if (score < bestScore)
                {
                    bestScore = score;
                    bestMove = col;
                }
            }
        }
    }

    // If no valid moves found (e.g., board is full, though isTerminalState should handle this),
    // or if all moves lead to immediate loss and bestMove remains -1,
    // we need to return *some* valid move if one exists, or handle the error.
    // Let's find the first valid move as a fallback if bestMove is still -1.
    if (bestMove == -1) {
        for (int col = 0; col < COLS; col++) {
            if (isValidMove(board, col)) {
                bestMove = col;
                break;
            }
        }
    }

    return bestMove; // Return the best column index found
}


// --- Interface Function for Python ---
// This is the function that will be called from Python via ctypes
// It takes a flattened board, the player to move, and the search depth.
// It returns the best move (column index) for that player.
int minimax_move(int *flat_board, int player, int depth) {
    int board[ROWS][COLS];

    // Copy the flattened board data from Python into the local 2D C array
    // Assumes row-major order from NumPy's flatten()
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            board[r][c] = flat_board[r * COLS + c];
        }
    }

    // Ensure the player value is valid
    Player currentPlayer = (player == 1) ? PLAYER1 : PLAYER2;

    // Call the C internal function to find the best move
    // We pass depth directly, findBestMove will decrement it for the recursive calls
    int bestMove = findBestMove(board, depth, currentPlayer);

    return bestMove;
}

/*
// Main function removed - no longer needed for shared library
int main()
{
    // Example initial board setup (now irrelevant for the library)
    int initialBoard[ROWS][COLS] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 2, 0, 0, 0, 0},
        {0, 1, 1, 2, 0, 0, 0},
        {0, 2, 1, 1, 0, 0, 0},
        {1, 1, 2, 2, 0, 0, 0}};

     int board[ROWS][COLS]; // Use local board in main if needed for testing
     for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            board[r][c] = initialBoard[r][c];
        }
     }

    printf("Initial Board Position:
");
    // printBoard(board); // Would need printBoard to take board param

    // Determine whose turn it is (Logic depends on how you define turn order)
    // Player nextPlayer = getNextPlayer(board); // Would need getNextPlayer
    // printf("Next player to move: Player %d
", nextPlayer);

    // Find the best move for Player 1 example
    int bestMove = findBestMove(board, 5, PLAYER1); // Depth 5
    printf("Best move for Player 1: Column %d
", bestMove);


    return 0;
}
*/