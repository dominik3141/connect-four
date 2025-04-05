import wandb
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import json
import os
import uuid
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

BoardState = List[List[int]]  # can be accessed as board[row][column]


@dataclass
class SavedGame:
    """
    A class to save a game to a file.
    Only used for the frontend.
    """

    id: str
    depth_player1: int
    depth_player2: int
    result: int
    states: List[BoardState]
    timestamp: str

    def __init__(
        self,
        depth_player1: int,
        depth_player2: int,
        result: int,
        states: List[BoardState],
    ):
        self.id = str(uuid.uuid4())
        self.depth_player1 = depth_player1
        self.depth_player2 = depth_player2
        self.result = result
        self.states = states
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "depth_player1": self.depth_player1,
            "depth_player2": self.depth_player2,
            "result": self.result,
            "states": self.states,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SavedGame":
        game = cls(
            depth_player1=data["depth_player1"],
            depth_player2=data["depth_player2"],
            result=data["result"],
            states=data["states"],
        )
        game.id = data["id"]
        game.timestamp = data["timestamp"]
        return game

    def save_to_file(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        filename = f"game_{self.id}.json"
        filepath = os.path.join(directory, filename)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_from_file(cls, filepath: str) -> "SavedGame":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def safe_log_to_wandb(
    data: Dict[str, Any], step: Optional[int] = None, wandb_run: Optional["Run"] = None
) -> None:
    """Logs data to W&B if a run is active, otherwise does nothing."""
    current_run = wandb_run if wandb_run is not None else wandb.run
    if current_run:
        try:
            current_run.log(data, step=step)
        except Exception as e:
            print(f"Warning: Failed to log data to W&B: {e}")
            print(f"Data attempted to log: {data}")
    # else:
    # Optional: print("W&B run not active, skipping logging.")


def create_board_image(board_state: np.ndarray, cell_size: int = 50) -> np.ndarray:
    """
    Generates an RGB image representation of the ConnectFour board state using Pillow.

    Args:
        board_state: A 6x7 numpy array (0=empty, 1=P1, 2=P2).
                      Assumes standard Connect Four indexing (e.g., board_state[0,0] is top-left).
        cell_size: The size of each grid cell in pixels.

    Returns:
        A numpy array representing the RGB image.
    """
    rows, cols = board_state.shape
    height = rows * cell_size
    width = cols * cell_size
    # Calculate padding for circles to not touch the cell borders
    circle_padding = int(cell_size * 0.1)  # e.g., 10% padding around circle

    # Define colors (RGB tuples)
    background_color = (0, 100, 200)  # Blue background
    empty_color = (255, 255, 255)  # White for empty slots
    player1_color = (255, 255, 0)  # Yellow
    player2_color = (255, 0, 0)  # Red

    # Create a new image with the blue background
    img = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(img)

    for r in range(rows):
        for c in range(cols):
            player = board_state[r, c]
            color = empty_color
            if player == 1:
                color = player1_color
            elif player == 2:
                color = player2_color

            # Calculate bounding box for the circle within the cell
            # Top-left corner (x0, y0) and bottom-right corner (x1, y1)
            x0 = c * cell_size + circle_padding
            y0 = r * cell_size + circle_padding
            x1 = (c + 1) * cell_size - circle_padding
            y1 = (r + 1) * cell_size - circle_padding

            # Draw the ellipse (circle) within the bounding box
            draw.ellipse(
                [x0, y0, x1, y1], fill=color, outline=None
            )  # No outline for cleaner look

    # Convert PIL Image to NumPy array for logging
    return np.array(img)
