import wandb
from dataclasses import dataclass
from typing import List
import json
import os
import uuid
from datetime import datetime

BoardState = List[List[int]]


@dataclass
class SavedGame:
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


def safe_log_to_wandb(data: dict[str, any]):
    """
    Logs data to wandb.
    If wandb is not initialized, it will not log anything.
    """
    if wandb.run is not None:
        wandb.log(data)
