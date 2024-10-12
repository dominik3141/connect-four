"""
Train the model using saved games.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import sqlite3
from typing import List, Tuple
from game_database import DATABASE_FILE, decode_game, BoardState, Game
import torch.optim as optim
from model import DecisionModel
import torch.nn as nn
import os
import wandb


class ConnectFourDataset(Dataset):
    def __init__(self, database_file: str):
        self.database_file = database_file
        self.game_ids: List[int] = self._load_game_ids()

    def _load_game_ids(self) -> List[int]:
        with sqlite3.connect(self.database_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM games")
            return [row[0] for row in cursor.fetchall()]

    def __len__(self) -> int:
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        game_id = self.game_ids[idx]
        game = self._load_game(game_id)

        # Select a random position from the game
        position_idx = torch.randint(0, len(game) - 1, (1,)).item()

        current_state = game[position_idx]
        next_state = game[position_idx + 1]

        # Convert BoardState to tensor
        current_tensor = torch.tensor(current_state, dtype=torch.float32)

        # Calculate the move (difference between states)
        move = self._calculate_move(current_state, next_state)

        return current_tensor, move

    def _load_game(self, game_id: int) -> Game:
        with sqlite3.connect(self.database_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT game FROM games WHERE id = ?", (game_id,))
            encoded_game = cursor.fetchone()[0]
        return decode_game(encoded_game)

    def _calculate_move(
        self, current_state: BoardState, next_state: BoardState
    ) -> torch.Tensor:
        for col in range(7):
            for row in range(6):
                if current_state[row][col] != next_state[row][col]:
                    return torch.tensor([col], dtype=torch.long)

        print("No move found between states")
        print(current_state)
        print(next_state)
        raise ValueError("No move found between states")


# Create dataset and dataloaders
def create_dataloaders(
    batch_size: int = 32, num_workers: int = 4, val_split: float = 0.2, seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders with a reproducible split.

    val_split: Fraction of the dataset to use for validation (0.0 to 1.0)
    seed: Random seed for reproducibility
    """
    dataset = ConnectFourDataset(DATABASE_FILE)

    # Calculate sizes for the split
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Use random_split with a generator for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


def train_model(
    model: DecisionModel,
    dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "mps",
) -> DecisionModel:
    """
    Train the DecisionModel using supervised learning.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize wandb
    wandb.init(
        project="connect_four",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": dataloader.batch_size,
        },
    )
    wandb.watch(model, log="all", log_freq=10)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_states, batch_moves in dataloader:
            batch_states, batch_moves = batch_states.to(device), batch_moves.to(device)

            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_moves.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_moves.squeeze()).sum().item()
            total_predictions += batch_moves.size(0)

        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
        )

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy,
            }
        )

        # Save the model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "model.pth")

    return model


def evaluate_model(
    model: DecisionModel,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "mps",
) -> Tuple[float, float]:
    """
    Evaluate the DecisionModel on the given dataloader.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_states, batch_moves in dataloader:
            batch_states, batch_moves = batch_states.to(device), batch_moves.to(device)
            outputs = model(batch_states)
            loss = criterion(outputs, batch_moves.squeeze())
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_moves.squeeze()).sum().item()
            total_predictions += batch_moves.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


# Example usage
if __name__ == "__main__":
    # HYPERPARAMETERS
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.005

    # Create dataset and dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=BATCH_SIZE, num_workers=4, val_split=0.2
    )

    # Initialize the model
    model = DecisionModel()

    # load the weights from last checkpoint if it exists
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))

    # Train the model
    trained_model = train_model(
        model, train_dataloader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )

    # Evaluate the model
    val_loss, val_accuracy = evaluate_model(trained_model, val_dataloader)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Log final validation metrics to wandb
    wandb.log(
        {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), "model.pth")

    # Log the model to wandb
    wandb.save("model.pth")

    # Finish the wandb run
    wandb.finish()
