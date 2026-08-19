"""CPU training and checkpoint reload helpers."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from agent.dataset import WorldDataset
from agent.models import WorldModelResNet


def train(
    db_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 0,
    checkpoint_path: str | Path = "data/checkpoints/kernel_model.pth",
) -> Path:
    """Train deterministically on trajectories and save a state-dict checkpoint."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    print(f"Device: {device}")

    full_dataset = WorldDataset(db_path)
    if len(full_dataset) == 0:
        full_dataset.close()
        raise ValueError("trajectory database contains no samples")

    train_size = max(1, int(0.9 * len(full_dataset)))
    val_size = len(full_dataset) - train_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, _val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    print(f"Train Samples: {len(train_dataset)}")

    model = WorldModelResNet(grid_size=20, num_actions=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Starting training (ResNet v2)...")
    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for grid, agent_map, action_vec, target_grid, target_energy in train_loader:
                grid = grid.unsqueeze(1).to(device)
                agent_map = agent_map.unsqueeze(1).to(device)
                action_vec = action_vec.to(device)
                target_grid = target_grid.unsqueeze(1).to(device)
                target_energy = target_energy.unsqueeze(1).to(device)

                optimizer.zero_grad()
                delta_grid_pred, next_energy_pred = model(grid, agent_map, action_vec)
                next_grid_pred = grid + delta_grid_pred
                loss = criterion(next_grid_pred, target_grid) + criterion(
                    next_energy_pred, target_energy
                )
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())

            average_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {average_loss:.4f}")
    finally:
        full_dataset.close()

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f"Model saved: {checkpoint}")
    return checkpoint


def load_model(checkpoint_path: str | Path, grid_size: int = 20, num_actions: int = 6) -> nn.Module:
    """Reload a state-dict checkpoint through the canonical model definition."""

    device = torch.device("cpu")
    model = WorldModelResNet(grid_size=grid_size, num_actions=num_actions).to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # compatibility with older supported PyTorch releases
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    train(db_path="data/kernel_db.db", epochs=20, seed=0)
