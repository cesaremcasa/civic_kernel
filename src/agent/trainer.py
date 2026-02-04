import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from agent.dataset import WorldDataset
from agent.models import WorldModelResNet

def train(db_path, epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cpu")
    print(f"Device: {device}")

    full_dataset = WorldDataset(db_path)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Train Samples: {len(train_dataset)}")

    model = WorldModelResNet(grid_size=20, num_actions=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Starting Training (ResNet v2)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (grid, agent_map, action_vec, target_grid, target_energy) in enumerate(train_loader):
            grid = grid.unsqueeze(1).to(device)
            agent_map = agent_map.unsqueeze(1).to(device)
            action_vec = action_vec.to(device)
            target_grid = target_grid.unsqueeze(1).to(device)
            target_energy = target_energy.unsqueeze(1).to(device)

            optimizer.zero_grad()

            # Forward: Predict DELTA
            delta_grid_pred, next_energy_pred = model(grid, agent_map, action_vec)

            # Reconstruction: S_t + Delta
            next_grid_pred = grid + delta_grid_pred

            # Loss
            loss_grid = criterion(next_grid_pred, target_grid)
            loss_energy = criterion(next_energy_pred, target_energy)
            loss = loss_grid + loss_energy

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    # Save
    Path("data/checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "data/checkpoints/kernel_model.pth")
    print("Model Saved.")

if __name__ == "__main__":
    train(db_path="data/kernel_db.db", epochs=20)
