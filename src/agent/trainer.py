import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.dataset import WorldDataset
from agent.models import WorldModelCNN

def train(db_path, epochs=10, batch_size=64, lr=0.001):
    device = torch.device("cpu")
    print(f"Device: {device}")

    full_dataset = WorldDataset(db_path)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    model = WorldModelCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # FIX: Combined Loss (MSE + L1)
    # MSE teaches general position, L1 forces sharpness (less blurring)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss() 

    print("Starting Training with L1 Regularization...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (grid, agent_map, action, target_grid, target_energy) in enumerate(train_loader):
            grid = grid.unsqueeze(1).to(device)
            agent_map = agent_map.unsqueeze(1).to(device)
            action = action.to(device)
            target_grid = target_grid.unsqueeze(1).to(device)
            target_energy = target_energy.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            pred_grid, pred_energy = model(grid, agent_map, action)
            
            # Combine Losses
            loss_grid_mse = criterion_mse(pred_grid, target_grid)
            loss_grid_l1 = criterion_l1(pred_grid, target_grid)
            loss_energy = criterion_mse(pred_energy, target_energy)
            
            # Weight L1 higher to fight blur
            loss = loss_grid_mse + (2.0 * loss_grid_l1) + loss_energy
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    checkpoint_path = "data/checkpoints/kernel_model.pth"
    Path(checkpoint_path).parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model Saved to {checkpoint_path}")

if __name__ == "__main__":
    train(db_path="data/kernel_db.db", epochs=10)
