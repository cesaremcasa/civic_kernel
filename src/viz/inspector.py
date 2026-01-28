import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.dataset import WorldDataset
from agent.models import WorldModelCNN

def visualize():
    db_path = "data/checkpoints/../kernel_db.db" # Path relative to script parent
    # Fix path for direct execution
    db_path = "data/kernel_db.db"
    
    dataset = WorldDataset(db_path)
    
    device = torch.device("cpu")
    model = WorldModelCNN().to(device)
    
    # FIX: Initialize lazy layers with dummy data before loading weights
    dummy_grid = torch.rand(1, 1, 20, 20)
    dummy_agent = torch.rand(1, 1, 20, 20)
    dummy_action = torch.rand(1, 6)
    _ = model(dummy_grid, dummy_agent, dummy_action)
    
    # Now load weights
    model.load_state_dict(torch.load("data/checkpoints/kernel_model.pth", map_location=device))
    model.eval()
    
    print("Model Loaded. Visualizing 5 samples...")
    
    chars = {0.0: '.', 0.5: '#', 1.0: '$'}
    
    for i in range(5):
        grid, agent_map, action, target_grid, target_energy = dataset[i]
        
        grid_in = grid.unsqueeze(0).unsqueeze(0).to(device)
        agent_in = agent_map.unsqueeze(0).unsqueeze(0).to(device)
        action_in = action.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_grid, pred_energy = model(grid_in, agent_in, action_in)
        
        pred_grid_np = pred_grid.squeeze().cpu().numpy() * 2.0
        target_grid_np = target_grid.numpy() * 2.0
        
        pred_e_val = pred_energy.item() * 100.0
        real_e_val = target_energy.item() * 100.0
        
        print(f"\n--- Sample {i+1} (Action: {action.argmax().item()}) ---")
        print(f"Energy: Real={real_e_val:.1f} | Pred={pred_e_val:.1f}")
        
        print("Real Grid | Pred Grid")
        for r in range(20):
            line_real = ""
            line_pred = ""
            for c in range(20):
                val_real = target_grid_np[r, c]
                val_pred = pred_grid_np[r, c]
                char_real = chars.get(round(val_real, 1), '?')
                char_pred = chars.get(round(val_pred, 1), '?')
                line_real += char_real + " "
                line_pred += char_pred + " "
            print(f"{line_real} | {line_pred}")

if __name__ == "__main__":
    visualize()
