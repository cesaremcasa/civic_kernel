import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.dataset import WorldDataset
from agent.models import WorldModelResNet

def visualize():
    db_path = "data/kernel_db.db"
    dataset = WorldDataset(db_path)

    device = torch.device("cpu")
    model = WorldModelResNet(grid_size=20, num_actions=6).to(device)
    model.load_state_dict(torch.load("data/checkpoints/kernel_model.pth", map_location=device))
    model.eval()

    print("Model Loaded. Visualizing 5 samples...")

    for i in range(5):
        grid, agent_map, action_vec, target_grid, target_energy = dataset[i]

        grid_in = grid.unsqueeze(0).unsqueeze(0).to(device)
        agent_in = agent_map.unsqueeze(0).unsqueeze(0).to(device)
        action_in = action_vec.unsqueeze(0).to(device)

        with torch.no_grad():
            delta_grid_pred, next_energy_pred = model(grid_in, agent_in, action_in)

        # Reconstruction
        next_grid_pred = grid_in + delta_grid_pred

        # Denormalize (0-1 -> 0-2)
        pred_grid_np = next_grid_pred.squeeze().cpu().numpy() * 2.0
        target_grid_np = target_grid.numpy() * 2.0

        pred_e_val = next_energy_pred.item() * 100.0
        real_e_val = target_energy.item() * 100.0

        print(f"\n--- Sample {i+1} (Action: {action_vec.argmax().item()}) ---")
        print(f"Energy: Real={real_e_val:.1f} | Pred={pred_e_val:.1f}")
        print("Real Grid | Pred Grid")

        for r in range(20):
            line_real = ""
            line_pred = ""
            for c in range(20):
                val_real = target_grid_np[r, c]
                val_pred = pred_grid_np[r, c]
                
                # Discretize Logic
                def get_char(v):
                    if v < 0.25: return '.'
                    if v < 0.85: return '#'  # Catches Walls (0.5) and partials
                    return '$'              # Catches Resources (1.0) and partials

                char_real = get_char(val_real)
                char_pred = get_char(val_pred)
                
                line_real += char_real + " "
                line_pred += char_pred + " "
            print(f"{line_real} | {line_pred}")

if __name__ == "__main__":
    visualize()
