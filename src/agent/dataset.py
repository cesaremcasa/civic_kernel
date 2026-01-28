import torch
from torch.utils.data import Dataset
import sqlite3
import numpy as np

class WorldDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT id FROM trajectories")
        self.ids = [row[0] for row in self.cursor.fetchall()]
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        row_id = self.ids[idx]
        self.cursor.execute("SELECT grid, agent_pos_x, agent_pos_y, agent_energy, action, next_grid, next_agent_energy FROM trajectories WHERE id=?", (row_id,))
        row = self.cursor.fetchone()
        
        grid_blob, ax, ay, ae, action, next_grid_blob, nae = row
        
        # 1. Decode Grids
        grid = np.frombuffer(grid_blob, dtype=np.int64).reshape(20, 20).astype(np.float32)
        next_grid = np.frombuffer(next_grid_blob, dtype=np.int64).reshape(20, 20).astype(np.float32)
        
        # 2. NORMALIZATION (The Fix)
        # Grid: 0, 1, 2 -> 0.0, 0.5, 1.0
        grid = grid / 2.0
        next_grid = next_grid / 2.0
        
        # Energy: 0 to 100 -> 0.0 to 1.0
        ae_norm = ae / 100.0
        nae_norm = nae / 100.0
        
        # 3. Create Agent Heatmap
        agent_map = np.zeros((20, 20), dtype=np.float32)
        agent_map[int(ax), int(ay)] = 1.0
        
        # 4. One-hot Action
        action_vec = np.zeros(6, dtype=np.float32)
        action_vec[action] = 1.0
        
        return (
            torch.tensor(grid),           # Input (20, 20) Normalized
            torch.tensor(agent_map),       # Input (20, 20)
            torch.tensor(action_vec),      # Input (6)
            torch.tensor(next_grid),       # Target (20, 20) Normalized
            torch.tensor(nae_norm, dtype=torch.float32) # Target (1) Normalized
        )
