import yaml
import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path

class GridState:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        self.GRID_SIZE = self.config['GRID_SIZE']
        self.CHANNEL_EMPTY = self.config['CHANNEL_EMPTY']
        self.CHANNEL_WALL = self.config['CHANNEL_WALL']
        self.CHANNEL_RESOURCE = self.config['CHANNEL_RESOURCE']
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.resources = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=float)
        
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.agent_energy = 0.0
        
        self.reset()

    def reset(self) -> None:
        self.grid.fill(self.CHANNEL_EMPTY)
        self.resources.fill(0.0)
        
        w = self.GRID_SIZE
        
        # Place Walls (Random 10%)
        wx, wy = np.random.randint(0, w, int(w*w*0.1)), np.random.randint(0, w, int(w*w*0.1))
        self.grid[wx, wy] = self.CHANNEL_WALL
        
        # Place Resources (Random 5% on Empty spots)
        rx, ry = np.random.randint(0, w, int(w*w*0.05)), np.random.randint(0, w, int(w*w*0.05))
        mask = self.grid[rx, ry] == self.CHANNEL_EMPTY
        self.grid[rx[mask], ry[mask]] = self.CHANNEL_RESOURCE
        self.resources[rx[mask], ry[mask]] = np.random.uniform(10.0, 50.0, size=np.sum(mask))
        
        # Place Agent
        empty_spots = np.argwhere(self.grid == self.CHANNEL_EMPTY)
        if len(empty_spots) > 0:
            # CRITICAL FIX: Cast to int() to avoid np.int64 serialization issues
            raw_pos = empty_spots[np.random.randint(0, len(empty_spots))]
            self.agent_pos = (int(raw_pos[0]), int(raw_pos[1]))
        
        self.agent_energy = 50.0

    def get_shape(self) -> Tuple[int, int]:
        return self.grid.shape
