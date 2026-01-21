import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, Any
from kernel.state import GridState

class TrajectoryLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                step_count INTEGER,
                grid BLOB,
                agent_pos_x INTEGER,
                agent_pos_y INTEGER,
                agent_energy REAL,
                action INTEGER,
                reward REAL,
                done BOOLEAN,
                next_grid BLOB,
                next_agent_pos_x INTEGER,
                next_agent_pos_y INTEGER,
                next_agent_energy REAL
            )
        ''')
        self.conn.commit()

    def log_step(self, episode_id: int, step: int, state: GridState, action: int, 
                 reward: float, done: bool, next_state: GridState):
        cursor = self.conn.cursor()
        
        # FIX: Use NumPy's native tobytes() instead of pickle.dumps()
        # This is much faster and less prone to hanging issues.
        grid_blob = sqlite3.Binary(state.grid.tobytes())
        next_grid_blob = sqlite3.Binary(next_state.grid.tobytes())
        
        cursor.execute('''
            INSERT INTO trajectories 
            (episode_id, step_count, grid, agent_pos_x, agent_pos_y, agent_energy, 
             action, reward, done, next_grid, next_agent_pos_x, next_agent_pos_y, next_agent_energy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            episode_id, step, grid_blob, state.agent_pos[0], state.agent_pos[1], state.agent_energy,
            action, reward, done, next_grid_blob, next_state.agent_pos[0], next_state.agent_pos[1], next_state.agent_energy
        ))

    def commit_episode(self):
        self.conn.commit()

    def close(self):
        self.conn.close()
