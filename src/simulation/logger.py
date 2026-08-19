"""SQLite trajectory persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from kernel.state import GridState


class TrajectoryLogger:
    def __init__(self, db_path: str | Path):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

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

    def log_step(
        self,
        episode_id: int,
        step: int,
        state: GridState,
        action: int,
        reward: float,
        done: bool,
        next_state: GridState,
    ) -> None:
        cursor = self.conn.cursor()
        
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

    def close(self) -> None:
        self.conn.close()
