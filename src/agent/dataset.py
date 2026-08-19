"""PyTorch dataset adapter for logged trajectories."""

from __future__ import annotations

import sqlite3

import numpy as np
import torch
from torch.utils.data import Dataset


class WorldDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Decode one-step trajectory records into normalized model tensors."""

    def __init__(self, db_path: str, grid_size: int = 20, num_actions: int = 6):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT id FROM trajectories ORDER BY id")
        self.ids = [row[0] for row in self.cursor.fetchall()]
        self.grid_size = grid_size
        self.num_actions = num_actions

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        row_id = self.ids[idx]
        self.cursor.execute(
            """
            SELECT grid, agent_pos_x, agent_pos_y, agent_energy, action,
                   next_grid, next_agent_energy
            FROM trajectories WHERE id=?
            """,
            (row_id,),
        )
        row = self.cursor.fetchone()
        if row is None:
            raise IndexError(idx)

        grid_blob, row_index, column_index, energy, action, next_grid_blob, next_energy = row
        shape = (self.grid_size, self.grid_size)
        grid = np.frombuffer(grid_blob, dtype=np.int64).reshape(shape).astype(np.float32)
        next_grid = np.frombuffer(next_grid_blob, dtype=np.int64).reshape(shape).astype(np.float32)

        # Grid channels are 0, 1, 2; energy is configured as 0..100.
        grid /= 2.0
        next_grid /= 2.0
        energy_normalized = float(energy) / 100.0
        next_energy_normalized = float(next_energy) / 100.0

        agent_map = np.zeros(shape, dtype=np.float32)
        agent_map[int(row_index), int(column_index)] = 1.0

        action_vec = np.zeros(self.num_actions, dtype=np.float32)
        action_vec[int(action)] = 1.0

        return (
            torch.from_numpy(grid),
            torch.from_numpy(agent_map),
            torch.from_numpy(action_vec),
            torch.from_numpy(next_grid),
            torch.tensor(next_energy_normalized, dtype=torch.float32),
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "WorldDataset":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
