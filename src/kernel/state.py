"""Deterministic grid state for the Civic Kernel simulation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


class GridState:
    """Mutable simulation state with a reproducible seeded reset.

    Positions are ``(row, column)`` pairs and grid arrays use the same order.
    ``snapshot``/``copy`` provide an independent value for trajectory logging
    before the engine mutates the live state.
    """

    def __init__(self, config: str | Path | Mapping[str, Any], seed: int | None = None):
        if isinstance(config, Mapping):
            self.config = dict(config)
        else:
            with Path(config).open(encoding="utf-8") as config_file:
                loaded = yaml.safe_load(config_file)
            if not isinstance(loaded, Mapping):
                raise ValueError("configuration must contain a mapping")
            self.config = dict(loaded)

        self.GRID_SIZE = int(self.config["GRID_SIZE"])
        self.CHANNEL_EMPTY = int(self.config["CHANNEL_EMPTY"])
        self.CHANNEL_WALL = int(self.config["CHANNEL_WALL"])
        self.CHANNEL_RESOURCE = int(self.config["CHANNEL_RESOURCE"])
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int64)
        self.resources = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float64)
        self.agent_pos: tuple[int, int] = (0, 0)
        self.agent_energy = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset the world to the same initial state for a fixed seed."""

        self._rng = np.random.default_rng(self.seed)
        self.grid.fill(self.CHANNEL_EMPTY)
        self.resources.fill(0.0)

        wall_count = int(self.GRID_SIZE * self.GRID_SIZE * 0.1)
        wall_rows = self._rng.integers(0, self.GRID_SIZE, wall_count)
        wall_columns = self._rng.integers(0, self.GRID_SIZE, wall_count)
        self.grid[wall_rows, wall_columns] = self.CHANNEL_WALL

        resource_count = int(self.GRID_SIZE * self.GRID_SIZE * 0.05)
        resource_rows = self._rng.integers(0, self.GRID_SIZE, resource_count)
        resource_columns = self._rng.integers(0, self.GRID_SIZE, resource_count)
        resource_mask = self.grid[resource_rows, resource_columns] == self.CHANNEL_EMPTY
        self.grid[resource_rows[resource_mask], resource_columns[resource_mask]] = (
            self.CHANNEL_RESOURCE
        )
        self.resources[resource_rows[resource_mask], resource_columns[resource_mask]] = (
            self._rng.uniform(10.0, 50.0, size=int(resource_mask.sum()))
        )

        empty_spots = np.argwhere(self.grid == self.CHANNEL_EMPTY)
        if len(empty_spots) == 0:
            raise ValueError("grid has no valid cell for the agent")
        raw_pos = empty_spots[int(self._rng.integers(0, len(empty_spots)))]
        self.agent_pos = (int(raw_pos[0]), int(raw_pos[1]))
        self.agent_energy = 50.0

    def snapshot(self) -> "GridState":
        """Return an independent copy suitable for before/after logging."""

        state = object.__new__(GridState)
        state.config = dict(self.config)
        state.GRID_SIZE = self.GRID_SIZE
        state.CHANNEL_EMPTY = self.CHANNEL_EMPTY
        state.CHANNEL_WALL = self.CHANNEL_WALL
        state.CHANNEL_RESOURCE = self.CHANNEL_RESOURCE
        state.seed = self.seed
        state._rng = np.random.default_rng()
        state._rng.bit_generator.state = deepcopy(self._rng.bit_generator.state)
        state.grid = self.grid.copy()
        state.resources = self.resources.copy()
        state.agent_pos = self.agent_pos
        state.agent_energy = self.agent_energy
        return state

    def copy(self) -> "GridState":
        """Alias for :meth:`snapshot`."""

        return self.snapshot()

    def get_shape(self) -> tuple[int, int]:
        return int(self.grid.shape[0]), int(self.grid.shape[1])
