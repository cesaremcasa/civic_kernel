"""Canonical state transition and physics API for Civic Kernel.

The engine owns the state transition.  The legacy ``kernel.physics`` module is
not imported here; it can only delegate to this module for compatibility.
Coordinates are represented as ``(row, column)`` pairs throughout the
simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

from kernel.state import GridState


@dataclass(frozen=True)
class Transition:
    """The validated effect of one action before it is applied."""

    success: bool
    reason: str
    delta_energy: float
    new_pos: tuple[int, int] | None = None
    consumed_pos: tuple[int, int] | None = None


class Action(IntEnum):
    """Actions accepted by :meth:`SimulationEngine.step`."""

    IDLE = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    GATHER = 5


class SimulationEngine:
    """Single authority for state, coordinates, energy, and terminal states."""

    def __init__(self, config: str | Path | Mapping[str, Any], seed: int | None = None):
        self.state = GridState(config, seed=seed)
        self.config = self.state.config
        self.COST_MOVE = float(self.config["COST_MOVE"])
        self.COST_GATHER = float(self.config["COST_GATHER"])
        self.MAX_ENERGY = float(self.config["MAX_ENERGY"])
        self.CHANNEL_EMPTY = self.config["CHANNEL_EMPTY"]
        self.CHANNEL_WALL = self.config["CHANNEL_WALL"]
        self.CHANNEL_RESOURCE = self.config["CHANNEL_RESOURCE"]
        self.ticks = 0

    @property
    def physics(self) -> "SimulationEngine":
        """Compatibility view for callers of the pre-v0.2 API.

        This property deliberately returns the canonical engine rather than
        importing or constructing a second physics implementation.
        """

        return self

    def reset(self) -> GridState:
        """Reset state and the tick counter using the configured seed."""

        self.state.reset()
        self.ticks = 0
        return self.state

    def calculate_move(self, dx: int, dy: int) -> Transition:
        """Validate a movement delta without mutating state.

        ``dx`` changes the row and ``dy`` changes the column.  The private
        alias remains for the legacy adapter.
        """

        if self.state.agent_energy < self.COST_MOVE:
            return Transition(False, "NO_ENERGY", 0.0)

        rows, columns = self.state.grid.shape
        row, column = self.state.agent_pos
        next_row, next_column = row + dx, column + dy
        if not (0 <= next_row < rows and 0 <= next_column < columns):
            return Transition(False, "OUT_OF_BOUNDS", 0.0)
        if self.state.grid[next_row, next_column] == self.CHANNEL_WALL:
            return Transition(False, "WALL", 0.0)
        return Transition(
            True,
            "OK",
            -self.COST_MOVE,
            new_pos=(next_row, next_column),
        )

    def _calculate_move(self, dx: int, dy: int) -> Transition:
        return self.calculate_move(dx, dy)

    def calculate_gather(self) -> Transition:
        """Validate gathering at the current cell without mutating state."""

        if self.state.agent_energy < self.COST_GATHER:
            return Transition(False, "NO_ENERGY", 0.0)

        row, column = self.state.agent_pos
        if self.state.grid[row, column] != self.CHANNEL_RESOURCE:
            return Transition(False, "NO_RESOURCE", 0.0)

        amount_available = max(0.0, float(self.state.resources[row, column]))
        energy_after_cost = self.state.agent_energy - self.COST_GATHER
        gain = min(amount_available, max(0.0, self.MAX_ENERGY - energy_after_cost))
        return Transition(
            True,
            "OK",
            -self.COST_GATHER + gain,
            consumed_pos=(row, column),
        )

    def _calculate_gather(self) -> Transition:
        return self.calculate_gather()

    @staticmethod
    def _coerce_action(action: Action | int) -> Action:
        try:
            return action if isinstance(action, Action) else Action(action)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unknown Action: {action}") from error

    def step(self, action: Action | int) -> tuple[GridState, float, bool, dict[str, str]]:
        """Apply one action and return ``(state, reward, done, info)``.

        Failed movement/gathering leaves state and energy unchanged, except a
        ``NO_ENERGY`` attempt transitions the engine to a terminal zero-energy
        state.  A successful action that reaches zero energy also terminates.
        """

        if self.state.agent_energy <= 0:
            return self.state, 0.0, True, {"reason": "ALREADY_DEAD"}

        action = self._coerce_action(action)
        if action == Action.MOVE_UP:
            transition = self.calculate_move(-1, 0)
        elif action == Action.MOVE_DOWN:
            transition = self.calculate_move(1, 0)
        elif action == Action.MOVE_LEFT:
            transition = self.calculate_move(0, -1)
        elif action == Action.MOVE_RIGHT:
            transition = self.calculate_move(0, 1)
        elif action == Action.GATHER:
            transition = self.calculate_gather()
        else:
            transition = Transition(True, "IDLE", 0.0)

        if transition.success:
            if transition.new_pos is not None:
                self.state.agent_pos = transition.new_pos
            if transition.consumed_pos is not None:
                row, column = transition.consumed_pos
                self.state.grid[row, column] = self.CHANNEL_EMPTY
                self.state.resources[row, column] = 0.0
            self.state.agent_energy = min(
                self.MAX_ENERGY,
                max(0.0, self.state.agent_energy + transition.delta_energy),
            )
            reward = transition.delta_energy
        else:
            reward = 0.0
            if transition.reason == "NO_ENERGY":
                self.state.agent_energy = 0.0
                return self.state, reward, True, {"reason": "NO_ENERGY"}

        if self.state.agent_energy <= 0.01:
            self.state.agent_energy = 0.0
            return self.state, reward, True, {"reason": "ENERGY_DEPLETED"}

        self.ticks += 1
        return self.state, reward, False, {"reason": transition.reason}
