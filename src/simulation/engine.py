from enum import IntEnum
from typing import Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass

# Import State directly
from kernel.state import GridState

@dataclass
class Transition:
    success: bool
    reason: str
    delta_energy: float
    new_pos: Tuple[int, int] = None
    consumed_pos: Tuple[int, int] = None

class Action(IntEnum):
    IDLE = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    GATHER = 5

class SimulationEngine:
    """
    Unified Engine: Handles State Management AND Physics Calculation.
    Eliminates circular import risks.
    """
    def __init__(self, config_path: str):
        # 1. Initialize State
        self.state = GridState(config_path)
        
        # 2. Load Physics Constants
        self.COST_MOVE = self.state.config['COST_MOVE']
        self.COST_GATHER = self.state.config['COST_GATHER']
        self.MAX_ENERGY = self.state.config['MAX_ENERGY']
        self.CHANNEL_EMPTY = self.state.config['CHANNEL_EMPTY']
        self.CHANNEL_WALL = self.state.config['CHANNEL_WALL']
        self.CHANNEL_RESOURCE = self.state.config['CHANNEL_RESOURCE']
        
        self.ticks = 0

    def reset(self) -> GridState:
        self.state.reset()
        self.ticks = 0
        return self.state

    def _calculate_move(self, dx: int, dy: int) -> Transition:
        """Internal Physics Logic for Movement"""
        # Check Energy
        if self.state.agent_energy < self.COST_MOVE:
            return Transition(False, 'NO_ENERGY', 0.0)

        h, w = self.state.grid.shape
        cx, cy = self.state.agent_pos
        nx, ny = cx + dx, cy + dy

        # Check Bounds
        if not (0 <= nx < h and 0 <= ny < w):
            return Transition(False, 'OUT_OF_BOUNDS', 0.0)

        # Check Walls
        if self.state.grid[nx, ny] == self.CHANNEL_WALL:
            return Transition(False, 'WALL', 0.0)

        return Transition(True, 'OK', -self.COST_MOVE, new_pos=(nx, ny))

    def _calculate_gather(self) -> Transition:
        """Internal Physics Logic for Gathering"""
        if self.state.agent_energy < self.COST_GATHER:
            return Transition(False, 'NO_ENERGY', 0.0)

        x, y = self.state.agent_pos
        
        if self.state.grid[x, y] != self.CHANNEL_RESOURCE:
            return Transition(False, 'NO_RESOURCE', 0.0)

        amount_available = self.state.resources[x, y]
        delta_e = -self.COST_GATHER + amount_available
        
        # Cap at Max
        if self.state.agent_energy + delta_e > self.MAX_ENERGY:
            actual_gain = self.MAX_ENERGY - self.state.agent_energy
            delta_e = -self.COST_GATHER + actual_gain

        return Transition(True, 'OK', delta_e, consumed_pos=(x, y))

    def step(self, action: Action) -> Tuple[GridState, float, bool, Dict]:
        trans = Transition(False, 'IDLE', 0.0)
        
        # Check Dead State
        if self.state.agent_energy <= 0:
            return self.state, 0.0, True, {"reason": "ALREADY_DEAD"}

        # Calculate Physics
        if action == Action.MOVE_UP:
            trans = self._calculate_move(0, -1)
        elif action == Action.MOVE_DOWN:
            trans = self._calculate_move(0, 1)
        elif action == Action.MOVE_LEFT:
            trans = self._calculate_move(-1, 0)
        elif action == Action.MOVE_RIGHT:
            trans = self._calculate_move(1, 0)
        elif action == Action.GATHER:
            trans = self._calculate_gather()
        elif action == Action.IDLE:
            trans = Transition(True, 'IDLE', 0.0)
        else:
            raise ValueError(f"Unknown Action: {action}")

        # Apply Transition
        reward = 0.0
        
        if trans.success:
            if trans.new_pos:
                self.state.agent_pos = trans.new_pos
            if trans.consumed_pos:
                cx, cy = trans.consumed_pos
                self.state.grid[cx, cy] = self.CHANNEL_EMPTY
                self.state.resources[cx, cy] = 0.0
            
            self.state.agent_energy += trans.delta_energy
            reward = trans.delta_energy
        else:
            if trans.reason == 'NO_ENERGY':
                self.state.agent_energy = 0.0
                return self.state, 0.0, True, {"reason": "NO_ENERGY"}
            reward = 0.0

        # Check Death
        if self.state.agent_energy <= 0.01:
            self.state.agent_energy = 0.0
            return self.state, reward, True, {"reason": "ENERGY_DEPLETED"}

        self.ticks += 1
        return self.state, reward, False, {"reason": trans.reason}
