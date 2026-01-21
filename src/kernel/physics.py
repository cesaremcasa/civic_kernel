from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING
import numpy as np

# Direct import to avoid string hint resolution issues on M2
if TYPE_CHECKING:
    from .state import GridState

@dataclass
class Transition:
    success: bool
    reason: str
    delta_energy: float
    new_pos: Optional[Tuple[int, int]] = None
    consumed_pos: Optional[Tuple[int, int]] = None

class PhysicsEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.COST_MOVE = self.config['COST_MOVE']
        self.COST_GATHER = self.config['COST_GATHER']
        self.MAX_ENERGY = self.config['MAX_ENERGY']
        self.CHANNEL_EMPTY = self.config['CHANNEL_EMPTY']
        self.CHANNEL_WALL = self.config['CHANNEL_WALL']
        self.CHANNEL_RESOURCE = self.config['CHANNEL_RESOURCE']

    def validate_state(self, state: 'GridState') -> bool:
        if state.agent_energy < 0: return False
        if state.agent_energy > self.MAX_ENERGY: return False
        
        h, w = state.grid.shape
        x, y = state.agent_pos
        if not (0 <= x < h and 0 <= y < w): return False
        return True

    # REMOVED 'GridState' string hint, replaced with direct object or Any for now
    def calculate_move(self, state: Any, dx: int, dy: int) -> Transition:
        if state.agent_energy < self.COST_MOVE:
            return Transition(False, 'NO_ENERGY', 0.0)

        h, w = state.grid.shape
        cx, cy = state.agent_pos
        nx, ny = cx + dx, cy + dy

        if not (0 <= nx < h and 0 <= ny < w):
            return Transition(False, 'OUT_OF_BOUNDS', 0.0)

        if state.grid[nx, ny] == self.CHANNEL_WALL:
            return Transition(False, 'WALL', 0.0)

        delta_e = -self.COST_MOVE
        return Transition(True, 'OK', delta_e, new_pos=(nx, ny))

    def calculate_gather(self, state: Any) -> Transition:
        if state.agent_energy < self.COST_GATHER:
            return Transition(False, 'NO_ENERGY', 0.0)

        x, y = state.agent_pos
        
        if state.grid[x, y] != self.CHANNEL_RESOURCE:
            return Transition(False, 'NO_RESOURCE', 0.0)

        amount_available = state.resources[x, y]
        
        delta_e = -self.COST_GATHER + amount_available
        
        if state.agent_energy + delta_e > self.MAX_ENERGY:
            actual_gain = self.MAX_ENERGY - state.agent_energy
            delta_e = -self.COST_GATHER + actual_gain

        return Transition(True, 'OK', delta_e, consumed_pos=(x, y))
