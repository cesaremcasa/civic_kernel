import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation.engine import Action, SimulationEngine, Transition


CONFIG = Path(__file__).parent.parent / "config" / "constants.yaml"


def test_canonical_engine_contract():
    engine = SimulationEngine(CONFIG, seed=7)
    assert isinstance(engine.calculate_move(0, 0), Transition)
    assert engine.state.get_shape() == (20, 20)


def test_move_success_deducts_energy():
    engine = SimulationEngine(CONFIG, seed=7)
    engine.state.grid.fill(engine.CHANNEL_EMPTY)
    engine.state.agent_pos = (5, 5)
    start_energy = engine.state.agent_energy

    transition = engine.calculate_move(0, 1)
    assert transition.success is True
    assert transition.delta_energy == -engine.COST_MOVE
    assert transition.new_pos == (5, 6)

    state, reward, done, info = engine.step(Action.MOVE_RIGHT)
    assert state.agent_pos == (5, 6)
    assert state.agent_energy == start_energy - engine.COST_MOVE
    assert reward == -engine.COST_MOVE
    assert done is False
    assert info["reason"] == "OK"


def test_move_into_wall_fails_without_mutation():
    engine = SimulationEngine(CONFIG, seed=7)
    engine.state.agent_pos = (5, 5)
    engine.state.grid[5, 6] = engine.CHANNEL_WALL
    start_energy = engine.state.agent_energy

    transition = engine.calculate_move(0, 1)
    assert transition.success is False
    assert transition.reason == "WALL"
    assert transition.delta_energy == 0.0

    state, reward, done, info = engine.step(Action.MOVE_RIGHT)
    assert state.agent_pos == (5, 5)
    assert state.agent_energy == start_energy
    assert reward == 0.0
    assert done is False
    assert info["reason"] == "WALL"


def test_gather_resource():
    engine = SimulationEngine(CONFIG, seed=7)
    row, column = engine.state.agent_pos
    engine.state.resources[row, column] = 20.0
    engine.state.grid[row, column] = engine.CHANNEL_RESOURCE

    transition = engine.calculate_gather()
    assert transition.success is True
    assert transition.consumed_pos == (row, column)
    assert transition.delta_energy == -engine.COST_GATHER + 20.0

    _, reward, done, info = engine.step(Action.GATHER)
    assert reward == transition.delta_energy
    assert done is False
    assert info["reason"] == "OK"
    assert engine.state.grid[row, column] == engine.CHANNEL_EMPTY
    assert engine.state.resources[row, column] == 0.0


def test_energy_floor_blocks_move():
    engine = SimulationEngine(CONFIG, seed=7)
    engine.state.agent_energy = 0.0
    transition = engine.calculate_move(0, 1)
    assert transition.success is False
    assert transition.reason == "NO_ENERGY"
