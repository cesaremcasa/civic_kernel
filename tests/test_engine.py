import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from simulation.engine import SimulationEngine, Action

def test_engine_initialization():
    sim = SimulationEngine(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    assert sim.state.agent_energy == 50.0
    assert sim.ticks == 0

def test_step_move_application():
    sim = SimulationEngine(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    start_energy = sim.state.agent_energy
    
    # Move Right
    new_state, reward, done, info = sim.step(Action.MOVE_RIGHT)
    
    # We assume the agent wasn't against the right wall (99% chance)
    # If they hit a wall, reward is 0 and pos doesn't change.
    # If they moved, reward is -Cost.
    
    if info['reason'] == 'OK':
        assert new_state.agent_energy == start_energy - sim.COST_MOVE
        assert reward == -sim.COST_MOVE
    else:
        # Hit wall or edge
        assert new_state.agent_energy == start_energy
        
    assert sim.ticks == 1

def test_step_gather_application():
    sim = SimulationEngine(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    
    # Force resource under agent
    sim.state.grid[sim.state.agent_pos] = sim.physics.config['CHANNEL_RESOURCE']
    sim.state.resources[sim.state.agent_pos] = 20.0
    
    start_energy = sim.state.agent_energy
    
    new_state, reward, done, info = sim.step(Action.GATHER)
    
    assert sim.state.grid[sim.state.agent_pos] == sim.physics.config['CHANNEL_EMPTY']
    expected_energy = start_energy - sim.physics.COST_GATHER + 20.0
    assert new_state.agent_energy == expected_energy
    assert info['reason'] == 'OK'

def test_terminal_state_energy_zero():
    sim = SimulationEngine(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    sim.state.agent_energy = 0.0
    
    new_state, reward, done, info = sim.step(Action.MOVE_RIGHT)
    
    # Should be done immediately
    assert done == True
    assert info['reason'] == 'ALREADY_DEAD'

def test_reset_cycle():
    sim = SimulationEngine(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    
    sim.state.agent_energy = 0.0
    sim.ticks = 100
    
    sim.reset()
    
    assert sim.state.agent_energy == 50.0
    assert sim.ticks == 0
