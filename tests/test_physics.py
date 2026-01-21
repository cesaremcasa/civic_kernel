import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kernel.state import GridState
from kernel.physics import PhysicsEngine, Transition

def test_validate_normal_state():
    state = GridState(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    physics = PhysicsEngine(state.config)
    assert physics.validate_state(state) == True

def test_move_success_deducts_energy():
    state = GridState(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    physics = PhysicsEngine(state.config)
    
    start_x, start_y = state.agent_pos
    
    # Find a valid move direction to avoid edge cases
    valid_move = None
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = start_x + dx, start_y + dy
        if (0 <= nx < state.GRID_SIZE and 0 <= ny < state.GRID_SIZE and 
            state.grid[nx, ny] != physics.config['CHANNEL_WALL']):
            valid_move = (dx, dy)
            break
            
    assert valid_move is not None, "Agent is trapped in a corner/wall by bad RNG"
    
    trans = physics.calculate_move(state, valid_move[0], valid_move[1])
    
    assert trans.success == True
    assert trans.delta_energy == -physics.COST_MOVE
    assert trans.new_pos == (start_x + valid_move[0], start_y + valid_move[1])

def test_move_into_wall_fails():
    state = GridState(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    physics = PhysicsEngine(state.config)
    
    x, y = state.agent_pos
    # Find a neighbor to turn into a wall
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < state.GRID_SIZE and 0 <= ny < state.GRID_SIZE:
            state.grid[nx, ny] = physics.config['CHANNEL_WALL']
            trans = physics.calculate_move(state, dx, dy)
            assert trans.success == False
            assert trans.reason == 'WALL'
            assert trans.delta_energy == 0.0 
            return # Done

def test_gather_resource():
    state = GridState(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    physics = PhysicsEngine(state.config)
    
    res_amt = 20.0
    state.resources[state.agent_pos] = res_amt
    state.grid[state.agent_pos] = physics.config['CHANNEL_RESOURCE']
    
    trans = physics.calculate_gather(state)
    
    assert trans.success == True
    assert trans.consumed_pos == state.agent_pos
    assert trans.delta_energy == (-physics.COST_GATHER + res_amt)

def test_energy_floor_blocks_move():
    state = GridState(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    physics = PhysicsEngine(state.config)
    state.agent_energy = 0.0
    
    trans = physics.calculate_move(state, 0, 1)
    assert trans.success == False
    assert trans.reason == 'NO_ENERGY'
