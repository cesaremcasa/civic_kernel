import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kernel.state import GridState
def test_grid_shape():
    state = GridState(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    assert state.grid.shape == (20, 20)
def test_agent_initial_energy():
    state = GridState(str(Path(__file__).parent.parent / "config" / "constants.yaml"))
    assert state.agent_energy == 50.0
