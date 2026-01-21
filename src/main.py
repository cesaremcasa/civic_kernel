import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kernel.state import GridState
def main():
    p = Path(__file__).parent.parent / "config" / "constants.yaml"
    s = GridState(str(p))
    print(f"Grid Shape: {s.get_shape()}")
    print(f"Agent Pos: {s.agent_pos}")
    print(f"Energy: {s.agent_energy}")
    print("Phase 1 OK.")
if __name__ == "__main__": main()
