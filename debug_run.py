import sys
import time
from pathlib import Path

# Clean path setup
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

print("1. Importing SimulationEngine...")
from simulation.engine import SimulationEngine

print("2. Initializing Engine (this is where it usually hangs)...")
config_path = str(PROJECT_ROOT / 'config' / 'constants.yaml')
print(f"   Config Path: {config_path}")

try:
    engine = SimulationEngine(config_path)
    print("   Engine Initialized Successfully.")
except Exception as e:
    print(f"   ERROR initializing: {e}")
    sys.exit(1)

print("3. Starting Step Loop...")
t = time.time()
c = 0
done = False

# Safety breaker
MAX_STEPS = 1000

while c < MAX_STEPS:
    # Force print to buffer
    if c % 100 == 0:
        print(f"   Step {c}, Energy: {engine.state.agent_energy}")
    
    ns, r, done, info = engine.step(1) # Move UP repeatedly
    c += 1
    
    if done:
        print(f"   Simulation Done. Reason: {info.get('reason')}")
        break

elapsed = time.time() - t
print(f"4. Finished. {c} steps in {elapsed:.4f}s")
