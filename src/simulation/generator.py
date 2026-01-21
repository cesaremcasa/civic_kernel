import sys
import random
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
except:
    PROJECT_ROOT = Path(__file__).parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulation.engine import SimulationEngine, Action
from simulation.logger import TrajectoryLogger

def generate_data(num_episodes: int, db_path: str):
    logger = TrajectoryLogger(db_path)
    
    print(f"Generating {num_episodes} episodes...")
    print(f"Database: {db_path}")
    
    for ep in range(num_episodes):
        config_path = PROJECT_ROOT / "config" / "constants.yaml"
        engine = SimulationEngine(str(config_path))
        
        step_count = 0
        
        while True:
            current_state_snapshot = engine.state 
            action = random.choice(list(Action))
            
            next_state, reward, done, info = engine.step(action)
            
            logger.log_step(
                episode_id=ep,
                step=step_count,
                state=current_state_snapshot,
                action=action.value,
                reward=reward,
                done=done,
                next_state=next_state
            )
            
            step_count += 1
            
            if done:
                # FLUSH: Write the entire episode to disk now
                logger.commit_episode()
                break
                
    logger.close()
    print(f"Data generation complete. Saved to {db_path}")

if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    
    DB_FILE = str(data_dir / "kernel_db.db")
    
    # Delete old DB to ensure clean state
    if Path(DB_FILE).exists():
        Path(DB_FILE).unlink()

    generate_data(num_episodes=1000, db_path=DB_FILE)
