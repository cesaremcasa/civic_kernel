"""Deterministic trajectory generation for the Civic Kernel."""

from __future__ import annotations

import random
from pathlib import Path

from simulation.engine import Action, SimulationEngine
from simulation.logger import TrajectoryLogger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "constants.yaml"


def _default_config() -> Path:
    """Return checkout configuration, or the config bundled in a wheel."""

    checkout_config = DEFAULT_CONFIG
    if checkout_config.is_file():
        return checkout_config

    from importlib.resources import files

    return Path(str(files("civic_kernel").joinpath("constants.yaml")))


def generate_data(
    num_episodes: int,
    db_path: str,
    seed: int = 0,
    config_path: str | Path | None = None,
) -> None:
    """Generate seeded trajectories in SQLite.

    ``seed`` controls both the action stream and each episode's initial state;
    episode ``n`` uses ``seed + n`` for its world state.  A fresh database path
    therefore produces byte-for-byte stable state/action data for a fixed
    seed.  Existing databases are appended to rather than deleted.
    """

    if num_episodes < 0:
        raise ValueError("num_episodes must be non-negative")

    logger = TrajectoryLogger(db_path)
    action_rng = random.Random(seed)
    constants_path = Path(config_path) if config_path is not None else _default_config()

    print(f"Generating {num_episodes} episodes...")
    print(f"Database: {db_path}")

    try:
        for episode_id in range(num_episodes):
            engine = SimulationEngine(constants_path, seed=seed + episode_id)
            step_count = 0

            while True:
                state_before = engine.state.snapshot()
                action = action_rng.choice(list(Action))
                next_state, reward, done, _info = engine.step(action)

                logger.log_step(
                    episode_id=episode_id,
                    step=step_count,
                    state=state_before,
                    action=action.value,
                    reward=reward,
                    done=done,
                    next_state=next_state,
                )
                step_count += 1

                if done:
                    logger.commit_episode()
                    break
    finally:
        logger.close()

    print(f"Data generation complete. Saved to {db_path}")


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    database = data_dir / "kernel_db.db"
    generate_data(num_episodes=1000, db_path=str(database), seed=0)
