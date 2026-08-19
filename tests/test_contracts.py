from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import torch

from civic_kernel import Action, SimulationEngine, Transition
from simulation.generator import generate_data
from agent.trainer import load_model, train


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/constants.yaml"


def test_canonical_exports_and_seeded_reset():
    first = SimulationEngine(CONFIG, seed=19)
    second = SimulationEngine(CONFIG, seed=19)
    assert Action.MOVE_RIGHT.value == 4
    assert Transition.__module__ == "simulation.engine"
    assert first.state.agent_pos == second.state.agent_pos
    np.testing.assert_array_equal(first.state.grid, second.state.grid)
    np.testing.assert_allclose(first.state.resources, second.state.resources)

    first.step(Action.IDLE)
    first.reset()
    np.testing.assert_array_equal(first.state.grid, second.state.grid)
    assert first.state.agent_pos == second.state.agent_pos


def test_coordinates_and_gather_cap_follow_main_engine_rules():
    engine = SimulationEngine(CONFIG, seed=19)
    engine.state.grid.fill(engine.CHANNEL_EMPTY)
    engine.state.resources.fill(0.0)
    engine.state.agent_pos = (1, 1)
    engine.state.agent_energy = 10.0

    state, reward, done, info = engine.step(Action.MOVE_UP)
    assert state.agent_pos == (0, 1)
    assert reward == -engine.COST_MOVE
    assert done is False and info["reason"] == "OK"

    state.agent_pos = (1, 1)
    state.grid[1, 1] = engine.CHANNEL_RESOURCE
    state.resources[1, 1] = 50.0
    state.agent_energy = engine.MAX_ENERGY - 0.5
    _, reward, done, info = engine.step(Action.GATHER)
    assert state.agent_energy == engine.MAX_ENERGY
    assert reward == 0.5
    assert done is False and info["reason"] == "OK"
    assert state.grid[1, 1] == engine.CHANNEL_EMPTY


def test_relevant_failure_is_terminal_and_does_not_advance_ticks():
    engine = SimulationEngine(CONFIG, seed=3)
    engine.state.agent_energy = 0.0
    _, reward, done, info = engine.step(Action.MOVE_RIGHT)
    assert done is True
    assert reward == 0.0
    assert info["reason"] == "ALREADY_DEAD"
    assert engine.ticks == 0


def test_boundary_and_no_resource_fail_without_state_or_energy_mutation():
    engine = SimulationEngine(CONFIG, seed=3)
    engine.state.grid.fill(engine.CHANNEL_EMPTY)
    engine.state.agent_pos = (0, 0)
    initial_energy = engine.state.agent_energy

    state, reward, done, info = engine.step(Action.MOVE_UP)
    assert state.agent_pos == (0, 0)
    assert state.agent_energy == initial_energy
    assert reward == 0.0
    assert done is False and info["reason"] == "OUT_OF_BOUNDS"

    state.grid[0, 0] = engine.CHANNEL_EMPTY
    _, reward, done, info = engine.step(Action.GATHER)
    assert state.agent_pos == (0, 0)
    assert state.agent_energy == initial_energy
    assert reward == 0.0
    assert done is False and info["reason"] == "NO_RESOURCE"


def test_generate_train_checkpoint_reload_smoke(tmp_path):
    db_path = tmp_path / "trajectory.db"
    checkpoint = tmp_path / "checkpoints/model.pth"
    generate_data(num_episodes=1, db_path=str(db_path), seed=23)
    assert db_path.is_file()

    saved = train(
        str(db_path),
        epochs=1,
        batch_size=64,
        seed=23,
        checkpoint_path=str(checkpoint),
    )
    assert saved == checkpoint
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0

    model = load_model(str(checkpoint))
    model.eval()
    grid = torch.zeros((1, 1, 20, 20))
    agent_map = torch.zeros((1, 1, 20, 20))
    action = torch.zeros((1, 6))
    action[:, Action.IDLE.value] = 1.0
    with torch.no_grad():
        predicted_grid, predicted_energy = model(grid, agent_map, action)
    assert predicted_grid.shape == (1, 1, 20, 20)
    assert predicted_energy.shape == (1, 1)


def test_seeded_generation_and_training_are_reproducible(tmp_path):
    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"
    first_checkpoint = tmp_path / "first.pth"
    second_checkpoint = tmp_path / "second.pth"

    generate_data(num_episodes=1, db_path=str(first_db), seed=31)
    generate_data(num_episodes=1, db_path=str(second_db), seed=31)

    query = """
        SELECT episode_id, step_count, grid, agent_pos_x, agent_pos_y,
               agent_energy, action, reward, done, next_grid,
               next_agent_pos_x, next_agent_pos_y, next_agent_energy
        FROM trajectories ORDER BY id
    """
    with sqlite3.connect(first_db) as first_connection:
        first_rows = first_connection.execute(query).fetchall()
    with sqlite3.connect(second_db) as second_connection:
        second_rows = second_connection.execute(query).fetchall()
    assert first_rows == second_rows

    train(str(first_db), epochs=1, batch_size=64, seed=31, checkpoint_path=first_checkpoint)
    train(str(first_db), epochs=1, batch_size=64, seed=31, checkpoint_path=second_checkpoint)
    first_state = torch.load(first_checkpoint, map_location="cpu", weights_only=True)
    second_state = torch.load(second_checkpoint, map_location="cpu", weights_only=True)
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)
