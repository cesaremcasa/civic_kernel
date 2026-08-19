# Civic Kernel v0.2.0

**Status:** reproducible simulation baseline

Civic Kernel is a small deterministic grid simulator and a CPU training smoke
path for world-model experiments. Version 0.2.0 makes the state-transition
engine importable, repairs the movement and energy edge cases, and documents a
seeded generation → training → checkpoint → reload path. It does not claim
production accuracy, convergence, throughput, or model quality.

## Public API

`SimulationEngine` is the sole state/physics authority. The stable package
exports are:

```python
from civic_kernel import Action, SimulationEngine, Transition

engine = SimulationEngine("config/constants.yaml", seed=7)
state, reward, done, info = engine.step(Action.MOVE_RIGHT)
```

Coordinates are `(row, column)` pairs. `MOVE_UP`/`MOVE_DOWN` change the row;
`MOVE_LEFT`/`MOVE_RIGHT` change the column. A successful movement spends
`COST_MOVE`. Gathering spends `COST_GATHER`, transfers the available resource
to the agent up to `MAX_ENERGY`, and consumes the resource cell. Failed wall,
boundary, and no-resource actions do not mutate energy. A no-energy action or
an action that reaches zero energy returns `done=True` with a terminal reason.

The obsolete `kernel.physics` dependency is not part of the main path; tests,
generation, and the public API all use `SimulationEngine` directly.

## Clean-environment quickstart

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cesaremcasa/civic_kernel.git
cd civic_kernel
uv sync --frozen --extra dev

# Generate a small deterministic SQLite trajectory database.
uv run civic-kernel generate \
  --episodes 1 --seed 7 --db /tmp/civic-kernel.db

# Train one CPU epoch and save a local checkpoint.
uv run civic-kernel train \
  --epochs 1 --seed 7 \
  --db /tmp/civic-kernel.db \
  --checkpoint /tmp/civic-kernel.pth
```

Generated databases and checkpoints are intentionally not release artifacts.
They can be reproduced with the command and provenance recorded in
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md). To run the complete local gate:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run pip-audit --skip-editable
```

## Layout

```text
config/constants.yaml       simulation constants
src/civic_kernel/            installed package, CLI, and packaged defaults
src/simulation/              canonical engine and trajectory generation
src/kernel/state.py          seeded grid state
src/agent/                   model, dataset, training, reload helpers
tests/                       contract, failure, and smoke coverage
```

## Scope and limitations

The simulator is a compact synthetic environment, not a physical-world
model. Resource placement uses a seeded NumPy generator; action selection in
the generator uses a seeded Python generator. Training is a deterministic CPU
smoke path for the included model architecture. Results depend on the pinned
Python/dependency environment and the supplied seed. No benchmark or accuracy
number is asserted by this release.

See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for the exact reproducibility
contract and [`LICENSE`](LICENSE) for licensing terms.
