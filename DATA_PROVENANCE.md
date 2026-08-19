# Data provenance

Civic Kernel v0.2.0 does not ship a generated trajectory database or model
checkpoint. Those files are experiment outputs and are ignored by Git.

The reproducible local smoke path is:

```bash
uv sync --frozen --extra dev
uv run civic-kernel generate --episodes 1 --seed 7 --db /tmp/civic-kernel.db
uv run civic-kernel train --epochs 1 --seed 7 \
  --db /tmp/civic-kernel.db --checkpoint /tmp/civic-kernel.pth
```

Inputs and controls:

- configuration: `config/constants.yaml` (also bundled in the wheel as the
  default for an installed CLI);
- world-state seed: `seed + episode_id`;
- action-stream seed: `seed` through Python's `random.Random`;
- state placement: NumPy's `default_rng`;
- training seed: Python and PyTorch seeds, deterministic CPU algorithms;
- runtime: Python 3.11 with the versions in `uv.lock`.

The generator snapshots the pre-action state before writing each SQLite row,
so `grid`/energy and `next_grid`/energy represent the actual transition.
Running the same command against a fresh database path yields the same ordered
trajectory values. SQLite metadata such as file timestamps is not treated as a
dataset signal.

The smoke checkpoint is a state dictionary for the included model definition;
it is not a trained or validated production model. Re-run the commands after
changing the configuration, dependency lock, model code, or seed and treat the
result as a new experiment.
