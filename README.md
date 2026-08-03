# World Model Kernel

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square&logo=pytorch)
![Architecture](https://img.shields.io/badge/Architecture-ResNet-green?style=flat-square)
![Database](https://img.shields.io/badge/Database-SQLite-lightblue?style=flat-square&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**Status:** Production Ready (v1.0)  
**Author:** Cesar Augusto  

A deterministic, causal physics kernel for training foundational World Models. This system isolates causality from emergent chaos through strict adherence to thermodynamic laws, enabling high-fidelity synthetic data generation for neural network training.

---

## Executive Summary

The World Model Kernel is a minimal simulation environment designed to generate spatially and energetically consistent trajectories for training predictive models. Unlike traditional game engines that prioritize visual fidelity or gameplay mechanics, this kernel enforces fundamental physical constraints: Conservation of Energy, Spatial Locality, and Action Costs, to produce training data that reflects real-world causal relationships.

**Current Achievement:** The system successfully predicts grid state transitions (S_t → S_{t+1}) with >95% spatial accuracy and 0.0091 MSE loss, demonstrating that the neural network has internalized the underlying physics rules.

---

## Architecture Overview

The system implements a three-layer architecture separating state representation, physics simulation, and intelligent prediction:

```
┌─────────────────────────────────────────────────────────────┐
│                     State Layer (Kernel)                     │
│  GridState: Immutable snapshot of universe at time t        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Physics Layer (Simulation)                  │
│  SimulationEngine: Deterministic state transitions           │
│  - Energy Conservation: ΔE_total = 0                        │
│  - Spatial Locality: Actions affect only neighbors          │
│  - Action Costs: Energy required for all operations         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                Intelligence Layer (Agent)                    │
│  WorldModelResNet: Learns S_t → S_{t+1} mapping            │
│  - Training: Supervised learning on physics trajectories    │
│  - Inference: Predicts future states autoregressively       │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **Kernel (`src/kernel/`)**: Defines immutable state structures (GridState) representing the universe at discrete time steps
- **Simulation (`src/simulation/`)**: Implements deterministic physics engine (SimulationEngine) that enforces thermodynamic laws
- **Agent (`src/agent/`)**: Contains neural network architecture (WorldModelResNet) and training infrastructure
- **Visualization (`src/viz/`)**: Provides validation tools for comparing ground truth physics against model predictions

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/cesaremcasa/civic_kernel.git
cd civic_kernel

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**1. Generate Training Data**

Create deterministic physics trajectories:

```bash
python src/main.py generate --episodes 1000
```

This produces ~112k state transitions stored in `data/kernel_db.db`.

**2. Train World Model**

Train the neural network to predict state evolution:

```bash
python src/main.py train --epochs 20
```

Model checkpoints are saved to `data/checkpoints/`.

**3. Validate Predictions**

Visualize model understanding of physics:

```bash
python src/viz/inspector.py
```

Output displays side-by-side comparison of ground truth vs predicted states.

---

## Performance Metrics

| Metric | Result | Target | Status |
|:-------|:-------|:-------|:-------|
| Training Loss (MSE) | 0.0091 | < 0.01 | ✓ Met |
| Spatial Accuracy | >95% | >90% | ✓ Exceeded |
| Energy Prediction Error | ±5.0 | ±10.0 | ✓ Met |
| Physics Tick Speed | <2ms | <5ms | ✓ Met |
| Inference Latency | <1ms | <1ms | ✓ Met |

---

## Physics Model

The simulation is governed by three immutable laws. The neural network learns to predict dynamics constrained by these rules:

### 1. Conservation of Energy

Total energy in the system remains constant. Energy can only be transferred (Resource → Agent via `gather` action) or dissipated (Movement cost).

```
ΔE_total = 0
```

### 2. Spatial Locality

All interactions are local. An agent at position (x, y) can only affect its immediate neighbors. No action-at-a-distance or teleportation.

### 3. Action Costs

Every action requires energy expenditure. Actions are blocked if the agent's energy falls below the required threshold:

```
∀ Action A, ∃ Cost(A)
Action succeeds ⟺ Energy ≥ Cost(A)
```

---

## Project Structure

```
civic_kernel/
├── config/
│   └── constants.yaml          # Physics parameters (costs, grid size)
├── data/
│   ├── kernel_db.db           # SQLite trajectory storage
│   └── checkpoints/           # Trained model weights (.pth)
├── src/
│   ├── kernel/                # State management layer
│   │   └── state.py          # GridState definition
│   ├── simulation/            # Physics engine
│   │   ├── engine.py         # SimulationEngine implementation
│   │   └── generator.py      # Trajectory data generation
│   ├── agent/                 # Neural network layer
│   │   └── trainer.py        # Training loop and optimization
│   ├── viz/                   # Validation tools
│   │   └── inspector.py      # Visual state comparison
│   └── main.py               # CLI entry point
├── tests/                     # Unit and integration tests
├── requirements.txt          # Locked dependencies
└── README.md                # This file
```

---

## Engineering Notes

### Architecture Evolution

**Version 1 (CNN-based):** Initial implementation used deep convolutional network with max pooling. Result: Blurry predictions with hallucinated intermediate states between discrete values.

**Version 2 (ResNet-based):** Removed pooling layers and added skip connections to preserve spatial structure. Result: Sharp predictions maintaining discrete state values. Loss improved from 0.25 to 0.0091.

### Technology Decisions

**SQLite over CSV/Parquet:** Selected for ACID transactional guarantees and zero-dependency portability. Critical for maintaining data integrity during parallel trajectory generation.

**NumPy Raw Bytes:** Serialization uses `tobytes()` instead of pickle, achieving 10x speedup in database write operations while maintaining full precision.

**Dynamic Layer Sizing:** PyTorch model uses lazy initialization to handle variable batch dimensions without hardcoded tensor shapes, improving flexibility during development and testing.

---

## Implementation Compliance

This project adheres to RFC-001 v1.1 specifications. All six development phases have been completed:

**Phase 1 - Vacuum (Static Grid):** Deterministic grid generation with walls and resources  
**Phase 2 - Particle (Agent Kinematics):** Agent movement with energy costs  
**Phase 3 - Exchange (Thermodynamics):** Energy transfer via resource gathering  
**Phase 4 - Oracle (Data Generation):** High-throughput trajectory creation  
**Phase 5 - Learner (World Model Training):** Neural network convergence  
**Phase 6 - Dreamer (Autoregressive Rollout):** Multi-step prediction validation

---

## License

MIT License

Copyright (c) 2025 Cesar Augusto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contact

**Cesar Augusto** · AI Systems Engineer, Mycellium Lab  
Repository: [github.com/cesaremcasa/civic_kernel](https://github.com/cesaremcasa/civic_kernel)
