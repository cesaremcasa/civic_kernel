"""Installed CLI entry point for Civic Kernel."""

from __future__ import annotations

import argparse

from agent.trainer import train as run_training
from simulation.generator import generate_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Civic Kernel deterministic simulation")
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate", help="generate trajectory data")
    generate.add_argument("--episodes", type=int, default=1000)
    generate.add_argument("--db", default="data/kernel_db.db")
    generate.add_argument("--seed", type=int, default=0)

    train = subparsers.add_parser("train", help="train the world model")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--db", default="data/kernel_db.db")
    train.add_argument("--checkpoint", default="data/checkpoints/kernel_model.pth")
    train.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "generate":
        print(f"Generating {args.episodes} episodes...")
        generate_data(args.episodes, args.db, seed=args.seed)
        print("Data generation complete.")
        return 0
    if args.command == "train":
        print(f"Training model for {args.epochs} epochs...")
        run_training(args.db, epochs=args.epochs, seed=args.seed, checkpoint_path=args.checkpoint)
        print(f"Training complete: {args.checkpoint}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
