"""
World Model Kernel - Entry Point
Orchestrates Data Generation, Training, and Inference.
"""
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from simulation.generator import generate_data
from agent.trainer import train as run_training

def main():
    parser = argparse.ArgumentParser(description="World Model Kernel CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Command: Data Generation
    parser_gen = subparsers.add_parser('generate', help='Generate trajectory data')
    parser_gen.add_argument('--episodes', type=int, default=1000, help='Number of episodes to generate')
    parser_gen.add_argument('--db', type=str, default='data/kernel_db.db', help='Path to database')

    # Command: Training
    parser_train = subparsers.add_parser('train', help='Train World Model')
    parser_train.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser_train.add_argument('--db', type=str, default='data/kernel_db.db', help='Path to database')

    args = parser.parse_args()

    if args.command == 'generate':
        print(f"🌱 Generating {args.episodes} episodes...")
        generate_data(num_episodes=args.episodes, db_path=args.db)
        print("✅ Data Generation Complete.")
    
    elif args.command == 'train':
        print(f"🧠 Training Model for {args.epochs} epochs...")
        run_training(db_path=args.db, epochs=args.epochs)
        print("✅ Training Complete.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
