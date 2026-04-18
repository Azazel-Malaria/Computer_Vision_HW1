from __future__ import annotations

import argparse
from pathlib import Path

from mlp.trainer import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3-layer MLP on Fashion-MNIST.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs/default_run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.08)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=int, nargs=2, default=[256, 128])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)
    summary = train_model(config, Path(args.output_dir))
    print("\nTraining finished.")
    print(f"Best validation accuracy: {summary['best_val_acc']:.4f}")
    print(f"Best epoch: {summary['best_epoch']}")
