from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from mlp.trainer import train_model
from mlp.utils import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search for Fashion-MNIST MLP.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--search_dir", type=str, default="./outputs/grid_search")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    search_dir = ensure_dir(args.search_dir)

    learning_rates = [0.1, 0.05, 0.01]
    weight_decays = [1e-4, 5e-4, 1e-3]
    hidden_dims_list = [(256, 128), (512, 256)]
    activations = ["relu", "tanh"]

    results = []
    trial_id = 0

    for lr, wd, hidden_dims, activation in itertools.product(
        learning_rates,
        weight_decays,
        hidden_dims_list,
        activations,
    ):
        trial_id += 1
        trial_dir = search_dir / f"trial_{trial_id:02d}"
        config = {
            "data_dir": args.data_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": lr,
            "lr_decay_gamma": 0.95,
            "weight_decay": wd,
            "hidden_dims": list(hidden_dims),
            "activation": activation,
            "val_ratio": args.val_ratio,
            "seed": args.seed + trial_id,
        }

        print(f"\n=== Trial {trial_id} / config={config} ===")
        summary = train_model(config, trial_dir)
        results.append(
            {
                "trial_id": trial_id,
                "config": config,
                "best_val_acc": summary["best_val_acc"],
                "best_epoch": summary["best_epoch"],
                "model_path": str(trial_dir / "best_model.npz"),
            }
        )

    results = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)
    save_json({"results": results, "best": results[0]}, search_dir / "search_results.json")

    print("\n=== Search finished ===")
    print("Top-3 configurations:")
    for item in results[:3]:
        print(item)
