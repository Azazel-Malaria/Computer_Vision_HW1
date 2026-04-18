from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from mlp.data import load_fashion_mnist
from mlp.evaluate import evaluate_split
from mlp.metrics import confusion_matrix
from mlp.model import MLPClassifier
from mlp.utils import ensure_dir, load_checkpoint, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run test")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default='./outputs/default_run/best_model.npz')
    parser.add_argument("--config", type=str, default='./outputs/default_run/config.json')
    parser.add_argument("--output_dir", type=str, default="./outputs/test_eval")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    config = load_json(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    data = load_fashion_mnist(root=args.data_dir, val_ratio=config["val_ratio"], seed=config["seed"])
    model = MLPClassifier(
        input_dim=28 * 28,
        hidden_dims=config["hidden_dims"],
        num_classes=10,
        activation=config["activation"],
        seed=config["seed"],
    )
    model.load_state_dict(checkpoint)

    metrics = evaluate_split(model, data["x_test"], data["y_test"], batch_size=args.batch_size)
    cm = confusion_matrix(data["y_test"], metrics["y_pred"], num_classes=10)

    print(f"Test accuracy: {metrics['acc']:.4f}")
    print("Confusion matrix:")
    print(cm)

    np.savetxt(output_dir / "confusion_matrix.txt", cm, fmt="%d")
    with open(output_dir / "test_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{metrics['acc']:.6f}\n")
