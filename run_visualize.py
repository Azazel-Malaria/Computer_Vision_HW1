from __future__ import annotations

import argparse
from pathlib import Path

from mlp.data import load_fashion_mnist
from mlp.evaluate import evaluate_split
from mlp.model import MLPClassifier
from mlp.utils import (
    ensure_dir,
    load_checkpoint,
    load_json,
    plot_first_layer_weights,
    plot_misclassified_examples,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize first-layer weights and misclassified examples.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/visualizations")
    parser.add_argument("--batch_size", type=int, default=256)
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

    plot_first_layer_weights(model.fc1.weight.data, output_dir / "first_layer_weights.png")

    metrics = evaluate_split(model, data["x_test"], data["y_test"], batch_size=args.batch_size)
    plot_misclassified_examples(
        data["x_test"],
        data["y_test"],
        metrics["y_pred"],
        data["class_names"],
        output_dir / "misclassified_examples.png",
        max_items=16,
    )

    print(f"Saved weight visualization to: {output_dir / 'first_layer_weights.png'}")
    print(f"Saved misclassified examples to: {output_dir / 'misclassified_examples.png'}")
