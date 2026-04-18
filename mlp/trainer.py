from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List
import numpy as np
from .data import load_fashion_mnist
from .evaluate import evaluate_split
from .losses import cross_entropy_loss
from .model import MLPClassifier
from .optim import ExponentialLRScheduler, SGD
from .tensor import Tensor
from .utils import (
    ensure_dir,
    iterate_minibatches,
    plot_training_curves,
    save_checkpoint,
    save_json,
    set_seed,
)

def train_model(config: Dict, output_dir: str | Path) -> Dict:
    set_seed(config["seed"])
    output_dir = ensure_dir(output_dir)

    data = load_fashion_mnist(
        root=config["data_dir"],
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )

    model = MLPClassifier(
        input_dim=28 * 28,
        hidden_dims=config["hidden_dims"],
        num_classes=10,
        activation=config["activation"],
        seed=config["seed"],
    )
    optimizer = SGD(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = ExponentialLRScheduler(optimizer, gamma=config["lr_decay_gamma"])

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, config["epochs"] + 1):
        epoch_losses: List[float] = []
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (xb, yb) in enumerate(
            iterate_minibatches(
                data["x_train"],
                data["y_train"],
                batch_size=config["batch_size"],
                shuffle=True,
                seed=config["seed"] + epoch,
            )
        ):
            x_tensor = Tensor(xb, requires_grad=False)
            logits = model(x_tensor)
            loss, probs = cross_entropy_loss(logits, yb)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.data))
            epoch_correct += int((probs.argmax(axis=1) == yb).sum())
            epoch_total += len(yb)

        scheduler.step()

        train_loss = float(np.mean(epoch_losses))
        train_acc = epoch_correct / epoch_total
        val_metrics = evaluate_split(model, data["x_val"], data["y_val"], batch_size=config["batch_size"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(optimizer.lr)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"lr={optimizer.lr:.6f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            save_checkpoint(model.state_dict(), output_dir / "best_model.npz")

    save_json(config, output_dir / "config.json")
    save_json(history, output_dir / "history.json")
    plot_training_curves(history, output_dir)

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
    }
    save_json(summary, output_dir / "summary.json")
    return summary
