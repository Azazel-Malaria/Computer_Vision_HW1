from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(state_dict: Dict[str, np.ndarray], path: str | Path) -> None:
    np.savez(path, **state_dict)


def load_checkpoint(path: str | Path) -> Dict[str, np.ndarray]:
    obj = np.load(path)
    return {key: obj[key] for key in obj.files}


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
):
    indices = np.arange(x.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def plot_training_curves(history: Dict[str, List[float]], save_dir: str | Path) -> None:
    save_dir = ensure_dir(save_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "val_accuracy_curve.png", dpi=200)
    plt.close()


def plot_first_layer_weights(weight: np.ndarray, save_path: str | Path, max_filters: int = 64) -> None:
    """
    weight shape: [784, hidden_dim]
    """
    hidden_dim = weight.shape[1]
    num_show = min(max_filters, hidden_dim)
    cols = 8
    rows = int(np.ceil(num_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i < num_show:
            img = weight[:, i].reshape(28, 28)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"W{i}", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_misclassified_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    save_path: str | Path,
    max_items: int = 16,
) -> None:
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        return

    wrong_idx = wrong_idx[:max_items]
    cols = 4
    rows = int(np.ceil(len(wrong_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i < len(wrong_idx):
            idx = wrong_idx[i]
            ax.imshow(images[idx].reshape(28, 28), cmap="gray")
            ax.set_title(
                f"T: {class_names[y_true[idx]]}\nP: {class_names[y_pred[idx]]}",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
