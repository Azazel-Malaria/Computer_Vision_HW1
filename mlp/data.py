from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


FASHION_MNIST_URLS = {
    "train_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    "train_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    "test_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def download_fashion_mnist(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, url in FASHION_MNIST_URLS.items():
        filename = root / Path(url).name
        if not filename.exists():
            print(f"[Data] Downloading {filename.name} ...")
            urllib.request.urlretrieve(url, filename)
    return root


def _read_images(path: str | Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def _read_labels(path: str | Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data.astype(np.int64)


def _flatten_and_normalize(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1).astype(np.float64) / 255.0


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def load_fashion_mnist(
    root: str | Path = "./data",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    root = download_fashion_mnist(root)

    x_train = _read_images(root / "train-images-idx3-ubyte.gz")
    y_train = _read_labels(root / "train-labels-idx1-ubyte.gz")
    x_test = _read_images(root / "t10k-images-idx3-ubyte.gz")
    y_test = _read_labels(root / "t10k-labels-idx1-ubyte.gz")

    x_train = _flatten_and_normalize(x_train)
    x_test = _flatten_and_normalize(x_test)

    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, val_ratio=val_ratio, seed=seed)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "class_names": np.array(CLASS_NAMES),
    }
