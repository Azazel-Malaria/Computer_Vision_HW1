from __future__ import annotations

from typing import Dict

import numpy as np

from .losses import cross_entropy_loss, softmax
from .metrics import accuracy_score
from .tensor import Tensor


def predict_logits(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    logits = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size]
        out = model(Tensor(xb, requires_grad=False))
        logits.append(out.data)
    return np.concatenate(logits, axis=0)


def evaluate_split(model, x: np.ndarray, y: np.ndarray, batch_size: int = 256) -> Dict[str, float]:
    logits = predict_logits(model, x, batch_size=batch_size)
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y, y_pred)
    return {
        "loss": float(loss),
        "acc": float(acc),
        "y_pred": y_pred,
        "logits": logits,
    }
