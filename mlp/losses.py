from __future__ import annotations

from typing import Tuple
import numpy as np
from .tensor import Tensor


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tuple[Tensor, np.ndarray]:
    probs = softmax(logits.data)
    batch_size = targets.shape[0]
    loss_value = -np.log(probs[np.arange(batch_size), targets] + 1e-12).mean()

    out = Tensor(
        np.array(loss_value),
        requires_grad=logits.requires_grad,
        _children=(logits,),
        _op="cross_entropy",
    )

    def _backward() -> None:
        if not logits.requires_grad:
            return
        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1.0
        grad /= batch_size
        logits.grad += grad * out.grad

    out._backward = _backward
    return out, probs
