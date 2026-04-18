from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .tensor import Tensor


class SGD:
    def __init__(self, params: Iterable[Tensor], lr: float = 0.1, weight_decay: float = 0.0) -> None:
        self.params: List[Tensor] = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            update = param.grad
            if self.weight_decay > 0:
                update = update + self.weight_decay * param.data
            param.data -= self.lr * update

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()


class ExponentialLRScheduler:
    def __init__(self, optimizer: SGD, gamma: float = 0.95) -> None:
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self) -> None:
        self.optimizer.lr *= self.gamma
