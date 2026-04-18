from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from .tensor import Tensor


class Module:
    def parameters(self) -> List[Tensor]:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def state_dict(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        limit = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            rng.standard_normal((in_features, out_features)) * limit,
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "weight": self.weight.data.copy(),
            "bias": self.bias.data.copy(),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.weight.data = state_dict["weight"].copy()
        self.bias.data = state_dict["bias"].copy()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def parameters(self) -> List[Tensor]:
        return []

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        return None


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()
    def parameters(self) -> List[Tensor]:
        return []
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {}
    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        return None


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
    def parameters(self) -> List[Tensor]:
        return []
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {}
    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        return None
