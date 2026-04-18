from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .layers import Linear, Module, ReLU, Sigmoid, Tanh
from .tensor import Tensor


def build_activation(name: str):
    key = name.lower()
    if key == "relu":
        return ReLU()
    if key == "sigmoid":
        return Sigmoid()
    if key == "tanh":
        return Tanh()


class MLPClassifier(Module):
    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: Sequence[int] = (256, 128),
        num_classes: int = 10,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.activation_name = activation

        self.fc1 = Linear(input_dim, hidden_dims[0], seed=seed)
        self.act1 = build_activation(activation)
        self.fc2 = Linear(hidden_dims[0], hidden_dims[1], seed=seed + 1)
        self.act2 = build_activation(activation)
        self.fc3 = Linear(hidden_dims[1], num_classes, seed=seed + 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "fc1.weight": self.fc1.weight.data.copy(),
            "fc1.bias": self.fc1.bias.data.copy(),
            "fc2.weight": self.fc2.weight.data.copy(),
            "fc2.bias": self.fc2.bias.data.copy(),
            "fc3.weight": self.fc3.weight.data.copy(),
            "fc3.bias": self.fc3.bias.data.copy(),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.fc1.weight.data = state_dict["fc1.weight"].copy()
        self.fc1.bias.data = state_dict["fc1.bias"].copy()
        self.fc2.weight.data = state_dict["fc2.weight"].copy()
        self.fc2.bias.data = state_dict["fc2.bias"].copy()
        self.fc3.weight.data = state_dict["fc3.weight"].copy()
        self.fc3.bias.data = state_dict["fc3.bias"].copy()
