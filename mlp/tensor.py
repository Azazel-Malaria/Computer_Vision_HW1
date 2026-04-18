from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Set, Tuple

import numpy as np


def _ensure_array(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float64, copy=False)
    return np.array(data, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if grad.shape == shape:
        return grad

    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: Sequence["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data = _ensure_array(data)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data) if requires_grad else None
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Tensor"] = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if grad is None:
            grad = np.ones_like(self.data)

        topo = []
        visited = set()

        def build(v: "Tensor") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = _ensure_array(grad)

        for node in reversed(topo):
            node._backward()

    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __sub__(self, other) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other) -> "Tensor":
        return other + (-self)

    def __neg__(self) -> "Tensor":
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="neg",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __truediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def pow(self, power: float) -> "Tensor":
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"pow({power})",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (power * (self.data ** (power - 1.0))) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        out = Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.T

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if not self.requires_grad:
                return

            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                adjusted = []
                for ax in axes:
                    adjusted.append(ax if ax >= 0 else ax + self.data.ndim)
                adjusted = tuple(sorted(adjusted))
                if not keepdims:
                    for ax in adjusted:
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = 1
            for ax in axes:
                denom *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(0.0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float64) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(
            sig,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += sig * (1.0 - sig) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def numpy(self) -> np.ndarray:
        return self.data.copy()
