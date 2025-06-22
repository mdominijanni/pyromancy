import math
from typing import Any, Iterator, overload

import einops as ein
import torch


class Shape:
    r"""Tensor shape with support for placeholder dimensions.

    Args:
        *shape (int | None): dimensions of the tensor, either positive integers for
            fixed dimensions or none for unspecified dimensions.

    Important:
        Scalar tensors (i.e. tensors with no dimensions) are unsupported, as are tensors
        with any dimension of size 0.
    """

    _rawshape: tuple[int | None, ...]
    _concrete_dims: tuple[int, ...]
    _virtual_dims: tuple[int, ...]
    _parseshp_str: str
    _coalesce_str: str
    _disperse_str: str

    def __init__(self, *shape: int | None) -> None:
        if not len(shape) > 0:
            raise ValueError("`shape` must contain at least one element")
        if not all(isinstance(s, int | None) for s in shape):
            raise TypeError("all elements of `shape` must be of type `int` or `None`")
        if not all(s > 0 for s in shape if s is not None):
            raise ValueError("all integer elements of `shape` must be positive")

        self._rawshape = tuple(int(s) if s is not None else None for s in shape)
        self._concrete_dims = tuple(
            d for d, s in enumerate(self._rawshape) if s is not None
        )
        self._virtual_dims = tuple(d for d, s in enumerate(self._rawshape) if s is None)

        dims = tuple(f"d{d}" for d in range(len(self._rawshape)))
        cdims = tuple(f"d{d}" for d in self._concrete_dims)
        vdims = tuple(f"d{d}" for d in self._virtual_dims)

        self._parseshp_str = " ".join(dims)
        self._coalesce_str = (
            f"{' '.join(dims)} -> ({' '.join(vdims)}) ({' '.join(cdims)})"
        )
        self._disperse_str = (
            f"({' '.join(vdims)}) ({' '.join(cdims)}) -> {' '.join(dims)}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(str(d) for d in self._rawshape)})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            return self._rawshape == other._rawshape
        elif isinstance(other, tuple):
            return self._rawshape == other
        else:
            return False

    @overload
    def __getitem__(self, index: int) -> int | None: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int | None, ...]: ...

    def __getitem__(self, index: int | slice) -> int | None | tuple[int | None, ...]:
        return self._rawshape[index]

    def __len__(self) -> int:
        return len(self._rawshape)

    def __iter__(self) -> Iterator[int | None]:
        return iter(self._rawshape)

    @property
    def rshape(self) -> tuple[int | None, ...]:
        r"""Tensor shape, including placeholder dimensions.

        Returns:
            tuple[int | None, ...]: raw tensor shape.
        """
        return self._rawshape

    @property
    def bshape(self) -> tuple[int, ...]:
        r"""Tensor shape, with placeholder dimensions set to unit length.

        Returns:
            tuple[int | None, ...]: broadcastable tensor shape.
        """
        return tuple(1 if s is None else s for s in self._rawshape)

    @property
    def size(self) -> int:
        r"""Number of elements specified by the shape.

        Returns:
            int: minimal number of tensor elements.
        """
        return math.prod(self.bshape)

    @property
    def ndim(self) -> int:
        r"""Number of dimensions specified by the shape.

        Returns:
            int: dimensionality of a compatible tensor.
        """
        return len(self._rawshape)

    @property
    def nconcrete(self) -> int:
        r"""Number of fixed dimensions.

        Returns:
            int: number of concrete dimensions.
        """
        return len(self._concrete_dims)

    @property
    def nvirtual(self) -> int:
        r"""Number of placeholder dimensions.

        Returns:
            int: number of virtual dimensions.
        """
        return len(self._virtual_dims)

    def compat(self, *shape: int) -> bool:
        r"""Tests if a shape is compatible with the specified constraints.

        Args:
            *shape (int | None): dimensions of the tensor.

        Returns:
            bool: if the shape is compatible.
        """
        if not all(isinstance(d, int) for d in shape):
            raise TypeError("all elements of `shape` must be of type `int`")
        if not all(d > 0 for d in shape):
            raise ValueError("all elements of `shape` must be positive")

        if len(shape) != len(self._rawshape):
            return False

        for dx, di in zip(shape, self._rawshape):
            if di is not None and dx != di:
                return False

        return True

    def filled(self, *fill: int) -> tuple[int, ...]:
        r"""Fills placeholder dimensions with specified values.

        Returns:
            tuple[int, ...]: shape with the placeholder dimensions filled.
        """
        if not len(fill) == self.nvirtual:
            raise ValueError(
                "`fill` must contain exactly the required number of placeholder elements"
            )
        if not all(isinstance(d, int) for d in fill):
            raise TypeError("all elements of `fill` must be of type `int`")
        if not all(d > 0 for d in fill):
            raise ValueError("all elements of `fill` must be positive")

        shape = [*self._rawshape]
        for n, d in enumerate(self._virtual_dims):
            shape[d] = fill[n]

        return tuple(shape)  # type: ignore

    def coalesce(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
        r"""Coalesces a tensor into a matrix, with placeholder dimensions first and fixed dimensions second.

        For a tensor with :math:`V_1, \ldots, V_m` placeholder dimensions and
        :math:`C_1, \ldots, C_n` fixed dimensions, the output matrix will have a shape of
        :math:`(V_1 \times \cdots \times V_m) \times (C_1 \times \cdots \times C_n)`, and
        dimensions of unit length will used if the tensor has no placeholder/fixed dimensions.

        Args:
            tensor (~torch.Tensor): tensor to coalesce.

        Returns:
            tuple[~torch.Tensor, dict[str, int]]: tuple of the coalesced tensor and the
                required shape information to revert it.
        """
        pragma = ein.parse_shape(tensor, self._parseshp_str)
        return ein.rearrange(tensor, self._coalesce_str), pragma

    def disperse(self, tensor: torch.Tensor, pragma: dict[str, int]) -> torch.Tensor:
        r"""Disperses dimensions of a coalesced tensor to their original positions.

        Args:
            tensor (~torch.Tensor): tensor to disperse.
            pragma (dict[str, int]): shape information to revert the tensor.

        Returns:
            ~torch.Tensor: dispersed tensor.
        """
        return ein.rearrange(tensor, self._disperse_str, **pragma)
