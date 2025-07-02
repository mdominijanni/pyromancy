import functools
import math
import types
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from typing import Any, Type, overload

import einops as ein
import torch
import torch.nn as nn


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


class LambdaModule(nn.Module):
    r"""Wrapper module for a Callable.

    Args:
        fn (Callable): callable to wrap.
        *args (Any): prepended positional arguments for ``fn``.
        *kwargs (Any): appended keyword arguments for ``fn``.

    Raises:
        TypeError: ``fn`` must be a ``~collections.abc.Callable``.

    Tip:
        The behavior for ``*args`` and ``**kwargs`` mirrors that of
        :py:func:`functools.partial`.
    """

    _fn: Callable
    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        if not isinstance(fn, Callable):
            raise TypeError("`fn` must be a `Callable`")

        nn.Module.__init__(self)
        self._fn = fn
        self._args = args
        self._kwargs = {k: v for k, v in kwargs.items()}

        @functools.wraps(fn)
        def forward(self, *fargs, **fkwargs):
            return self._fn(*fargs, *self._args, **(self._kwargs | fkwargs))

        self.forward = types.MethodType(forward, self)

    def extra_repr(self) -> str:
        if isinstance(self._fn, types.MethodType):
            name = self._fn.__qualname__
        else:
            name = self._fn.__name__

        return (
            f"{name}{', ' if self._args else ''}"
            f"{', '.join(f'{a}' for a in self._args)}{', ' if self._kwargs else ''}"
            f"{', '.join(f'{k}={v}' for k, v in self._kwargs.items())}"
        )


class TypedModuleDict[T: nn.Module](nn.Module, MutableMapping):
    r"""Holds submodules of a specific type in a dictionary.

    This class is nearly identical to :py:class:`~torch.nn.ModuleDict`, but it allows
    for the type of permitted submodules to be narrowed.

    Args:
        modules (Mapping[str, T] | None, optional): initial modules to add along with
            their names. Defaults to ``None``.
        narrowing (Type[T], optional): subclass to narrow inserted modules to.
            Defaults to :py:class:`~torch.nn.Module`.

    Raises:
        TypeError: ``narrowing`` must specify a type of :py:class:`~torch.nn.Module`.
    """

    _modules: dict[str, T]  # type: ignore[assignment]
    _modtype: Type[T]

    def __init__(
        self, modules: Mapping[str, T] | None = None, narrowing: Type[T] = nn.Module
    ) -> None:
        if not issubclass(narrowing, nn.Module):
            raise TypeError(
                f"the type specified by `narrowing`, `{narrowing.__name__}` "
                f"must be a subclass of `{nn.Module.__name__}`"
            )

        nn.Module.__init__(self)
        self._modtype = narrowing
        if modules is not None:
            self.update(modules)

    def __contains__(self, key: object) -> bool:
        return key in self._modules

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __getitem__(self, key: str) -> T:
        return self._modules[key]

    def __setitem__(self, key: str, module: T) -> None:
        if not isinstance(key, str):
            raise TypeError(f"`key` of type `{type(key).__name__}` must be a `str`")
        if not isinstance(module, self._modtype):
            raise TypeError(
                f"`module` of type `{type(module).__name__}` "
                f"must be an instance of `{self._modtype.__name__}`"
            )
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)
