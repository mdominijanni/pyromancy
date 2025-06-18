import einops as ein
import math
import torch
from typing import Iterator, overload


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
        assert len(shape) > 0
        assert all(isinstance(s, int | None) for s in shape)
        assert all(s > 0 for s in shape if s is not None)

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
        return f"{type(self).__name__}({", ".join(str(d) for d in self._rawshape)})"

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
    def rawshape(self) -> tuple[int | None, ...]:
        return self._rawshape

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(1 if s is None else s for s in self._rawshape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self._rawshape)

    @property
    def nconcrete(self) -> int:
        return len(self._concrete_dims)

    @property
    def nvirtual(self) -> int:
        return len(self._virtual_dims)

    def compat(self, *shape: int) -> bool:
        assert all(isinstance(d, int) for d in shape)
        assert all(d > 0 for d in shape)

        if len(shape) != len(self._rawshape):
            return False

        for dx, di in zip(shape, self._rawshape):
            if di is not None and dx != di:
                return False

        return True

    def filled(self, *fill: int) -> tuple[int, ...]:
        assert len(fill) == self.nvirtual
        assert all(isinstance(d, int) for d in fill)
        assert all(d > 0 for d in fill)

        shape = [*self._rawshape]
        for n, d in enumerate(self._virtual_dims):
            shape[d] = fill[n]

        return tuple(shape)

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
