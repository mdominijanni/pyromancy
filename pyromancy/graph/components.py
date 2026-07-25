from .._internal import _Final

from collections.abc import Mapping
import itertools
from types import MappingProxyType
from typing import Any, Self


class NodeSrc(metaclass=_Final):

    _name: str

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("`name` must be a `str`")
        if "." in name:
            raise ValueError("`name` cannot contain character '.'")
        if len(name) == 0:
            raise ValueError("`name` cannot be an empty string")

        self._name = name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name})"

    @property
    def name(self) -> str:
        return self._name


class OpSrc(metaclass=_Final):

    _name: str

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("`name` must be a `str`")
        if "." in name:
            raise ValueError("`name` cannot contain character '.'")
        if len(name) == 0:
            raise ValueError("`name` cannot be an empty string")

        self._name = name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name})"

    @property
    def name(self) -> str:
        return self._name


class InputSrc(metaclass=_Final):

    _name: str

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("`name` must be a `str`")
        if len(name) == 0:
            raise ValueError("`name` cannot be an empty string")

        self._name = name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name})"

    @property
    def name(self) -> str:
        return self._name


class NodeCfg:

    _activity_kwargs: dict[str, Any]
    _initialize_kwargs: dict[str, Any]
    _energy_kwargs: dict[str, Any]
    _reset_kwargs: dict[str, Any]

    def __init__(
        self,
        *,
        activity_kwargs: Mapping[str, Any] | None = None,
        initialize_kwargs: Mapping[str, Any] | None = None,
        energy_kwargs: Mapping[str, Any] | None = None,
        reset_kwargs: Mapping[str, Any] | None = None,
    ) -> None:

        if activity_kwargs is not None:
            self._activity_kwargs = {**activity_kwargs}
        else:
            self._activity_kwargs = {}

        if initialize_kwargs is not None:
            self._initialize_kwargs = {**initialize_kwargs}
        else:
            self._initialize_kwargs = {}

        if energy_kwargs is not None:
            self._energy_kwargs = {**energy_kwargs}
        else:
            self._energy_kwargs = {}

        if reset_kwargs is not None:
            self._reset_kwargs = {**reset_kwargs}
        else:
            self._reset_kwargs = {}

    @property
    def activity_kwargs(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(self._activity_kwargs)

    @activity_kwargs.setter
    def activity_kwargs(self, value: Mapping[str, Any] | None) -> None:
        if value is not None:
            self._activity_kwargs = {**value}
        else:
            self._activity_kwargs = {}

    @property
    def initialize_kwargs(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(self._initialize_kwargs)

    @initialize_kwargs.setter
    def initialize_kwargs(self, value: Mapping[str, Any] | None) -> None:
        if value is not None:
            self._initialize_kwargs = {**value}
        else:
            self._initialize_kwargs = {}

    @property
    def energy_kwargs(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(self._energy_kwargs)

    @energy_kwargs.setter
    def energy_kwargs(self, value: Mapping[str, Any] | None) -> None:
        if value is not None:
            self._energy_kwargs = {**value}
        else:
            self._energy_kwargs = {}

    def add_activity_kwargs(self, **kwargs: Any) -> Self:
        self._activity_kwargs |= kwargs
        return self

    def add_initialize_kwargs(self, **kwargs: Any) -> Self:
        self._initialize_kwargs |= kwargs
        return self

    def add_energy_kwargs(self, **kwargs: Any) -> Self:
        self._energy_kwargs |= kwargs
        return self

    def add_reset_kwargs(self, **kwargs: Any) -> Self:
        self._reset_kwargs |= kwargs
        return self

    def del_activity_kwargs(self, **args: str) -> Self:
        for key in args:
            try:
                del self._activity_kwargs[key]
            except KeyError:
                pass
            except Exception as e:
                raise e
        return self

    def del_initialize_kwargs(self, **args: str) -> Self:
        for key in args:
            try:
                del self._initialize_kwargs[key]
            except KeyError:
                pass
            except Exception as e:
                raise e
        return self

    def del_energy_kwargs(self, **args: str) -> Self:
        for key in args:
            try:
                del self._energy_kwargs[key]
            except KeyError:
                pass
            except Exception as e:
                raise e
        return self

    def del_reset_kwargs(self, **args: str) -> Self:
        for key in args:
            try:
                del self._reset_kwargs[key]
            except KeyError:
                pass
            except Exception as e:
                raise e
        return self


class OpInput:

    _args: tuple[NodeSrc | OpSrc | InputSrc | Any, ...]
    _kwargs: dict[str, NodeSrc | OpSrc | InputSrc | Any]

    def __init__(
        self,
        /,
        *args: NodeSrc | OpSrc | InputSrc | Any,
        **kwargs: NodeSrc | OpSrc | InputSrc | Any,
    ) -> None:

        nodes = set()
        ops = set()

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, NodeSrc):
                if arg.name in ops:
                    raise RuntimeError(f"input `{arg}` conflicts with previous input")
                nodes.add(arg.name)
            elif isinstance(arg, OpSrc):
                if arg.name in nodes:
                    raise RuntimeError(f"input `{arg}` conflicts with previous input")
                ops.add(arg.name)

        self._args = args
        self._kwargs = kwargs

    @property
    def args(self) -> tuple[NodeSrc | OpSrc | InputSrc | Any, ...]:
        return self._args

    @property
    def kwargs(self) -> MappingProxyType[str, NodeSrc | OpSrc | InputSrc | Any]:
        return MappingProxyType(self._kwargs)
