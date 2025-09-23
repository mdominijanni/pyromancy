from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar, overload

import torch.nn as nn

T = TypeVar("T")


@overload
def _get_declared_estep_params(obj: object | type, /) -> dict[str, None]: ...
@overload  # noqa:E302
def _get_declared_estep_params(
    obj: object | type, /, *default: T
) -> dict[str, None] | T: ...


def _get_declared_estep_params(  # noqa:E302
    obj: object | type, /, *default: T
) -> dict[str, None] | T:
    r"""Get all dynamic and declared E-step parameter names in the MRO chain.

    Args:
        obj (object | type): object or class to find E-step parameters for.
        default (T, optional): default return value. Defaults to an empty :py:class:`dict`.

    Returns:
        dict[str, None] | None: E-step parameters names if any are declared, otherwise ``None``.
    """
    params: dict[str, None] = {}
    found = False

    if not isinstance(obj, type):
        chain = (obj, *type(obj).__mro__)
    else:
        chain = (*obj.__mro__,)

    for c in chain:
        p = c.__dict__.get("_e_params_", None)
        if p is not None:
            params |= p
            found = True

    if default:
        return params if found else default[0]
    else:
        return params


@overload
def _get_declared_mstep_params(obj: object | type, /) -> dict[str, None]: ...
@overload  # noqa:E302
def _get_declared_mstep_params(
    obj: object | type, /, *default: T
) -> dict[str, None] | T: ...


def _get_declared_mstep_params(  # noqa:E302
    obj: object | type, /, *default: T
) -> dict[str, None] | T:
    r"""Get all dynamic and declared M-step parameter names in the MRO chain.

    Args:
        obj (object | type): object or class to find M-step parameters for.
        default (T, optional): default return value. Defaults to an empty :py:class:`dict`.

    Returns:
        dict[str, None] | None: M-step parameters names if any are declared, otherwise ``None``.
    """
    params: dict[str, None] = {}
    found = False

    if not isinstance(obj, type):
        chain = (obj, *type(obj).__mro__)
    else:
        chain = (*obj.__mro__,)

    for c in chain:
        p = c.__dict__.get("_m_params_", None)
        if p is not None:
            params |= p
            found = True

    if default:
        return params if found else default[0]
    else:
        return params


def set_dynamic_estep_params(obj: nn.Module, *attrs: str) -> None:
    r"""Sets the E-step parameters for an object.

    Args:
        obj (nn.Module): object to set E-step parameters for.
        *attrs (str): object attributes to register as E-step parameters.

    Raises:
        TypeError: ``obj`` must be a PyTorch :py:class:`~torch.nn.Module`.
        TypeError: all elements of ``attrs`` must be strings.

    Tip:
        Generally this should just be set once in the initializer for an object,
        where which parameters are trainable is based on that object's configuration.
    """
    if not isinstance(obj, nn.Module):
        raise TypeError("`obj` must be of type `torch.nn.Module`")
    if not all(isinstance(a, str) for a in attrs):
        raise TypeError("all elements of `attrs` must be of type str")

    obj._e_params_ = {a: None for a in attrs}  # type: ignore


def set_dynamic_mstep_params(obj: nn.Module, *attrs: str) -> None:
    r"""Sets the M-step parameters for an object.

    Args:
        obj (nn.Module): object to set M-step parameters for.
        *attrs (str): object attributes to register as M-step parameters.

    Raises:
        TypeError: ``obj`` must be a PyTorch :py:class:`~torch.nn.Module`.
        TypeError: all elements of ``attrs`` must be strings.

    Tip:
        Generally this should just be set once in the initializer for an object,
        where which parameters are trainable is based on that object's configuration.
    """
    if not isinstance(obj, nn.Module):
        raise TypeError("`obj` must be of type `torch.nn.Module`")
    if not all(isinstance(a, str) for a in attrs):
        raise TypeError("all elements of `attrs` must be of type str")

    obj._m_params_ = {a: None for a in attrs}  # type: ignore


def eparameters(*fields: str) -> Callable[[type[T]], type[T]]:
    r"""Sets the E-step parameters for a class.

    Returns:
        ~collections.abc.Callable[[type[T]], type[T]]: class decorator.

    Raises:
        TypeError: all elements of ``fields`` must be strings.

    Important:
        If a class inherits from multiple classes defining E-step parameters, then even
        if it does not directly specify any E-step parameters it should still be
        decorated with ``@eparameters()`` to collate all superclass E-step parameters.

    Tip:
        This defines the maximal set of possible E-step parameters. If a class may or
        may not contain a parameter that should be updated on E-steps, it should still
        be included here.

    Tip:
        E-step parameters can be set on any class, which may be useful in some cases
        (e.g., when defining a mixin class), but the parameters can only be accessed
        from objects that inherit from :py:class:`~torch.nn.Module`.
    """

    def decorator_eparameters(cls: type[T]) -> type[T]:
        if not all(isinstance(f, str) for f in fields):
            raise TypeError("all elements of `fields` must be of type str")

        if "_e_params_" not in cls.__dict__:
            cls._e_params_ = {f: None for f in fields} | _get_declared_estep_params(cls)  # type: ignore
            cls.__annotations__["_e_params_"] = dict[str, None]

        return cls

    return decorator_eparameters


def mparameters(*fields: str) -> Callable[[type[T]], type[T]]:
    r"""Sets the M-step parameters for a class.

    Returns:
        ~collections.abc.Callable[[type[T]], type[T]]: class decorator.

    Raises:
        TypeError: all elements of ``fields`` must be strings.

    Important:
        If a class inherits from multiple classes defining M-step parameters, then even
        if it does not directly specify any M-step parameters it should still be
        decorated with ``@mparameters()`` to collate all superclass M-step parameters.

    Tip:
        This defines the maximal set of possible M-step parameters. If a class may or
        may not contain a parameter that should be updated on M-steps, it should still
        be included here.

    Tip:
        M-step parameters can be set on any class, which may be useful in some cases
        (e.g., when defining a mixin class), but the parameters can only be accessed
        from objects that inherit from :py:class:`~torch.nn.Module`.
    """

    def decorator_mparameters(cls: type[T]) -> type[T]:
        if not all(isinstance(f, str) for f in fields):
            raise TypeError("all elements of `fields` must be of type str")

        if "_m_params_" not in cls.__dict__:
            cls._m_params_ = {f: None for f in fields} | _get_declared_mstep_params(cls)  # type: ignore
            cls.__annotations__["_m_params_"] = dict[str, None]

        return cls

    return decorator_mparameters


def get_named_estep_params(
    module: nn.Module,
    default: bool = False,
    exclude: Sequence[nn.Parameter | nn.Module] | None = None,
    prefix: str = "",
    recurse: bool = True,
    remove_duplicate=True,
) -> Iterator[tuple[str, nn.Parameter]]:
    r"""Returns an iterator over E-step parameters, yielding both the name of
    the parameter and the parameter itself.

    Args:
        module (~torch.nn.Module): module from which to retrieve E-step parameters.
        exclude (~collections.abc.Sequence[~torch.nn.parameter.Parameter | ~torch.nn.Module] | None):
            parameters and modules to exclude. Defaults to None.
        default (bool, optional): if unspecified parameters should default to E-step parameters.
            Defaults to False.
        prefix (str, optional): prefix to prepend to all parameter names.
            Defaults to "".
        recurse (bool, optional): if parameters that are not direct members should be included.
            Defaults to True.
        remove_duplicate (bool, optional): if duplicated parameters should be excluded.
            Defaults to True.

    Yields:
        tuple[str, torch.nn.Parameter]: tuple containing the name and parameter.

    Raises:
        TypeError: all elements of ``exclude`` must be of type
            :py:class:`~torch.nn.parameter.Parameter` or :py:class:`~torch.nn.Module`.

    Note:
        Resolution is performed as follows:

        - if ``_e_params_`` is defined and the identifier for a parameter is in ``_e_params_``,
          then the parameter is included.
        - if ``_e_params_`` is not defined but ``_m_params_`` is, and the identifier is in ``_m_params_``,
          then the parameter is excluded.
        - if ``_e_params_`` is not defined and ``_m_params_``, if present, does not contain the identifier,
          then the parameter is included if ``default`` is true and excluded if it is false.

        This resolution is performed on the combined ``_e_params_`` from the given module's
        class and all its superclasses.

    Note:
        The E-step parameters for a class that inherits from :py:class:`~torch.nn.Module`
        are determined by the class attribute ``_e_params_``, containing a list of
        attribute names.
    """
    if exclude is None:
        exclude = ()

    memo = set()
    for item in exclude:
        if isinstance(item, nn.Parameter):
            memo |= {item}
        elif isinstance(item, nn.Module):
            memo |= {*item.parameters()}
        else:
            raise TypeError(
                "all elements of `exclude` must be of type "
                "`torch.nn.Parameter` or of type `torch.nn.Module"
            )

    if recurse:
        modules = module.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
    else:
        modules = [(prefix, module)]

    for p, m in modules:
        eparams = _get_declared_estep_params(m, None)
        if eparams is not None:
            eparams = frozenset(eparams)
        mparams = _get_declared_mstep_params(m, None)
        if mparams is not None:
            mparams = frozenset(mparams)

        params = m._parameters.items()
        for k, v in params:
            # skip none and memoized parameters
            if v is None or v in memo:
                continue
            # skip if explicitly not in e-step parameters
            if eparams is not None and k not in eparams:
                continue
            # skip if explicitly in m-step parameters
            if eparams is None and mparams is not None and k in mparams:
                continue
            # skip if defaulting to false
            if eparams is None and not default:
                continue
            # memoize
            if remove_duplicate:
                memo.add(v)
            # yield parameter
            name = p + ("." if p else "") + k
            yield name, v


def get_estep_params(
    module: nn.Module,
    default: bool = False,
    exclude: Sequence[nn.Parameter | nn.Module] | None = None,
    recurse: bool = True,
) -> Iterator[nn.Parameter]:
    r"""Returns an iterator over E-step parameters.

    Args:
        module (~torch.nn.Module): module from which to retrieve E-step parameters.
        default (bool, optional): if unspecified parameters should default to E-step parameters.
            Defaults to False.
        exclude (~collections.abc.Sequence[~torch.nn.parameter.Parameter | ~torch.nn.Module] | None):
            parameters and modules to exclude. Defaults to None.
        recurse (bool, optional): if parameters that are not direct members should be included.
            Defaults to True.

    Yields:
        torch.nn.Parameter: E-step parameter.

    Note:
        Resolution is performed as follows:

        - if ``_e_params_`` is defined and the identifier for a parameter is in ``_e_params_``,
          then the parameter is included.
        - if ``_e_params_`` is not defined but ``_m_params_`` is, and the identifier is in ``_m_params_``,
          then the parameter is excluded.
        - if ``_e_params_`` is not defined and ``_m_params_``, if present, does not contain the identifier,
          then the parameter is included if ``default`` is true and excluded if it is false.

        This resolution is performed on the combined ``_e_params_`` from the given module's
        class and all its superclasses.

    Note:
        The E-step parameters for a class that inherits from :py:class:`~torch.nn.Module`
        are determined by the class attribute ``_e_params_``, containing a dictionary of
        attribute names with ``None`` values.
    """
    for _, p in get_named_estep_params(
        module, default, exclude, recurse=recurse, remove_duplicate=True
    ):
        yield p


def get_named_mstep_params(
    module: nn.Module,
    default: bool = True,
    exclude: Sequence[nn.Parameter | nn.Module] | None = None,
    prefix: str = "",
    recurse: bool = True,
    remove_duplicate=True,
) -> Iterator[tuple[str, nn.Parameter]]:
    r"""Returns an iterator over M-step parameters, yielding both the name of
    the parameter and the parameter itself.

    Args:
        module (~torch.nn.Module): module from which to retrieve M-step parameters.
        default (bool, optional): if unspecified parameters should default to M-step parameters.
            Defaults to True.
        exclude (~collections.abc.Sequence[~torch.nn.parameter.Parameter | ~torch.nn.Module] | None):
            parameters and modules to exclude. Defaults to None.
        prefix (str, optional): prefix to prepend to all parameter names.
            Defaults to "".
        recurse (bool, optional): if parameters that are not direct members should be included.
            Defaults to True.
        remove_duplicate (bool, optional): if duplicated parameters should be excluded.
            Defaults to True.

    Yields:
        tuple[str, torch.nn.Parameter]: tuple containing the name and parameter.

    Raises:
        TypeError: all elements of ``exclude`` must be of type
            :py:class:`~torch.nn.parameter.Parameter` or :py:class:`~torch.nn.Module`.

    Note:
        Resolution is performed as follows:

        - if ``_m_params_`` is defined and the identifier for a parameter is in ``_m_params_``,
          then the parameter is included.
        - if ``_m_params_`` is not defined but ``_e_params_`` is, and the identifier is in ``_e_params_``,
          then the parameter is excluded.
        - if ``_m_params_`` is not defined and ``_e_params_``, if present, does not contain the identifier,
          then the parameter is included if ``default`` is true and excluded if it is false.

        This resolution is performed on the combined ``_m_params_`` from the given module's
        class and all its superclasses.

    Note:
        The M-step parameters for a class that inherits from :py:class:`~torch.nn.Module`
        are determined by the class attribute ``_m_params_``, containing a dictionary of
        attribute names with ``None`` values.
    """
    if exclude is None:
        exclude = ()

    memo = set()
    for item in exclude:
        if isinstance(item, nn.Parameter):
            memo |= {item}
        elif isinstance(item, nn.Module):
            memo |= {*item.parameters()}
        else:
            raise TypeError(
                "all elements of `exclude` must be of type "
                "`torch.nn.Parameter` or of type `torch.nn.Module"
            )

    if recurse:
        modules = module.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
    else:
        modules = [(prefix, module)]

    for p, m in modules:
        eparams = _get_declared_estep_params(m, None)
        if eparams is not None:
            eparams = frozenset(eparams)
        mparams = _get_declared_mstep_params(m, None)
        if mparams is not None:
            mparams = frozenset(mparams)

        params = m._parameters.items()
        for k, v in params:
            # skip none and memoized parameters
            if v is None or v in memo:
                continue
            # skip if explicitly not in m-step parameters
            if mparams is not None and k not in mparams:
                continue
            # skip if explicitly in e-step parameters
            if mparams is None and eparams is not None and k in eparams:
                continue
            # skip if defaulting to false
            if mparams is None and not default:
                continue
            # memoize
            if remove_duplicate:
                memo.add(v)
            # yield parameter
            name = p + ("." if p else "") + k
            yield name, v


def get_mstep_params(
    module: nn.Module,
    default: bool = True,
    exclude: Sequence[nn.Parameter | nn.Module] | None = None,
    recurse: bool = True,
) -> Iterator[nn.Parameter]:
    r"""Returns an iterator over M-step parameters.

    Args:
        module: module from which to retrieve M-step parameters.
        default (bool, optional): if unspecified parameters should default to M-step parameters.
            Defaults to True.
        exclude (~collections.abc.Sequence[~torch.nn.parameter.Parameter | ~torch.nn.Module] | None):
            parameters and modules to exclude. Defaults to None.
        recurse (bool, optional): if parameters that are not direct members should be included.
            Defaults to True.

    Yields:
        torch.nn.Parameter: M-step parameter.

    Note:
        Resolution is performed as follows:

        - if ``_m_params_`` is defined and the identifier for a parameter is in ``_m_params_``,
          then the parameter is included.
        - if ``_m_params_`` is not defined but ``_e_params_`` is, and the identifier is in ``_e_params_``,
          then the parameter is excluded.
        - if ``_m_params_`` is not defined and ``_e_params_``, if present, does not contain the identifier,
          then the parameter is included if ``default`` is true and excluded if it is false.

        This resolution is performed on the combined ``_m_params_`` from the given module's
        class and all its superclasses.

    Note:
        The M-step parameters for a class that inherits from :py:class:`~torch.nn.Module`
        are determined by the class attribute ``_m_params_``, containing a list of
        attribute names.
    """
    for _, p in get_named_mstep_params(
        module, default, exclude, recurse=recurse, remove_duplicate=True
    ):
        yield p
