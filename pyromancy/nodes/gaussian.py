from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
from .base import VariationalNode
from ..utils import mparameters


class AbstractGaussian(VariationalNode, ABC):
    r"""Base class for predictive coding nodes modelling Gaussian distributions.

    A multivariate Gaussian distribution is described by the following probability density function:

    .. math::
        f(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) =
        \frac{1}{\sqrt{(2\pi)^N \lvert\boldsymbol{\Sigma}\rvert}}
        \exp \left(-\frac{1}{2} (\mathbf{z} - \boldsymbol{\mu})
        \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal \right)

    where :math:`\mathbf{x}` is a sample, :math:`\boldsymbol{\mu}` is the mean,
    and :math:`\boldsymbol{\Sigma}` is the covariance matrix,
    for an :math:`N`-dimensional distribution.

    Args:
        *shape (int | None): shape of the node's learned state.

    Attributes:
        value (nn.Parameter): current value of the node.
    """

    def __init__(self, *shape: int | None) -> None:
        VariationalNode.__init__(self, *shape)

    @property
    @abstractmethod
    def covariance(self) -> torch.Tensor:
        r"""Covariance matrix of the Gaussian distribution.

        Args:
            value (float | torch.Tensor): new covariance for the distribution.

        Raises:
            NotImplementedError: must be implemented by subclasses.

        Returns:
            torch.Tensor: covariance of the distribution.
        """
        raise NotImplementedError

    @covariance.setter
    @abstractmethod
    def covariance(self, value: float | torch.Tensor) -> None:
        raise NotImplementedError


class StandardGaussian(AbstractGaussian):
    r"""Gaussian predictive coding node with unit variance.

    Assumes the covariance matrix is an identity matrix.

    .. math::
        \boldsymbol{\Sigma} = \mathbf{I}

    Args:
        *shape (int | None): shape of the node's learned state.

    Attributes:
        value (nn.Parameter): value of the node :math:`\mathbf{z}`.
    """

    def __init__(self, *shape: int | None) -> None:
        AbstractGaussian.__init__(self, *shape)

    @property
    def covariance(self) -> torch.Tensor:
        r"""Covariance matrix of the Gaussian distribution.

        .. math::
            \boldsymbol{\Sigma} = \mathbf{I}

        Args:
            value (float | torch.Tensor): new covariance for the distribution.

        Raises:
            RuntimeError: covariance is a fixed value.

        Returns:
            torch.Tensor: covariance of the distribution.
        """
        return torch.eye(self.size, dtype=self.value.dtype, device=self.value.device)

    @covariance.setter
    def covariance(self, value: float | torch.Tensor) -> None:
        raise RuntimeError(f"{type(self).__name__} has fixed covariance")

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Error between the prediction and node state.

        .. math::
            \boldsymbol{\varepsilon} = \mathbf{z} - \boldsymbol{\mu}

        Args:
            pred (torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        return self.value - pred

    def energy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Variational free energy with respect to the prediction.

        .. math::
            \begin{aligned}
                F
                &= \frac{1}{2} (\mathbf{z} - \boldsymbol{\mu})
                (\mathbf{z} - \boldsymbol{\mu})^\intercal \\
                &= \frac{1}{2} \lVert\mathbf{z} - \boldsymbol{\mu}\rVert_2^2
            \end{aligned}

        Args:
            pred (torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: variational free energy :math:`F`.
        """
        diff = (self.value - pred).flatten(1)
        return 0.5 * (diff.unsqueeze(1) @ diff.unsqueeze(2)).flatten()


@mparameters("logvar")
class IsotropicGaussian(AbstractGaussian):
    r"""Gaussian predictive coding node with scalar variance.

    Assumes the covariance matrix is a scalar matrix.

    .. math::
        \boldsymbol{\Sigma} = \sigma\mathbf{I}

    Args:
        *shape (int | None): shape of the node's learned state.
        variance (float, optional): initial variance. Defaults to 1.0.

    Attributes:
        value (nn.Parameter): value of the node :math:`\mathbf{z}`.
        logvar (nn.Parameter): log of the distribution variance :math:`\log{\sigma}`.
    """

    logvar: nn.Parameter

    def __init__(self, *shape: int | None, variance: float = 1.0) -> None:
        assert variance > 0
        AbstractGaussian.__init__(self, *shape)
        self.logvar = nn.Parameter(torch.empty([]), True)

        with torch.no_grad():
            self.logvar.fill_(math.log(variance))

    @property
    def covariance(self) -> torch.Tensor:
        r"""Covariance matrix of the Gaussian distribution.

        .. math::
            \boldsymbol{\Sigma} = \sigma\mathbf{I}

        Args:
            value (float | torch.Tensor): new covariance for the distribution.

        Returns:
            torch.Tensor: covariance of the distribution.
        """
        return self.logvar.exp() * torch.eye(
            self.size, dtype=self.logvar.dtype, device=self.logvar.device
        )

    @covariance.setter
    @torch.no_grad()
    def covariance(self, value: float | torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            assert value > 0
            self.logvar.fill_(math.log(value))
        else:
            match value.ndim:
                case 0:
                    assert value > 0
                    self.logvar.fill_(value.log())
                case 1:
                    assert value.numel() == self.size
                    assert torch.all(value > 0)
                    self.logvar.fill_(value.mean().log())
                case 2:
                    assert all(sz == self.size for sz in value.shape)
                    _, info = torch.linalg.cholesky_ex(value)
                    assert info.item() == 0
                    self.logvar.fill_(value.diag().mean().log())
                case _:
                    assert value.ndim <= 2

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Error between the prediction and node state.

        .. math::
            \boldsymbol{\varepsilon} = \frac{\mathbf{z} - \boldsymbol{\mu}}{\sigma}

        Args:
            pred (torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        return (self.value - pred) / self.logvar.exp()

    def energy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Variational free energy with respect to the prediction.

        .. math::
            \begin{aligned}
                F
                &= \frac{1}{2} \left((\mathbf{z} - \boldsymbol{\mu})
                ((\mathbf{z} - \boldsymbol{\mu}) \sigma^{-1})^\intercal
                + N \log \sigma\right) \\
                &= \frac{1}{2} \left(\frac{\lVert\mathbf{z} - \boldsymbol{\mu}\rVert_2^2}{\sigma}
                + N \log \sigma\right)
            \end{aligned}

        Args:
            pred (torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: variational free energy :math:`F`.
        """
        diff = (self.value - pred).flatten(1)
        y = diff / self.logvar.exp()
        logdet = self.size * self.logvar
        return 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()


@mparameters("logvar")
class FactorizedGaussian(AbstractGaussian):
    r"""Gaussian predictive coding node with diagonal variances.

    Assumes the covariance matrix is a diagonal matrix.

    .. math::
        \boldsymbol{\Sigma} =
        \begin{bmatrix}
            \sigma_1 & 0 & \cdots & 0 \\
            0 & \sigma_2 & \cdots & 0 \\
            \vdots & \vdots & \ddots & \vdots \\
            0 & 0 & \cdots & \sigma_N
        \end{bmatrix}

    Args:
        *shape (int | None): shape of the node's learned state.
        variance (float, optional): initial variance. Defaults to 1.0.

    Attributes:
        value (nn.Parameter): value of the node :math:`\mathbf{z}`.
        logvar (nn.Parameter): log of the distribution variances :math:`\log{\boldsymbol{\sigma}}`.
    """

    logvar: nn.Parameter

    def __init__(self, *shape: int | None, variance: float = 1.0) -> None:
        assert variance > 0
        AbstractGaussian.__init__(self, *shape)
        self.logvar = nn.Parameter(torch.empty([self.size]), True)

        with torch.no_grad():
            self.logvar.fill_(math.log(variance))

    @property
    def covariance(self) -> torch.Tensor:
        r"""Covariance matrix of the Gaussian distribution.

        .. math::
            \boldsymbol{\Sigma} =
            \operatorname{diag}(\sigma_1, \sigma_2, \ldots, \sigma_N)

        Args:
            value (float | torch.Tensor): new covariance for the distribution.

        Returns:
            torch.Tensor: covariance of the distribution.
        """
        return torch.diag(self.logvar.exp())

    @covariance.setter
    @torch.no_grad()
    def covariance(self, value: float | torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            assert value > 0
            self.logvar.fill_(math.log(value))
        else:
            match value.ndim:
                case 0:
                    assert value > 0
                    self.logvar.fill_(value.log())
                case 1:
                    assert value.numel() == self.size
                    assert torch.all(value > 0)
                    self.logvar.copy_(value.log())
                case 2:
                    assert all(sz == self.size for sz in value.shape)
                    _, info = torch.linalg.cholesky_ex(value)
                    assert info.item() == 0
                    self.logvar.copy_(value.diag().log())
                case _:
                    assert value.ndim <= 2

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Error between the prediction and node state.

        .. math::
            \boldsymbol{\varepsilon} =
            (\mathbf{z} - \boldsymbol{\mu}) \oslash \boldsymbol{\sigma}

        Args:
            pred (torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        return self.shapeobj.disperse(diff / self.logvar.exp(), pragma)

    def energy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Variational free energy with respect to the prediction.

        .. math::
            F = \frac{1}{2} \left(
            (\mathbf{z} - \boldsymbol{\mu})
            ((\mathbf{z} - \boldsymbol{\mu}) \oslash \boldsymbol{\sigma})^\intercal
            + N \log \sigma\right)

        Args:
            pred (torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: variational free energy :math:`F`.
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        y = diff / self.logvar.exp()

        diff = self.shapeobj.disperse(diff, pragma).flatten(1)
        y = self.shapeobj.disperse(y, pragma).flatten(1)
        logdet = self.logvar.sum()

        return 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()


@mparameters("covar_ldl_logdiag", "covar_ldl_offtril")
class MultivariateGaussian(AbstractGaussian):
    r"""Gaussian predictive coding node with full covariance, using LDL decomposition.

    The covariances of the distribution are represented as a full covariance matrix,
    that is, a matrix that is symmetric and positive-definite.

    Internally, the covariance matrix is stored as two parts, :math:`\mathbf{L}`
    and :math:`\log \mathbf{D}`, based on the LDL decomposition of the covariance
    matrix :math:`\boldsymbol{\Sigma}`.

    .. math::
        \boldsymbol{\Sigma} = \mathbf{L}\mathbf{D}\mathbf{L}^\intercal

    Args:
        *shape (int | None): shape of the node's learned state.
        variance (float, optional): initial variance. Defaults to 1.0.

    Attributes:
        value (nn.Parameter): value of the node :math:`\mathbf{z}`.
        covar_ldl_logdiag (nn.Parameter): log of the diagonal matrix for the LDL
            decomposition of the distribution covariances.
        covar_ldl_offtril (nn.Parameter): lower triangular matrix for the LDL
            decomposition of the distribution covariances.
    """

    covar_ldl_logdiag: nn.Parameter
    covar_ldl_offtril: nn.Parameter

    def __init__(self, *shape: int | None, variance: float = 1.0) -> None:
        assert variance > 0
        AbstractGaussian.__init__(self, *shape)

        self.covar_ldl_logdiag = nn.Parameter(torch.empty([self.size]), True)
        self.covar_ldl_offtril = nn.Parameter(torch.empty([self.size, self.size]), True)

        with torch.no_grad():
            self.covar_ldl_logdiag.fill_(math.log(variance))
            self.covar_ldl_offtril.fill_(0.0).fill_diagonal_(1.0)

    def _ldl_factor_l(self) -> torch.Tensor:
        r"""Computes the LDL decomposition factor :math:`L` of the covariance matrix.

        Returns:
            torch.Tensor: LDL factor :math:`L`.
        """
        return self.covar_ldl_offtril.tril(-1) + torch.eye(
            self.size, out=torch.empty_like(self.covar_ldl_offtril)
        )

    def _ldl_factor_d(self) -> torch.Tensor:
        r"""Computes the LDL decomposition factor :math:`D` of the covariance matrix.

        Returns:
            torch.Tensor: LDL factor :math:`D`.
        """
        return self.covar_ldl_logdiag.exp().diag()

    @property
    def covariance(self) -> torch.Tensor:
        L = self._ldl_factor_l()
        D = self._ldl_factor_d()
        return L @ D @ L.t()

    @covariance.setter
    @torch.no_grad()
    def covariance(self, value: float | torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            assert value > 0
            self.covar_ldl_logdiag.fill_(math.log(value))
            self.covar_ldl_offtril.fill_(0.0).fill_diagonal_(1.0)
        else:
            match value.ndim:
                case 0:
                    assert value > 0
                    self.covar_ldl_logdiag.fill_(value.log())
                    self.covar_ldl_offtril.fill_(0.0).fill_diagonal_(1.0)
                case 1:
                    assert value.numel() == self.size
                    assert torch.all(value > 0)
                    self.covar_ldl_logdiag.copy_(value.log())
                    self.covar_ldl_offtril.fill_(0.0).fill_diagonal_(1.0)
                case 2:
                    assert all(sz == self.size for sz in value.shape)
                    _, info = torch.linalg.cholesky_ex(value)
                    assert info.item() == 0
                    LD, pivots, info = torch.linalg.ldl_factor_ex(value)
                    assert info.item() == 0
                    assert torch.all(
                        pivots
                        == torch.arange(
                            1, self.size + 1, dtype=pivots.dtype, device=pivots.device
                        )
                    )
                    self.covar_ldl_logdiag.copy_(LD.diag().log())
                    self.covar_ldl_offtril.copy_(LD).fill_diagonal_(1.0)
                case _:
                    assert value.ndim <= 2

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Error between the prediction and node state.

        .. math::
            \boldsymbol{\varepsilon} = \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal

        Args:
            pred (torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.

        Note:
            Using :py:func:`torch.linalg.solve_triangular` with ``unitriangular=True``
            should keep the diagonal of :math:`\mathbf{L}` as 1, even with gradient updates.
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        L = self.covar_ldl_offtril
        d = self.covar_ldl_logdiag.exp()

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False, unitriangular=True)
        v = u / d.unsqueeze(-1)
        y = torch.linalg.solve_triangular(L.t(), v, upper=True, unitriangular=True)

        return self.shapeobj.disperse(y.t(), pragma)

    def energy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Variational free energy with respect to the prediction.

        .. math::
            F = \frac{1}{2} \left(
            (\mathbf{z} - \boldsymbol{\mu})
            \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal
            + \log \lvert\boldsymbol{\Sigma}\rvert \right)

        Args:
            pred (torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: variational free energy :math:`F`.

        Note:
            Using :py:func:`torch.linalg.solve_triangular` with ``unitriangular=True``
            should keep the diagonal of :math:`\mathbf{L}` as 1, even with gradient updates.
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        L = self.covar_ldl_offtril
        d = self.covar_ldl_logdiag.exp()

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False, unitriangular=True)
        v = u / d.unsqueeze(-1)
        y = torch.linalg.solve_triangular(L.t(), v, upper=True, unitriangular=True)

        diff = self.shapeobj.disperse(diff, pragma).flatten(1)
        y = self.shapeobj.disperse(y.t(), pragma).flatten(1)
        logdet = self.covar_ldl_logdiag.sum()

        return (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()
