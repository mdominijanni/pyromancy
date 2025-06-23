import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..utils import mparameters
from .base import VariationalNode


class AbstractGaussianNode(VariationalNode, ABC):
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
        value (~torch.nn.parameter.Parameter): current value of the node.
    """

    def __init__(self, *shape: int | None) -> None:
        VariationalNode.__init__(self, *shape)

    @property
    @abstractmethod
    def covariance(self) -> torch.Tensor:
        r"""Covariance matrix of the Gaussian distribution.

        Args:
            value (float | ~torch.Tensor): new covariance for the distribution.

        Raises:
            NotImplementedError: must be implemented by subclasses.

        Returns:
            ~torch.Tensor: covariance of the distribution.
        """
        raise NotImplementedError

    @covariance.setter
    @abstractmethod
    def covariance(self, value: float | torch.Tensor) -> None:
        raise NotImplementedError


class StandardGaussianNode(AbstractGaussianNode):
    r"""Gaussian predictive coding node with unit variance.

    Assumes the covariance matrix is an identity matrix.

    .. math::
        \boldsymbol{\Sigma} = \mathbf{I}

    Args:
        *shape (int | None): shape of the node's learned state.

    Attributes:
        value (~torch.nn.parameter.Parameter): value of the node :math:`\mathbf{z}`.
    """

    def __init__(self, *shape: int | None) -> None:
        AbstractGaussianNode.__init__(self, *shape)

    @property
    def covariance(self) -> torch.Tensor:
        r"""Covariance matrix of the Gaussian distribution.

        .. math::
            \boldsymbol{\Sigma} = \mathbf{I}

        Args:
            value (float | ~torch.Tensor): new covariance for the distribution.

        Raises:
            RuntimeError: covariance is a fixed value.

        Returns:
            ~torch.Tensor: covariance of the distribution.
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
            pred (~torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        return self.value - pred

    def energy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Variational free energy with respect to the prediction.

        .. math::
            \begin{aligned}
                \mathcal{F}
                &= \frac{1}{2} (\mathbf{z} - \boldsymbol{\mu})
                (\mathbf{z} - \boldsymbol{\mu})^\intercal \\
                &= \frac{1}{2} \lVert\mathbf{z} - \boldsymbol{\mu}\rVert_2^2
            \end{aligned}

        Args:
            pred (~torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: variational free energy :math:`\mathcal{F}`.
        """
        diff = (self.value - pred).flatten(1)
        return 0.5 * (diff.unsqueeze(1) @ diff.unsqueeze(2)).flatten()

    def sample(
        self, value: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Samples from the learned variational distribution.

        Args:
            value (~torch.Tensor): location parameter of the variational distribution
                for sampling.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            ~torch.Tensor: samples from the variational distribution.
        """
        mu, pragma = self.shapeobj.coalesce(value)
        x = torch.randn(mu.shape, generator=generator, out=torch.empty_like(mu))
        return self.shapeobj.disperse(x, pragma)


@mparameters("logvar")
class IsotropicGaussianNode(AbstractGaussianNode):
    r"""Gaussian predictive coding node with scalar variance.

    Assumes the covariance matrix is a scalar matrix.

    .. math::
        \boldsymbol{\Sigma} = \sigma\mathbf{I}

    Args:
        *shape (int | None): shape of the node's learned state.
        variance (float | ~torch.Tensor, optional): initial variance. Defaults to 1.0.

    Attributes:
        value (~torch.nn.parameter.Parameter): value of the node :math:`\mathbf{z}`.
        logvar (~torch.nn.parameter.Parameter): log of the distribution variance :math:`\log{\sigma}`.
    """

    logvar: nn.Parameter

    def __init__(
        self, *shape: int | None, variance: float | torch.Tensor = 1.0
    ) -> None:
        AbstractGaussianNode.__init__(self, *shape)
        self.logvar = nn.Parameter(torch.empty([]), True)
        self.covariance = variance

    @property
    def covariance(self) -> torch.Tensor:
        r"""Covariance matrix of the Gaussian distribution.

        .. math::
            \boldsymbol{\Sigma} = \sigma\mathbf{I}

        Args:
            value (float | ~torch.Tensor): new covariance for the distribution.

        Returns:
            ~torch.Tensor: covariance of the distribution.

        Note:
            Assigment of variances is performed as follows:

            - 0D-Tensor (or float): single variance is used.
            - 1D-Tensor: vector of variances are averaged.
            - 2D-Tensor: diagonal of the covariance matrix is averaged.
        """
        return self.logvar.exp() * torch.eye(
            self.size, dtype=self.logvar.dtype, device=self.logvar.device
        )

    @covariance.setter
    @torch.no_grad()
    def covariance(self, value: float | torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            if not value > 0:
                raise ValueError("variance must be positive")

            self.logvar.fill_(math.log(value))

        else:
            match value.ndim:
                # scalar (isotropic multivariate)
                case 0:
                    if not value > 0:
                        raise ValueError("variance must be positive")

                    self.logvar.fill_(value.log())

                # vector (factorized multivariate)
                case 1:
                    if not value.numel() == self.size:
                        raise ValueError(
                            "`covariance` must be specified as a scalar, a vector of "
                            f"{self.size}, or a {self.size} x {self.size} matrix"
                        )
                    if not torch.all(value > 0):
                        raise ValueError(
                            "all elements of the variance vector must be positive"
                        )

                    self.logvar.fill_(value.mean().log())

                # matrix (full multivariate)
                case 2:
                    if not all(sz == self.size for sz in value.shape):
                        raise ValueError(
                            "`covariance` must be specified as a scalar, a vector of "
                            f"{self.size}, or a {self.size} x {self.size} matrix"
                        )

                    _, info = torch.linalg.cholesky_ex(value)

                    if not info.item() == 0:
                        raise ValueError(
                            "the covariance matrix must be "
                            "symmetric and positive-definite"
                        )

                    self.logvar.fill_(value.diag().mean().log())

                # invalid tensor dimensionality
                case _:
                    raise ValueError(
                        "`covariance` must be specified as a scalar, a vector of "
                        f"{self.size}, or a {self.size} x {self.size} matrix"
                    )

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Error between the prediction and node state.

        .. math::
            \boldsymbol{\varepsilon} = \frac{\mathbf{z} - \boldsymbol{\mu}}{\sigma}

        Args:
            pred (~torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        return (self.value - pred) / self.logvar.exp()

    def energy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Variational free energy with respect to the prediction.

        .. math::
            \begin{aligned}
                \mathcal{F}
                &= \frac{1}{2} \left((\mathbf{z} - \boldsymbol{\mu})
                ((\mathbf{z} - \boldsymbol{\mu}) \sigma^{-1})^\intercal
                + N \log \sigma\right) \\
                &= \frac{1}{2} \left(\frac{\lVert\mathbf{z} - \boldsymbol{\mu}\rVert_2^2}{\sigma}
                + N \log \sigma\right)
            \end{aligned}

        Args:
            pred (~torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: variational free energy :math:`\mathcal{F}`.
        """
        diff = (self.value - pred).flatten(1)
        y = diff / self.logvar.exp()
        logdet = self.size * self.logvar
        return 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()

    def sample(
        self, value: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Samples from the learned variational distribution.

        Args:
            value (~torch.Tensor): location parameter of the variational distribution
                for sampling.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            ~torch.Tensor: samples from the variational distribution.
        """
        mu, pragma = self.shapeobj.coalesce(value)
        std = self.logvar.exp().sqrt()
        x = std * torch.randn(mu.shape, generator=generator, out=torch.empty_like(mu))
        return self.shapeobj.disperse(x, pragma)


@mparameters("logvar")
class FactorizedGaussianNode(AbstractGaussianNode):
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
        value (~torch.nn.parameter.Parameter): value of the node :math:`\mathbf{z}`.
        logvar (~torch.nn.parameter.Parameter): log of the distribution variances :math:`\log{\boldsymbol{\sigma}}`.
    """

    logvar: nn.Parameter

    def __init__(
        self, *shape: int | None, variance: float | torch.Tensor = 1.0
    ) -> None:
        AbstractGaussianNode.__init__(self, *shape)
        self.logvar = nn.Parameter(torch.empty([self.size]), True)
        self.covariance = variance

    @property
    def covariance(self) -> torch.Tensor:
        r"""Covariance matrix of the Gaussian distribution.

        .. math::
            \boldsymbol{\Sigma} =
            \operatorname{diag}(\sigma_1, \sigma_2, \ldots, \sigma_N)

        Args:
            value (float | ~torch.Tensor): new covariance for the distribution.

        Returns:
            ~torch.Tensor: covariance of the distribution.

        Note:
            Assigment of variances is performed as follows:

            - 0D-Tensor (or float): single variance is used.
            - 1D-Tensor: vector of variances is used.
            - 2D-Tensor: diagonal of the covariance matrix is used.
        """
        return torch.diag(self.logvar.exp())

    @covariance.setter
    @torch.no_grad()
    def covariance(self, value: float | torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            if not value > 0:
                raise ValueError("variance must be positive")

            self.logvar.fill_(math.log(value))

        else:
            match value.ndim:
                # scalar (isotropic multivariate)
                case 0:
                    if not value > 0:
                        raise ValueError("variance must be positive")

                    self.logvar.fill_(value.log())

                # vector (factorized multivariate)
                case 1:
                    if not value.numel() == self.size:
                        raise ValueError(
                            "`covariance` must be specified as a scalar, a vector of "
                            f"{self.size}, or a {self.size} x {self.size} matrix"
                        )
                    if not torch.all(value > 0):
                        raise ValueError(
                            "all elements of the variance vector must be positive"
                        )

                    self.logvar.copy_(value.log())

                # matrix (full multivariate)
                case 2:
                    if not all(sz == self.size for sz in value.shape):
                        raise ValueError(
                            "`covariance` must be specified as a scalar, a vector of "
                            f"{self.size}, or a {self.size} x {self.size} matrix"
                        )

                    _, info = torch.linalg.cholesky_ex(value)

                    if not info.item() == 0:
                        raise ValueError(
                            "the covariance matrix must be "
                            "symmetric and positive-definite"
                        )

                    self.logvar.copy_(value.diag().log())

                # invalid tensor dimensionality
                case _:
                    raise ValueError(
                        "`covariance` must be specified as a scalar, a vector of "
                        f"{self.size}, or a {self.size} x {self.size} matrix"
                    )

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Error between the prediction and node state.

        .. math::
            \boldsymbol{\varepsilon} =
            (\mathbf{z} - \boldsymbol{\mu}) \oslash \boldsymbol{\sigma}

        Args:
            pred (~torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        return self.shapeobj.disperse(diff / self.logvar.exp(), pragma)

    def energy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Variational free energy with respect to the prediction.

        .. math::
            \mathcal{F} = \frac{1}{2} \left(
            (\mathbf{z} - \boldsymbol{\mu})
            ((\mathbf{z} - \boldsymbol{\mu}) \oslash \boldsymbol{\sigma})^\intercal
            + N \log \sigma\right)

        Args:
            pred (~torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: variational free energy :math:`\mathcal{F}`.
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        y = diff / self.logvar.exp()

        diff = self.shapeobj.disperse(diff, pragma).flatten(1)
        y = self.shapeobj.disperse(y, pragma).flatten(1)
        logdet = self.logvar.sum()

        return 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()

    def sample(
        self, value: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Samples from the learned variational distribution.

        Args:
            value (~torch.Tensor): location parameter of the variational distribution
                for sampling.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            ~torch.Tensor: samples from the variational distribution.
        """
        mu, pragma = self.shapeobj.coalesce(value)
        std = self.logvar.exp().sqrt()
        x = std * torch.randn(mu.shape, generator=generator, out=torch.empty_like(mu))
        return self.shapeobj.disperse(x, pragma)


@mparameters("covar_cf_logdiag", "covar_cf_offtril")
class MultivariateGaussianNode(AbstractGaussianNode):
    r"""Gaussian predictive coding node with full covariance.

    The covariances of the distribution are represented as a full covariance matrix,
    that is, a matrix that is symmetric and positive-definite.

    Internally, the covariance matrix is stored as two parts that can be combined into
    the Cholesky factor :math:`\mathbf{L}` of the covariance matrix :math:`\boldsymbol{\Sigma}`.

    .. math::
        \boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\ast

    Args:
        *shape (int | None): shape of the node's learned state.
        variance (float, optional): initial variance. Defaults to 1.0.

    Attributes:
        value (~torch.nn.parameter.Parameter): value of the node :math:`\mathbf{z}`.
        covar_cf_logdiag (~torch.nn.parameter.Parameter): log of the diagonal of the
            Cholesky factor for the distribution covariance.
        covar_cf_offtril (~torch.nn.parameter.Parameter): Cholesky factor for the
            distribution covariances, with the diagonal zeroed.
    """

    covar_cf_logdiag: nn.Parameter
    covar_cf_offtril: nn.Parameter

    def __init__(
        self, *shape: int | None, covariance: float | torch.Tensor = 1.0
    ) -> None:
        AbstractGaussianNode.__init__(self, *shape)
        self.covar_cf_logdiag = nn.Parameter(torch.empty([self.size]), True)
        self.covar_cf_offtril = nn.Parameter(torch.empty([self.size, self.size]), True)
        self.covariance = covariance

    def _cholesky_factor_l(self) -> torch.Tensor:
        r"""Computes the Cholesky decomposition factor :math:`L` of the covariance matrix.

        Returns:
            ~torch.Tensor: Cholesky factor :math:`L`.
        """
        return self.covar_cf_offtril.tril(-1) + self.covar_cf_logdiag.exp().diag()

    @property
    def covariance(self) -> torch.Tensor:
        r"""Covariance matrix of the Gaussian distribution.

        .. math::
            \boldsymbol{\Sigma} =
            \begin{bmatrix}
                \sigma_{1,1} & \sigma_{1,2} & \cdots & \sigma_{1,N} \\
                \sigma_{2,1} & \sigma_{2,2} & \cdots & \sigma_{2,N} \\
                \vdots & \vdots & \ddots & \vdots \\
                \sigma_{N,1} & \sigma_{N,2} & \cdots & \sigma_{N,N} \\
            \end{bmatrix}

        Args:
            value (float | ~torch.Tensor): new covariance for the distribution.

        Returns:
            ~torch.Tensor: covariance of the distribution.

        Note:
            Assigment of covariances is performed as follows:

            - 0D-Tensor (or float): single variance is used, with zero covariance.
            - 1D-Tensor: vector of variances is used, with zero covariance.
            - 2D-Tensor: covariance matrix is used.
        """
        L = self._cholesky_factor_l()
        return L @ L.t()

    @covariance.setter
    @torch.no_grad()
    def covariance(self, value: float | torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            if not value > 0:
                raise ValueError("variance must be positive")

            self.covar_cf_logdiag.fill_(math.log(math.sqrt(value)))
            self.covar_cf_offtril.fill_(0.0)

        else:
            match value.ndim:
                # scalar (isotropic multivariate)
                case 0:
                    if not value > 0:
                        raise ValueError("variance must be positive")

                    self.covar_cf_logdiag.fill_(value.sqrt().log())
                    self.covar_cf_offtril.fill_(0.0)

                # vector (factorized multivariate)
                case 1:
                    if not all(sz == self.size for sz in value.shape):
                        raise ValueError(
                            "`covariance` must be specified as a scalar, a vector of "
                            f"{self.size}, or a {self.size} x {self.size} matrix"
                        )
                    if not torch.all(value > 0):
                        raise ValueError(
                            "all elements of the variance vector must be positive"
                        )

                    self.covar_cf_logdiag.copy_(value.sqrt().log())
                    self.covar_cf_offtril.fill_(0.0)

                # matrix (full multivariate)
                case 2:
                    if not all(sz == self.size for sz in value.shape):
                        raise ValueError(
                            "`covariance` must be specified as a scalar, a vector of "
                            f"{self.size}, or a {self.size} x {self.size} matrix"
                        )

                    L, info = torch.linalg.cholesky_ex(value)

                    if not info.item() == 0:
                        raise ValueError(
                            "the covariance matrix must be "
                            "symmetric and positive-definite"
                        )

                    self.covar_cf_logdiag.copy_(L.diag().log())
                    self.covar_cf_offtril.copy_(L).fill_diagonal_(0.0)

                # invalid tensor dimensionality
                case _:
                    raise ValueError(
                        "`covariance` must be specified as a scalar, a vector of "
                        f"{self.size}, or a {self.size} x {self.size} matrix"
                    )

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Error between the prediction and node state.

        .. math::
            \boldsymbol{\varepsilon} = \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal

        Args:
            pred (~torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        L = self._cholesky_factor_l()

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True)

        return self.shapeobj.disperse(y.t(), pragma)

    def energy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Variational free energy with respect to the prediction.

        .. math::
            \mathcal{F} = \frac{1}{2} \left(
            (\mathbf{z} - \boldsymbol{\mu})
            \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal
            + \log \lvert\boldsymbol{\Sigma}\rvert \right)

        Args:
            pred (~torch.Tensor): predicted distribution mean :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: variational free energy :math:`\mathcal{F}`.
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        L = self._cholesky_factor_l()

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True)

        diff = self.shapeobj.disperse(diff, pragma).flatten(1)
        y = self.shapeobj.disperse(y.t(), pragma).flatten(1)
        logdet = 2.0 * self.covar_cf_logdiag.sum()

        return 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()

    def sample(
        self, value: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Samples from the learned variational distribution.

        Args:
            value (~torch.Tensor): location parameter of the variational distribution
                for sampling.
            generator (torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            ~torch.Tensor: samples from the variational distribution.
        """
        L = self._cholesky_factor_l()
        mu, pragma = self.shapeobj.coalesce(value)
        x = L @ torch.randn(mu.shape, generator=generator, out=torch.empty_like(mu)).t()
        return self.shapeobj.disperse(x.t(), pragma)
