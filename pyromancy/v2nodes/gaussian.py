import math
from abc import ABC, abstractmethod
from typing import Any

import einops as ein
import torch
import torch.nn as nn

from ..params import mparameters
from .base import VariationalNode, ValueNodeMixin


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

        Returns:
            ~torch.Tensor: covariance of the distribution.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @covariance.setter
    @abstractmethod
    def covariance(self, value: float | torch.Tensor) -> None:
        raise NotImplementedError


@mparameters("cov_cf_logdiag", "cov_cf_offtril")
class MultivariateGaussianNode(ValueNodeMixin, AbstractGaussianNode):
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
        cov_cf_logdiag (~torch.nn.parameter.Parameter): log of the diagonal of the
            Cholesky factor for the distribution covariance.
        cov_cf_offtril (~torch.nn.parameter.Parameter): Cholesky factor for the
            distribution covariances, with the diagonal zeroed.
    """

    cov_cf_logdiag: nn.Parameter
    cov_cf_offtril: nn.Parameter

    _ln2pi: float

    def __init__(
        self, *shape: int | None, covariance: float | torch.Tensor = 1.0
    ) -> None:
        AbstractGaussianNode.__init__(self, *shape)
        ValueNodeMixin.__init__(self)

        self.cov_cf_logdiag = nn.Parameter(torch.empty(self.size), True)
        self.cov_cf_offtril = nn.Parameter(torch.empty(self.size, self.size), True)
        self._ln2pi = math.log(2.0 * math.pi)

        self.covariance = covariance

    def _cholesky_factor_cov(self) -> torch.Tensor:
        r"""Computes the Cholesky decomposition factor :math:`L` of the covariance matrix.

        Returns:
            ~torch.Tensor: Cholesky factor :math:`L`.
        """
        return self.cov_cf_offtril.tril(-1) + self.cov_cf_logdiag.exp().diag()

    def _logdet_cov(self) -> torch.Tensor:
        r"""Computes the log of the determinant of the covariance matrix.

        Returns:
            ~torch.Tensor: log determinant :math:`\log \lvert \boldsymbol{\Sigma} \rvert`.
        """
        return 2.0 * self.cov_cf_logdiag.sum()

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

        Raises:
            ValueError: ``covariance`` must be a scalar, vector, or square matrix,
                with a corresponding number of elements.
            ValueError: scalar variance must be positive.
            ValueError: all elements of variance vector must be positive.
            ValueError: covariance matrix must be symmetric and positive-definite.

        Note:
            Assigment of covariances is performed as follows:

            - 0D-Tensor (or float): single variance is used, with zero covariance.
            - 1D-Tensor: vector of variances is used, with zero covariance.
            - 2D-Tensor: covariance matrix is used.
        """
        L = self._cholesky_factor_cov()
        return L @ L.t()

    @covariance.setter
    @torch.no_grad()
    def covariance(self, value: float | torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            if not value > 0:
                raise ValueError("variance must be positive")

            self.cov_cf_logdiag.fill_(math.log(math.sqrt(value)))
            self.cov_cf_offtril.fill_(0.0)

        else:
            match value.ndim:
                # scalar (isotropic multivariate)
                case 0:
                    if not value > 0:
                        raise ValueError("variance must be positive")

                    self.cov_cf_logdiag.fill_(value.sqrt().log())
                    self.cov_cf_offtril.fill_(0.0)

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

                    self.cov_cf_logdiag.copy_(value.sqrt().log())
                    self.cov_cf_offtril.fill_(0.0)

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

                    self.cov_cf_logdiag.copy_(L.diag().log())
                    self.cov_cf_offtril.copy_(L).fill_diagonal_(0.0)

                # invalid tensor dimensionality
                case _:
                    raise ValueError(
                        "`covariance` must be specified as a scalar, a vector of "
                        f"{self.size}, or a {self.size} x {self.size} matrix"
                    )

    def prediction(self, *pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes the single prediction for the node.

        Args:
            *pred (~torch.Tensor): predictions for the basis of initialization.

        Returns:
            ~torch.Tensor: prediction of the node's activity.

        Important:
            Every input must have the same shape.
        """
        return ein.reduce(list(pred), "m ... -> ...", "mean")

    def energy(self, *pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes batchwise energy between predictions and the node's activity.

        .. math::
            \mathcal{E} = \frac{1}{2} \sum_{i=1}^{m}
            \left((\mathbf{z} - \boldsymbol{\mu}_i)
            \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu}_i)^\intercal
            + \log \lvert\boldsymbol{\Sigma}\rvert + 2 \pi n\right)

        Where :math:`n` is the number of variables in the multivariate distribution.

        Args:
            *pred (~torch.Tensor): predictions of the node's activity,
                :math:`\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_m`.

        Returns:
            ~torch.Tensor: batchwise energy between the activity and the predictions.

        Important:
            Every input must have the same shape.
        """
        m = len(pred)  # number of predictions
        n = self.shape.size  # number of variables
        p = self.activity.numel() / (self.activity.size(0) * n)  # number of features

        # quadratic
        z, pragma = self.shape.coalesce(self.activity)
        mu = torch.vmap(lambda t, s=self.shape: s.coalesce(t)[0])(torch.stack(pred, 0))

        L = self._cholesky_factor_cov()
        d = z - mu
        y = torch.cholesky_solve(d.unsqueeze(-1), L).squeeze(-1)

        d = torch.vmap(lambda t, p=pragma, s=self.shape: s.disperse(t, p))(d).flatten(2)
        y = torch.vmap(lambda t, p=pragma, s=self.shape: s.disperse(t, p))(y).flatten(2)

        q = d.unsqueeze(-2) @ y.unsqueeze(-1)
        q = 0.5 * ein.reduce(q, "m b ... -> b", "sum")

        # normalization
        logdet = 2.0 * L.diag().log().sum()
        norm = 0.5 * m * p * (logdet + n * self._ln2pi)

        # energy
        E = q + norm
        return E

    def error(self, *pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes elementwise error between predictions and the node's activity.

        .. math::
            \boldsymbol{\epsilon} = \sum_{i=1}^{m}
            \left(\boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu}_i)^\intercal
            \right)

        Args:
            *pred (~torch.Tensor): predictions of the node's activity,
                :math:`\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_m`.

        Returns:
            ~torch.Tensor: elementwise error between the activity and the predictions.

        Important:
            Every input must have the same shape.
        """
        z, pragma = self.shape.coalesce(self.activity)
        mu = torch.vmap(lambda t, s=self.shape: s.coalesce(t)[0])(torch.stack(pred, 0))

        L = self._cholesky_factor_cov()
        d = z - mu
        y = torch.cholesky_solve(d.unsqueeze(-1), L).squeeze(-1)

        err = torch.vmap(lambda t, p=pragma, s=self.shape: s.disperse(t, p))(y)
        err = ein.reduce(err, "m ... -> ...", "sum")

        return err

    def sample(
        self,
        *pred: torch.Tensor,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Samples from the learned conditional distribution.

        Args:
            *pred (~torch.Tensor): predictions of the node's activity, used as
                parameters of the distribution.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: samples from the conditional distribution.

        Important:
            Every input must have the same shape.
        """
        m = len(pred)  # number of predictions

        # PoG mean
        mu = self.prediction(*pred, **kwargs)
        mu, pragma = self.shape.coalesce(mu)

        # white noise
        L = self._cholesky_factor_cov()
        eps = torch.randn(mu.shape, generator=generator, out=torch.empty_like(mu))

        # color noise and shift
        z = torch.addmm(mu, eps, L.t(), alpha=(1 / math.sqrt(m)))
        return self.shape.disperse(z, pragma)
