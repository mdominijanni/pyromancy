import math
import torch
import torch.nn as nn
from .gaussian import AbstractGaussian
from ..utils import mparameters


@mparameters("covar_cf_logdiag", "covar_cf_offtril")
class CholeskyMultivariateGaussian(AbstractGaussian):
    r"""Gaussian predictive coding node with full covariance, using Cholesky decomposition.

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
        value (nn.Parameter): value of the node :math:`\mathbf{z}`.
        covar_cf_logdiag (nn.Parameter): log of the diagonal of the Cholesky factor for
            the distribution covariance.
        covar_cf_offtril (nn.Parameter): Cholesky factor for the distribution covariances,
            with the diagonal zeroed.
    """

    covar_cf_logdiag: nn.Parameter
    covar_cf_offtril: nn.Parameter

    def __init__(self, *shape: int | None, variance: float = 1.0) -> None:
        assert variance > 0
        AbstractGaussian.__init__(self, *shape)

        self.covar_cf_logdiag = nn.Parameter(torch.empty([self.size]), True)
        self.covar_cf_offtril = nn.Parameter(torch.empty([self.size, self.size]), True)

        with torch.no_grad():
            self.covar_cf_logdiag.fill_(math.log(math.sqrt(variance)))
            self.covar_cf_offtril.fill_(0.0)

    def _cholesky_factor_l(self) -> torch.Tensor:
        r"""Computes the Cholesky decomposition factor :math:`L` of the covariance matrix.

        Returns:
            torch.Tensor: Cholesky factor :math:`L`.
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
            value (float | torch.Tensor): new covariance for the distribution.

        Returns:
            torch.Tensor: covariance of the distribution.
        """
        L = self._cholesky_factor_l()
        return L @ L.t()

    @covariance.setter
    @torch.no_grad()
    def covariance(self, value: float | torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            assert value > 0
            self.covar_cf_logdiag.fill_(math.log(math.sqrt(value)))
            self.covar_cf_offtril.fill_(0.0)
        else:
            match value.ndim:
                case 0:
                    assert value > 0
                    self.covar_cf_logdiag.fill_(value.sqrt().log())
                    self.covar_cf_offtril.fill_(0.0)
                case 1:
                    assert value.numel() == self.size
                    assert torch.all(value > 0)
                    self.covar_cf_logdiag.copy_(value.sqrt().log())
                    self.covar_cf_offtril.fill_(0.0)
                case 2:
                    assert all(sz == self.size for sz in value.shape)
                    L, info = torch.linalg.cholesky_ex(value)
                    assert info.item() == 0
                    self.covar_cf_logdiag.copy_(L.diag().log())
                    self.covar_cf_offtril.copy_(L).fill_diagonal_(0.0)
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
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        L = self._cholesky_factor_l()

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True)

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
        """
        diff, pragma = self.shapeobj.coalesce(self.value - pred)
        L = self._cholesky_factor_l()

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True)

        diff = self.shapeobj.disperse(diff, pragma).flatten(1)
        y = self.shapeobj.disperse(y.t(), pragma).flatten(1)
        logdet = 2.0 * self.covar_cf_logdiag.sum()

        return 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()
