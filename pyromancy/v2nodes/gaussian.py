import math
from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
import torch.nn as nn

from ..params import eparameters, mparameters
from .base import VariationalNode


@eparameters("mean")
class AbstractGaussianNode(VariationalNode, ABC):
    r"""Base class for predictive coding nodes modelling Gaussian distributions.

    A multivariate Gaussian distribution is described by the following probability density function:

    .. math::
        f(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) =
        \frac{1}{\sqrt{(2\pi)^k \lvert\boldsymbol{\Sigma}\rvert}}
        \exp \left(-\frac{1}{2} (\mathbf{z} - \boldsymbol{\mu})
        \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal \right)

    where :math:`\mathbf{x}` is a sample, :math:`\boldsymbol{\mu}` is the mean,
    and :math:`\boldsymbol{\Sigma}` is the covariance matrix,
    for a :math:`k`-dimensional distribution.

    Args:
        *shape (int | None): shape of the node's learned state.

    Attributes:
        mean (~torch.nn.parameter.Parameter): current value of the node :math:`\mathbf{z}`.
    """

    mean: nn.Parameter
    _ln2pi: float
    _nvariate: float

    def __init__(self, *shape: int | None, **kwargs: Any) -> None:
        VariationalNode.__init__(self, *shape, **kwargs)
        self.mean = nn.Parameter(torch.empty(0), True)
        self._ln2pi = math.log(2.0 * math.pi)
        self._nvariate = self.shapeobj.size

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

    def activity(self, **kwargs) -> nn.Parameter:
        r"""Activity of the node.

        Returns:
            ~torch.Tensor: activity of the node.
        """
        return self.mean

    @torch.no_grad()
    def initialize(
        self,
        pred: torch.Tensor,
        sample: bool = False,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> nn.Parameter:
        r"""Initializes the node's state.

        Args:
            pred (~torch.Tensor): prediction for the basis of initialization.
            sample (bool, optional): if the activity should be initialized using
                random sampling. Defaults to False.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.
        """

        if sample:
            mu = self.sample(pred, generator=generator, **kwargs)
        else:
            mu = self.prediction(pred, **kwargs)

        if not self.shapeobj.compat(*mu.shape):
            raise ValueError(
                f"shape specified by `pred` {(*mu.shape,)} "
                f"is incompatible with node shape {(*self.shapeobj,)}"
            )

        self.mean.data = self.mean.data.new_empty(*mu.shape)
        self.mean.copy_(mu)

        return self.mean

    def prediction(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Transforms a raw prediction into one compatible with the node.

        Args:
            pred (~torch.Tensor): raw prediction from the parent.

        Returns:
            ~torch.Tensor: expanded bias vector.
        """
        return pred

    @torch.no_grad()
    def reset(self, **kwargs: Any) -> None:
        r"""Resets the node state."""
        self.zero_grad()
        self.mean.data = self.mean.new_empty(0)


class StandardGaussianNode(AbstractGaussianNode):
    r"""Gaussian predictive coding node with unit variance.

    Assumes the covariance matrix is an identity matrix.

    .. math::
        \boldsymbol{\Sigma} = \mathbf{I}

    Args:
        *shape (int | None): shape of the node's learned state.

    Attributes:
        mean (~torch.nn.parameter.Parameter): current value of the node :math:`\mathbf{z}`.
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

        Returns:
            ~torch.Tensor: covariance of the distribution.

        Raises:
            RuntimeError: covariance is a fixed value.
        """
        return torch.eye(self.size, dtype=self.mean.dtype, device=self.mean.device)

    @covariance.setter
    def covariance(self, value: float | torch.Tensor) -> None:
        raise RuntimeError(f"{type(self).__name__} has fixed covariance")

    def energy(
        self, pred: torch.Tensor, fn: Literal["nll", "kld", "ce"] = "nll", **kwargs: Any
    ) -> torch.Tensor:
        r"""Computes batchwise energy between predictions and the node's activity.

        .. math::
            \begin{aligned}
                \mathcal{E}_\text{NLL} &= \frac{1}{2}
                \lVert \mathbf{z} - \boldsymbol{\mu} \rVert_2^2
                + \frac{k}{2} \log 2 \pi \\
                \mathcal{E}_\text{KL} &= \frac{1}{2}
                \lVert \mathbf{z} - \boldsymbol{\mu} \rVert_2^2 \\
                \mathcal{E}_\text{CE} &= \frac{1}{2}
                \lVert \mathbf{z} - \boldsymbol{\mu} \rVert_2^2
                + \frac{k}{2} \log 2 \pi + \frac{k}{2}
            \end{aligned}

        For a :math:`k`-variate distribution.

        Args:
            pred (~torch.Tensor): prediction of the node's activity, :math:`\boldsymbol{\mu}`.
            fn (Literal["nll", "kld", "ce"], optional): mode for computing the energy.
                Defaults to "nll".

        Returns:
            ~torch.Tensor: batchwise energy between the activity and the predictions.

        Raises:
            ValueError: invalid ``fn`` specified.

        Info:
            The `fn` parameter controls how energy is computed.

            - Negative Log-Likelihood ("nll"): the node's activity is treated as a point
              estimate of the distribution, and NLL is taken.
            - Kullback–Leibler Divergence ("kld"): the node's activity is treated as the
              as the distribution's mean, and reverse KL-divergence is taken.
            - Cross-Entropy ("ce"): the node's activity is treated as the
              as the distribution's mean, and cross-entropy is taken. This is the
              expected value of NLL.

            Since cross-entropy is evaluated in closed-form for multivariate Gaussians,
            this is equivalent to negative log-likelihood up to a constant.
        """
        # quadratic loss term (common to all methods)
        mu = self.prediction(pred, **kwargs)
        diff, pragma = self.shapeobj.coalesce(self.mean - mu)
        quad = 0.5 * (diff.unsqueeze(1) @ diff.unsqueeze(2)).flatten()

        # method dependent normalization
        match fn:
            case "nll":
                norm = 0.5 * (self._nvariate * self._ln2pi)
                E = quad + norm
            case "kld":
                E = quad
            case "ce":
                norm = 0.5 * (self._nvariate * (1.0 + self._ln2pi))
                E = quad + norm
            case _:
                raise ValueError(f"invalid `fn` of '{fn}' specified")

        # reshape and sum over plates
        E = self.shapeobj.disperse(E, pragma, "plate")
        return E.flatten(1).sum(1)

    def error(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes elementwise error between predictions and the node's activity.

        .. math::
            \boldsymbol{\epsilon} = \mathbf{z} - \boldsymbol{\mu}

        Args:
            pred (~torch.Tensor): prediction of the node's activity, :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error between the activity and the predictions.
        """
        mu = self.prediction(pred, **kwargs)
        return self.mean - mu

    def sample(
        self,
        pred: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Samples from the learned conditional distribution.

        Args:
            pred (~torch.Tensor | None, optional): prediction of the node's activity,
                used as the mean of the distribution if provided. If None, the node's
                activity is used for the prediction. Defaults to None.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            ~torch.Tensor: samples from the conditional distribution.
        """
        if pred is None:
            mu = self.mean
        else:
            mu = self.prediction(pred, **kwargs)

        mu, pragma = self.shapeobj.coalesce(mu)

        # white noise
        eps = torch.randn(mu.shape, generator=generator, out=torch.empty_like(mu))

        # shift
        z = mu + eps
        return self.shapeobj.disperse(z, pragma)


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
        mean (~torch.nn.parameter.Parameter): current value of the node :math:`\mathbf{z}`.
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

        Raises:
            ValueError: ``covariance`` must be a scalar, vector, or square matrix,
                with a corresponding number of elements.
            ValueError: scalar variance must be positive.
            ValueError: all elements of variance vector must be positive.
            ValueError: covariance matrix must be symmetric and positive-definite.


        Note:
            Assigment of variances is performed as follows:

            - 0D-Tensor (or float): single variance is used.
            - 1D-Tensor: vector of variances are averaged.
            - 2D-Tensor: diagonal of the covariance matrix is averaged.
        """
        return self.logvar.exp() * torch.eye(
            self.size, dtype=self.logvar.dtype, device=self.logvar.device
        )

    def _logdet_cov(self) -> torch.Tensor:
        r"""Computes the log of the determinant of the covariance matrix.

        Returns:
            ~torch.Tensor: log determinant :math:`\log \lvert \boldsymbol{\Sigma} \rvert`.
        """
        return self._nvariate * self.logvar

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

    def energy(
        self, pred: torch.Tensor, fn: Literal["nll", "kld", "ce"] = "nll", **kwargs: Any
    ) -> torch.Tensor:
        r"""Computes batchwise energy between predictions and the node's activity.

        .. math::
            \begin{aligned}
                \mathcal{E}_\text{NLL} &= \frac{1}{2} \left(
                (\mathbf{z} - \boldsymbol{\mu})
                ((\mathbf{z} - \boldsymbol{\mu}) \sigma^{-1})^\intercal
                + k \log \sigma \right)
                + \frac{k}{2} \log 2 \pi \\
                \mathcal{E}_\text{KL} &= \frac{1}{2} \left(
                (\mathbf{z} - \boldsymbol{\mu})
                ((\mathbf{z} - \boldsymbol{\mu}) \sigma^{-1})^\intercal \right) \\
                \mathcal{E}_\text{CE} &= \frac{1}{2} \left(
                (\mathbf{z} - \boldsymbol{\mu})
                ((\mathbf{z} - \boldsymbol{\mu}) \sigma^{-1})^\intercal
                + k \log \sigma \right)
                + \frac{k}{2} \log 2 \pi + \frac{k}{2}
            \end{aligned}

        For a :math:`k`-variate distribution.

        Args:
            pred (~torch.Tensor): prediction of the node's activity, :math:`\boldsymbol{\mu}`.
            fn (Literal["nll", "kld", "ce"], optional): mode for computing the energy.
                Defaults to "nll".

        Returns:
            ~torch.Tensor: batchwise energy between the activity and the predictions.

        Raises:
            ValueError: invalid ``fn`` specified.

        Info:
            The `fn` parameter controls how energy is computed.

            - Negative Log-Likelihood ("nll"): the node's activity is treated as a point
              estimate of the distribution, and NLL is taken.
            - Kullback–Leibler Divergence ("kld"): the node's activity is treated as the
              as the distribution's mean, and reverse KL-divergence is taken.
            - Cross-Entropy ("ce"): the node's activity is treated as the
              as the distribution's mean, and cross-entropy is taken. This is the
              expected value of NLL.

            Since cross-entropy is evaluated in closed-form for multivariate Gaussians,
            this is equivalent to negative log-likelihood up to a constant.
        """
        # quadratic loss term (common to all methods)
        mu = self.prediction(pred, **kwargs)
        diff, pragma = self.shapeobj.coalesce(self.mean - mu)
        y = diff / self.logvar.exp()
        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()

        # method dependent normalization
        match fn:
            case "nll":
                norm = 0.5 * (self._logdet_cov() + self._nvariate * self._ln2pi)
                E = quad + norm
            case "kld":
                E = quad
            case "ce":
                norm = 0.5 * (self._logdet_cov() + self._nvariate * (1.0 + self._ln2pi))
                E = quad + norm
            case _:
                raise ValueError(f"invalid `fn` of '{fn}' specified")

        # reshape and sum over plates
        E = self.shapeobj.disperse(E, pragma, "plate")
        return E.flatten(1).sum(1)

    def error(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes elementwise error between predictions and the node's activity.

        .. math::
            \boldsymbol{\epsilon} = \frac{\mathbf{z} - \boldsymbol{\mu}}{\sigma}

        Args:
            pred (~torch.Tensor): prediction of the node's activity, :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error between the activity and the predictions.
        """
        mu = self.prediction(pred, **kwargs)
        return (self.mean - mu) / self.logvar.exp()

    def sample(
        self,
        pred: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Samples from the learned conditional distribution.

        Args:
            pred (~torch.Tensor | None, optional): prediction of the node's activity,
                used as the mean of the distribution if provided. If None, the node's
                activity is used for the prediction. Defaults to None.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            ~torch.Tensor: samples from the conditional distribution.
        """
        if pred is None:
            mu = self.mean
        else:
            mu = self.prediction(pred, **kwargs)

        mu, pragma = self.shapeobj.coalesce(mu)

        # white noise
        eps = torch.randn(mu.shape, generator=generator, out=torch.empty_like(mu))

        # color noise and shift
        z = torch.add(mu, eps, alpha=self.logvar.exp().sqrt().item())
        return self.shapeobj.disperse(z, pragma)


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
        mean (~torch.nn.parameter.Parameter): current value of the node :math:`\mathbf{z}`.
        logvar (~torch.nn.parameter.Parameter): log of the distribution variances :math:`\log{\boldsymbol{\sigma}}`.
    """

    logvar: nn.Parameter

    def __init__(
        self, *shape: int | None, variance: float | torch.Tensor = 1.0
    ) -> None:
        AbstractGaussianNode.__init__(self, *shape)

        self.logvar = nn.Parameter(torch.empty([self.size]), True)
        self.covariance = variance

    def _logdet_cov(self) -> torch.Tensor:
        r"""Computes the log of the determinant of the covariance matrix.

        Returns:
            ~torch.Tensor: log determinant :math:`\log \lvert \boldsymbol{\Sigma} \rvert`.
        """
        return self.logvar.sum()

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

        Raises:
            ValueError: ``covariance`` must be a scalar, vector, or square matrix,
                with a corresponding number of elements.
            ValueError: scalar variance must be positive.
            ValueError: all elements of variance vector must be positive.
            ValueError: covariance matrix must be symmetric and positive-definite.

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

    def energy(
        self, pred: torch.Tensor, fn: Literal["nll", "kld", "ce"] = "nll", **kwargs: Any
    ) -> torch.Tensor:
        r"""Computes batchwise energy between predictions and the node's activity.

        .. math::
            \begin{aligned}
                \mathcal{E}_\text{NLL} &= \frac{1}{2} \left(
                (\mathbf{z} - \boldsymbol{\mu})
                ((\mathbf{z} - \boldsymbol{\mu}) \oslash \boldsymbol{\sigma})^\intercal
                + \sum_i \log \boldsymbol{\sigma}_i \right)
                + \frac{k}{2} \log 2 \pi \\
                \mathcal{E}_\text{KL} &= \frac{1}{2} \left(
                (\mathbf{z} - \boldsymbol{\mu})
                ((\mathbf{z} - \boldsymbol{\mu}) \oslash \boldsymbol{\sigma})^\intercal
                \right) \\
                \mathcal{E}_\text{CE} &= \frac{1}{2} \left(
                (\mathbf{z} - \boldsymbol{\mu})
                ((\mathbf{z} - \boldsymbol{\mu}) \oslash \boldsymbol{\sigma})^\intercal
                + \sum_i \log \boldsymbol{\sigma}_i \right)
                + \frac{k}{2} \log 2 \pi + \frac{k}{2}
            \end{aligned}

        For a :math:`k`-variate distribution.

        Args:
            pred (~torch.Tensor): prediction of the node's activity, :math:`\boldsymbol{\mu}`.
            fn (Literal["nll", "kld", "ce"], optional): mode for computing the energy.
                Defaults to "nll".

        Returns:
            ~torch.Tensor: batchwise energy between the activity and the predictions.

        Raises:
            ValueError: invalid ``fn`` specified.

        Info:
            The `fn` parameter controls how energy is computed.

            - Negative Log-Likelihood ("nll"): the node's activity is treated as a point
              estimate of the distribution, and NLL is taken.
            - Kullback–Leibler Divergence ("kld"): the node's activity is treated as the
              as the distribution's mean, and reverse KL-divergence is taken.
            - Cross-Entropy ("ce"): the node's activity is treated as the
              as the distribution's mean, and cross-entropy is taken. This is the
              expected value of NLL.

            Since cross-entropy is evaluated in closed-form for multivariate Gaussians,
            this is equivalent to negative log-likelihood up to a constant.
        """
        # quadratic loss term (common to all methods)
        mu = self.prediction(pred, **kwargs)
        diff, pragma = self.shapeobj.coalesce(self.mean - mu)
        y = diff / self.logvar.exp()
        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()

        # method dependent normalization
        match fn:
            case "nll":
                norm = 0.5 * (self._logdet_cov() + self._nvariate * self._ln2pi)
                E = quad + norm
            case "kld":
                E = quad
            case "ce":
                norm = 0.5 * (self._logdet_cov() + self._nvariate * (1.0 + self._ln2pi))
                E = quad + norm
            case _:
                raise ValueError(f"invalid `fn` of '{fn}' specified")

        # reshape and sum over plates
        E = self.shapeobj.disperse(E, pragma, "plate")
        return E.flatten(1).sum(1)

    def error(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes elementwise error between predictions and the node's activity.

        .. math::
            \boldsymbol{\epsilon} =
            (\mathbf{z} - \boldsymbol{\mu}) \oslash \boldsymbol{\sigma}

        Args:
            pred (~torch.Tensor): prediction of the node's activity, :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error between the activity and the predictions.
        """
        mu = self.prediction(pred, **kwargs)
        diff, pragma = self.shapeobj.coalesce(self.mean - mu)
        err = diff / self.logvar.exp()
        err = self.shapeobj.disperse(err, pragma)
        return err

    def sample(
        self,
        pred: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Samples from the learned conditional distribution.

        Args:
            pred (~torch.Tensor | None, optional): prediction of the node's activity,
                used as the mean of the distribution if provided. If None, the node's
                activity is used for the prediction. Defaults to None.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            ~torch.Tensor: samples from the conditional distribution.
        """
        if pred is None:
            mu = self.mean
        else:
            mu = self.prediction(pred, **kwargs)

        mu, pragma = self.shapeobj.coalesce(mu)

        # white noise
        eps = torch.randn(mu.shape, generator=generator, out=torch.empty_like(mu))

        # color noise and shift
        z = mu + self.logvar.exp().sqrt() * eps
        return self.shapeobj.disperse(z, pragma)


@mparameters("cov_cf_logdiag", "cov_cf_offtril")
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
        variance (float | ~torch.Tensor, optional): initial variance. Defaults to 1.0.

    Attributes:
        mean (~torch.nn.parameter.Parameter): current value of the node :math:`\mathbf{z}`.
        cov_cf_logdiag (~torch.nn.parameter.Parameter): log of the diagonal of the
            Cholesky factor for the distribution covariance.
        cov_cf_offtril (~torch.nn.parameter.Parameter): Cholesky factor for the
            distribution covariances, with the diagonal zeroed.
    """

    cov_cf_logdiag: nn.Parameter
    cov_cf_offtril: nn.Parameter

    def __init__(
        self, *shape: int | None, covariance: float | torch.Tensor = 1.0
    ) -> None:
        AbstractGaussianNode.__init__(self, *shape)

        self.cov_cf_logdiag = nn.Parameter(torch.empty(self.size), True)
        self.cov_cf_offtril = nn.Parameter(torch.empty(self.size, self.size), True)

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

    def energy(
        self, pred: torch.Tensor, fn: Literal["nll", "kld", "ce"] = "nll", **kwargs: Any
    ) -> torch.Tensor:
        r"""Computes batchwise energy between predictions and the node's activity.

        .. math::
            \begin{aligned}
                \mathcal{E}_\text{NLL} &= \frac{1}{2} \left(
                (\mathbf{z} - \boldsymbol{\mu})
                \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal
                + \log \lvert\boldsymbol{\Sigma}\rvert\right) + \frac{k}{2} \log 2 \pi \\
                \mathcal{E}_\text{KL} &= \frac{1}{2} \left(
                (\mathbf{z} - \boldsymbol{\mu})
                \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal \right) \\
                \mathcal{E}_\text{CE} &= \frac{1}{2} \left(
                (\mathbf{z} - \boldsymbol{\mu})
                \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal
                + \log \lvert\boldsymbol{\Sigma}\rvert\right)
                + \frac{k}{2} \log 2 \pi + \frac{k}{2}
            \end{aligned}

        For a :math:`k`-variate distribution.

        Args:
            pred (~torch.Tensor): prediction of the node's activity, :math:`\boldsymbol{\mu}`.
            fn (Literal["nll", "kld", "ce"], optional): mode for computing the energy.
                Defaults to "nll".

        Returns:
            ~torch.Tensor: batchwise energy between the activity and the predictions.

        Raises:
            ValueError: invalid ``fn`` specified.

        Info:
            The `fn` parameter controls how energy is computed.

            - Negative Log-Likelihood ("nll"): the node's activity is treated as a point
              estimate of the distribution, and NLL is taken.
            - Kullback–Leibler Divergence ("kld"): the node's activity is treated as the
              as the distribution's mean, and reverse KL-divergence is taken.
            - Cross-Entropy ("ce"): the node's activity is treated as the
              as the distribution's mean, and cross-entropy is taken. This is the
              expected value of NLL.

            Since cross-entropy is evaluated in closed-form for multivariate Gaussians,
            this is equivalent to negative log-likelihood up to a constant.
        """
        # quadratic loss term (common to all methods)
        mu = self.prediction(pred, **kwargs)
        L = self._cholesky_factor_cov()
        diff, pragma = self.shapeobj.coalesce(self.mean - mu)
        Y = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        quad = 0.5 * (Y * Y).sum(0)

        # method dependent normalization
        match fn:
            case "nll":
                norm = 0.5 * (self._logdet_cov() + self._nvariate * self._ln2pi)
                E = quad + norm
            case "kld":
                E = quad
            case "ce":
                norm = 0.5 * (self._logdet_cov() + self._nvariate * (1.0 + self._ln2pi))
                E = quad + norm
            case _:
                raise ValueError(f"invalid `fn` of '{fn}' specified")

        # reshape and sum over plates
        E = self.shapeobj.disperse(E, pragma, "plate")
        return E.flatten(1).sum(1)

    def error(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes elementwise error between predictions and the node's activity.

        .. math::
            \boldsymbol{\epsilon} =
            \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal

        Args:
            pred (~torch.Tensor): prediction of the node's activity, :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error between the activity and the predictions.
        """
        mu = self.prediction(pred, **kwargs)
        L = self._cholesky_factor_cov()
        diff, pragma = self.shapeobj.coalesce(self.mean - mu)

        y = torch.cholesky_solve(diff.t(), L)
        err = self.shapeobj.disperse(y.t(), pragma)

        return err

    def sample(
        self,
        pred: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Samples from the learned conditional distribution.

        Args:
            pred (~torch.Tensor | None, optional): prediction of the node's activity,
                used as the mean of the distribution if provided. If None, the node's
                activity is used for the prediction. Defaults to None.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            ~torch.Tensor: samples from the conditional distribution.
        """
        if pred is None:
            mu = self.mean
        else:
            mu = self.prediction(pred, **kwargs)

        mu, pragma = self.shapeobj.coalesce(mu)
        L = self._cholesky_factor_cov()

        # white noise
        eps = torch.randn(mu.shape, generator=generator, out=torch.empty_like(mu))

        # color noise and shift
        z = torch.addmm(mu, eps, L.t())
        return self.shapeobj.disperse(z, pragma)
