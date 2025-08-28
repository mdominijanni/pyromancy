from abc import ABC, abstractmethod
from itertools import repeat
import math
import torch
import torch.nn as nn
from typing import Any


class DiffusionSchedule(nn.Module, ABC):

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @property
    @abstractmethod
    def timesteps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor | int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Computes forward diffusion steps.

        .. math::
            \mathbf{x}_t \sim q(\mathbf{x}_t | \mathbf{x}_0)

        Args:
            x0 (torch.Tensor): data prior to any diffusion process, :math:`\mathbf{x}_0`.
            t (torch.Tensor | int): time steps to sample from, :math:`t`.

        Returns:
            torch.Tensor: diffused inputs, :math:`\mathbf{x}_t`.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def reverse_diffusion(
        self,
        xt: torch.Tensor,
        t: torch.Tensor | int,
        pred: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Computes a reverse diffusion step.

        Args:
            xt (torch.Tensor): diffused data at time :math:`t`, :math:`\mathbf{x}_t`.
            t (torch.Tensor | int): time step of the diffusion process, :math:`t`.
            pred (torch.Tensor): prediction of the noise at time :math:`t`,
                :math:`\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)`.

        Returns:
            torch.Tensor: sampled outputs, :math:`\mathbf{x}_{t - 1}`.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | int,
        pred: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Computes either a forward or reverse diffusion step.

        Args:
            x (torch.Tensor): data to perform forward or reverse diffusion on.
            t (torch.Tensor | int): time to diffuse out to or the time the data has
                been diffused to.
            pred (torch.Tensor | None): model's prediction of the noise.

        Returns:
            torch.Tensor: data modified by the forward or reverse diffusion process.

        Raises:
            NotImplementedError: must be implemented by subclasses.

        Note:
            When implementing this method, forward diffusion should be used when
            ``pred`` is not specified, and reverse diffusion when it is.
        """
        raise NotImplementedError


class FixedGaussianSchedule(DiffusionSchedule):

    betas: nn.Buffer
    sqrt_alphas_bar: nn.Buffer
    sqrt_one_minus_alphas_bar: nn.Buffer
    sqrt_betas_tilde: nn.Buffer
    mulinv_sqrt_alphas: nn.Buffer
    mulinv_sqrt_alphas_bar: nn.Buffer
    rev_pred_scale: nn.Buffer
    rev_xt_scale: nn.Buffer
    rev_x0_scale: nn.Buffer

    def __init__(
        self,
        betas: torch.Tensor,
        zero_terminal_snr: bool = False,
    ) -> None:
        if betas.ndim != 1:
            raise ValueError("`betas` must be one-dimensional")
        if betas.max() > 1 or betas.min() < 0:
            raise ValueError("elements of `betas` must be between 0 and 1 (inclusive)")

        DiffusionSchedule.__init__(self)

        with torch.no_grad():
            if zero_terminal_snr:
                betas = self._enforce_zero_terminal_snr(betas)

            alphas = 1 - betas
            sqrt_alphas = alphas.sqrt()
            mulinv_sqrt_alphas = 1.0 / sqrt_alphas

            alphas_bar = torch.cumprod(alphas, dim=0)
            one_minus_alphas_bar = 1.0 - alphas_bar
            sqrt_alphas_bar = alphas_bar.sqrt()
            mulinv_sqrt_alphas_bar = 1.0 / sqrt_alphas_bar
            sqrt_one_minus_alphas_bar = one_minus_alphas_bar.sqrt()

            betas_tilde = torch.zeros_like(betas)
            betas_tilde[1:] = (
                one_minus_alphas_bar[:-1] / one_minus_alphas_bar[1:]
            ) * betas[1:]
            sqrt_betas_tilde = betas_tilde.sqrt()

            rev_pred_scale = torch.zeros_like(betas)
            rev_pred_scale[1:] = betas[1:] / sqrt_one_minus_alphas_bar[1:]

            rev_xt_scale = torch.zeros_like(betas)
            rev_xt_scale[1:] = (
                sqrt_alphas[1:] * one_minus_alphas_bar[:-1]
            ) / one_minus_alphas_bar[1:]

            rev_x0_scale = torch.zeros_like(betas)
            rev_x0_scale[1:] = (
                sqrt_alphas_bar[:-1] * betas[1:]
            ) / one_minus_alphas_bar[1:]

        self.betas = nn.Buffer(betas)
        self.sqrt_alphas_bar = nn.Buffer(sqrt_alphas_bar)
        self.sqrt_one_minus_alphas_bar = nn.Buffer(sqrt_one_minus_alphas_bar)
        self.sqrt_betas_tilde = nn.Buffer(sqrt_betas_tilde)
        self.mulinv_sqrt_alphas = nn.Buffer(mulinv_sqrt_alphas)
        self.mulinv_sqrt_alphas_bar = nn.Buffer(mulinv_sqrt_alphas_bar)
        self.rev_pred_scale = nn.Buffer(rev_pred_scale)
        self.rev_xt_scale = nn.Buffer(rev_xt_scale)
        self.rev_x0_scale = nn.Buffer(rev_x0_scale)

    @property
    def timesteps(self) -> int:
        return self.betas.size(0) - 1

    @staticmethod
    def _enforce_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
        if betas.max() > 1 or betas.min() < 0:
            raise ValueError("elements of `betas` must be between 0 and 1 (inclusive)")

        # convert betas to sqrt_alphas_bar
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        sqrt_alphas_bar = alphas_bar.sqrt()

        # apply zero terminal snr scaling
        sqrt_alphas_bar = (
            (sqrt_alphas_bar - sqrt_alphas_bar[-1])
            * sqrt_alphas_bar[0]
            / (sqrt_alphas_bar[0] - sqrt_alphas_bar[-1])
        )

        # convert sqrt_alphas_bar to betas
        alphas_bar = sqrt_alphas_bar**2
        alphas = torch.cat((alphas_bar[0:1], alphas_bar[1:] / alphas_bar[:-1]), dim=0)
        betas = 1 - alphas

        return betas

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor | int,
        noise: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Computes forward diffusion steps.

        .. math::
            \begin{aligned}
                \mathbf{x}_t &\sim q(\mathbf{x}_t | \mathbf{x}_0) \\
                &\sim \mathcal{N}\big(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,
                \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}\big) \\
                &= \sqrt{\bar{\alpha}_t} \mathbf{x}_0
                + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon} \\ \\
                \boldsymbol{\epsilon} &\sim \mathcal{N}(\mathbf{0}, \mathbf{I})
            \end{aligned}

        Args:
            x0 (torch.Tensor): data prior to any diffusion process, :math:`\mathbf{x}_0`.
            t (torch.Tensor | int): time steps out to diffuse, :math:`t`.
            noise (torch.Tensor): sampling noise, :math:`\boldsymbol{\epsilon}`.

        Returns:
            torch.Tensor: diffused inputs, :math:`\mathbf{x}_t`.

        Important:
            ``noise`` should have the same shape as ``x0``, and it should be on the
            same device and have the same data type as ``x0``.

        Important:
            ``t`` should either be a scalar (an int or a 0d tensor), or a vector
            (a 1d tensor) the same size as the batch dimension of ``x0``.
        """
        if isinstance(t, int):
            t = x0.new_full((1,), t, dtype=torch.int64)

        if t.ndim == 1:
            t = t.view(-1, *repeat(1, x0.ndim - 1))

        return self.sqrt_alphas_bar[t] * x0 + self.sqrt_one_minus_alphas_bar[t] * noise

    def reverse_diffusion(
        self,
        xt: torch.Tensor,
        t: torch.Tensor | int,
        pred: torch.Tensor,
        noise: torch.Tensor,
        clamp_x0: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Computes a reverse diffusion step.

        .. math::
            \begin{aligned}
                \mathbf{x}_{t - 1}
                &\sim q(\mathbf{x}_{t - 1} | \mathbf{x}_{t}, \mathbf{x}_0) \\
                &\sim \mathcal{N}\big(\mathbf{x}_{t - 1};
                \boldsymbol{\tilde{\mu}}_t, \tilde{\beta}_t \mathbf{I}\big) \\
                &= \boldsymbol{\tilde{\mu}}_t + \sqrt{\tilde{\beta}_t}
                \boldsymbol{\epsilon} \\ \\
                \boldsymbol{\tilde{\mu}}_t &=
                \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t - 1})}{1 - \bar{\alpha}_t}\mathbf{x}_t
                + \frac{\sqrt{\bar{\alpha}_{t - 1}} \beta_t}{1 - \bar{\alpha}_t}\mathbf{x}_0 \\
                &\approx \frac{1}{\sqrt{\alpha_t}} \left(\mathbf{x}_{t}
                - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}
                \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) \\ \\
                \tilde{\beta}_t &= \frac{1 - \bar{\alpha}_{t - 1}}{1 - \bar{\alpha}_t}
                \beta_t \\ \\
                \mathbf{x}_0 &\approx \frac{1}{\sqrt{\bar{\alpha}_t}}
                \left(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}
                \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) \\ \\
                \boldsymbol{\epsilon} &\sim \mathcal{N}(\mathbf{0}, \mathbf{I})
            \end{aligned}

        Args:
            xt (torch.Tensor): diffused data at time :math:`t`, :math:`\mathbf{x}_t`.
            t (torch.Tensor | int): time step of the diffusion process, :math:`t`.
            pred (torch.Tensor): prediction of the noise at time :math:`t`,
                :math:`\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)`.
            noise (torch.Tensor): sampling noise, :math:`\boldsymbol{\epsilon}`.
            clamp_x0 (bool, optional): if the predicted value of :math:`\mathbf{x}_0`
                should be clamped to a range of :math:`[-1, 1]`. Defaults to False.

        Returns:
            torch.Tensor: sampled outputs, :math:`\mathbf{x}_{t - 1}`.

        Important:
            ``noise`` should have the same shape as ``xt``, and it should be on the
            same device and have the same data type as ``xt``.

        Important:
            ``t`` should either be a scalar (an int or a 0d tensor), or a vector
            (a 1d tensor) the same size as the batch dimension of ``xt``.
        """
        if isinstance(t, int):
            t = xt.new_full((1,), t, dtype=torch.int64)

        if t.ndim == 1:
            t = t.view(-1, *repeat(1, xt.ndim - 1))

        if clamp_x0:
            x0 = self.mulinv_sqrt_alphas_bar[t] * (
                xt - self.sqrt_one_minus_alphas_bar[t] * pred
            ).clamp(-1.0, 1.0)
            mean = self.rev_xt_scale[t] * xt + self.rev_x0_scale[t] * x0
        else:
            mean = self.mulinv_sqrt_alphas[t] * (xt - self.rev_pred_scale[t] * pred)

        return mean + self.sqrt_betas_tilde[t] * noise

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | int,
        pred: torch.Tensor | None,
        noise: torch.Tensor | torch.Generator | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not isinstance(noise, torch.Tensor):
            noise = torch.randn(*x.shape, generator=noise, out=torch.empty_like(x))

        if pred is None:
            return self.forward_diffusion(x, t, noise, **kwargs)
        else:
            return self.reverse_diffusion(x, t, pred, noise, **kwargs)


class ConstantGaussianSchedule(FixedGaussianSchedule):

    def __init__(
        self,
        timesteps: int,
        beta: float,
        zero_terminal_snr: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.get_default_device()

        betas = torch.full((timesteps,), float(beta), dtype=dtype, device=device)
        FixedGaussianSchedule.__init__(self, betas, zero_terminal_snr)


class LinearGaussianSchedule(FixedGaussianSchedule):

    def __init__(
        self,
        timesteps: int,
        beta_1: float | None = 1e-4,
        beta_T: float = 0.02,
        zero_terminal_snr: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.get_default_device()

        if beta_1 is None:
            betas = torch.linspace(0, beta_T, timesteps + 1, dtype=dtype, device=device)
        else:
            betas = torch.zeros(timesteps + 1, dtype=dtype, device=device)
            betas[1:] = torch.linspace(
                beta_1, beta_T, timesteps, dtype=dtype, device=device
            )

        FixedGaussianSchedule.__init__(self, betas, zero_terminal_snr)


class CosineGaussianSchedule(FixedGaussianSchedule):

    def __init__(
        self,
        timesteps: int,
        offset: float = 0.008,
        exponent: float = 2.0,
        beta_max: float | None = 0.999,
        zero_terminal_snr: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.get_default_device()

        t = torch.arange(0, timesteps + 1, dtype=dtype, device=device)
        f = (
            torch.cos(((t / timesteps + offset) / (1.0 + offset)) * (math.pi / 2))
            ** exponent
        )

        alphas_bar = f / f[0]
        betas = torch.zeros(timesteps + 1, dtype=dtype, device=device)
        betas[1:] = 1.0 - alphas_bar[1:] / alphas_bar[:-1]

        if beta_max is not None:
            betas = torch.clamp_max(betas, beta_max)

        FixedGaussianSchedule.__init__(self, betas, zero_terminal_snr)
