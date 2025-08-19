import math
import torch
import torch.nn as nn
from typing import Self


class VarianceSchedule(nn.Module):

    mean_scale: nn.Buffer
    var_scale: nn.Buffer

    def __init__(self, mean_scale: torch.Tensor, var_scale: torch.Tensor) -> None:
        if mean_scale.ndim != 1 or var_scale.ndim != 1:
            raise ValueError("`mean_scale` and `var_scale` must be 1-dimensional")
        if mean_scale.shape != var_scale.shape:
            raise ValueError("`mean_scale` and `var_scale` must have the same shape")

        nn.Module.__init__(self)

        self.mean_scale = nn.Buffer(mean_scale)
        self.var_scale = nn.Buffer(var_scale)

    @classmethod
    @torch.no_grad()
    def from_betas(
        cls, betas: torch.Tensor, zero_terminal_snr: bool = False, prepend: bool = True
    ) -> Self:
        if betas.ndim != 1:
            raise ValueError("`betas` must be 1-dimensional")
        if betas.max() >= 1:
            raise RuntimeError("values of `betas` must be less than 1")
        if betas.min() <= 0:
            raise RuntimeError("values of `betas` must be greater than 0")

        alphas_bar = torch.cumprod(1.0 - betas, dim=0)

        if prepend:
            alphas_bar = torch.cat((alphas_bar.new_ones(1), alphas_bar), dim=0)

        if zero_terminal_snr:
            return cls.from_alphas_bar(
                torch.sqrt(alphas_bar), zero_terminal_snr=True, prepend=False
            )
        else:
            return cls(
                torch.sqrt(alphas_bar),
                torch.sqrt(1.0 - alphas_bar),
            )

    @classmethod
    @torch.no_grad()
    def from_alphas(
        cls, alphas: torch.Tensor, zero_terminal_snr: bool = False, prepend: bool = True
    ) -> Self:
        if alphas.ndim != 1:
            raise ValueError("`alphas` must be 1-dimensional")
        if alphas.max() >= 1:
            raise RuntimeError("values of `alphas` must be less than 1")
        if alphas.min() <= 0:
            raise RuntimeError("values of `alphas` must be greater than 0")

        alphas_bar = torch.cumprod(alphas, dim=0)

        if prepend:
            alphas_bar = torch.cat((alphas_bar.new_ones(1), alphas_bar), dim=0)

        if zero_terminal_snr:
            return cls.from_alphas_bar(
                torch.sqrt(alphas_bar), zero_terminal_snr=True, prepend=False
            )
        else:
            return cls(
                torch.sqrt(alphas_bar),
                torch.sqrt(1.0 - alphas_bar),
            )

    @classmethod
    @torch.no_grad()
    def from_alphas_bar(
        cls,
        alphas_bar: torch.Tensor,
        zero_terminal_snr: bool = False,
        prepend: bool = True,
    ) -> Self:
        if alphas_bar.ndim != 1:
            raise ValueError("`alphas_bar` must be 1-dimensional")
        if alphas_bar.max() >= 1:
            raise RuntimeError("values of `alphas_bar` must be less than or equal to 1")
        if alphas_bar.min() <= 0:
            raise RuntimeError(
                "values of `alphas_bar` must be greater than or equal to 0"
            )

        if prepend:
            alphas_bar = torch.cat((alphas_bar.new_ones(1), alphas_bar), dim=0)

        if zero_terminal_snr:
            return cls.from_alphas_bar(
                torch.sqrt(alphas_bar), zero_terminal_snr=True, prepend=False
            )
        else:
            return cls(
                torch.sqrt(alphas_bar),
                torch.sqrt(1.0 - alphas_bar),
            )

    @classmethod
    @torch.no_grad()
    def from_sqrt_alphas_bar(
        cls,
        sqrt_alphas_bar: torch.Tensor,
        zero_terminal_snr: bool = False,
        prepend: bool = True,
    ) -> Self:
        if sqrt_alphas_bar.ndim != 1:
            raise ValueError("`sqrt_alphas_bar` must be 1-dimensional")
        if sqrt_alphas_bar.max() >= 1:
            raise RuntimeError(
                "values of `sqrt_alphas_bar` must be less than or equal to 1"
            )
        if sqrt_alphas_bar.min() <= 0:
            raise RuntimeError(
                "values of `sqrt_alphas_bar` must be greater than or equal to 0"
            )

        if prepend:
            sqrt_alphas_bar = torch.cat(
                (sqrt_alphas_bar.new_ones(1), sqrt_alphas_bar), dim=0
            )

        if zero_terminal_snr:
            sqrt_alphas_bar = (
                (sqrt_alphas_bar - sqrt_alphas_bar[-1])
                * sqrt_alphas_bar[0]
                / (sqrt_alphas_bar[0] - sqrt_alphas_bar[-1])
            )

        return cls(sqrt_alphas_bar, torch.sqrt(1.0 - sqrt_alphas_bar.pow(2.0)))

    @property
    def timesteps(self) -> int:
        return self.mean_scale.size(0) - 1

    def forward(
        self,
        x0: torch.Tensor,
        step: int,
        noise_or_gen: torch.Tensor | torch.Generator | None,
    ) -> torch.Tensor:
        if not isinstance(noise_or_gen, torch.Tensor):
            noise_or_gen = torch.randn(
                *x0.shape, generator=noise_or_gen, out=torch.empty_like(x0)
            )
        return self.mean_scale[step] * x0 + self.var_scale[step] * noise_or_gen


@torch.no_grad()
def constant_schedule(
    timesteps: int,
    beta: float,
    zero_terminal_snr: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> VarianceSchedule:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()

    betas = torch.full((timesteps,), float(beta), dtype=dtype, device=device)
    return VarianceSchedule.from_betas(betas, zero_terminal_snr, True)


@torch.no_grad()
def linear_schedule(
    timesteps: int,
    beta_1: float | None = 1e-4,
    beta_T: float = 0.02,
    zero_terminal_snr: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> VarianceSchedule:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()

    if beta_1 is None:
        betas = torch.linspace(0, beta_T, timesteps + 1, dtype=dtype, device=device)
        return VarianceSchedule.from_betas(betas, zero_terminal_snr, False)
    else:
        betas = torch.linspace(beta_1, beta_T, timesteps, dtype=dtype, device=device)
        return VarianceSchedule.from_betas(betas, zero_terminal_snr, True)


@torch.no_grad()
def cosine_schedule(
    timesteps: int,
    offset: float = 0.008,
    exponent: float = 2.0,
    beta_max: float | None = 0.999,
    zero_terminal_snr: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> VarianceSchedule:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()

    t = torch.arange(0, timesteps + 1, dtype=dtype, device=device)
    f = torch.cos(((t / timesteps + offset) / (1.0 + offset)) * (math.pi / 2)).pow(
        exponent
    )

    alphas_bar = f / f[0]

    if beta_max is None:
        return VarianceSchedule.from_alphas_bar(alphas_bar, zero_terminal_snr, False)
    else:
        betas = torch.clamp_max(1.0 - alphas_bar[1:] / alphas_bar[:-1], beta_max)
        return VarianceSchedule.from_betas(betas, zero_terminal_snr, True)
