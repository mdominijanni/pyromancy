import einops as ein
import math
import pytest
import torch

from pyromancy.v2nodes import MultivariateGaussianNode


class TestMultivariateGaussianNode:

    @pytest.fixture
    def shape(self) -> tuple[int | None, ...]:
        return (3, None, 4, None, 2)

    @pytest.fixture
    def batch_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (8, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    @pytest.fixture
    def sample_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (10000, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    @pytest.fixture
    def nvariate(self, shape: tuple[int | None, ...]) -> int:
        return math.prod(filter(lambda s: s is not None, shape))  # type: ignore

    @pytest.fixture
    def seed(self) -> int:
        return 42

    def test_covariance_float(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        var = torch.rand((), generator=g).item()

        node.covariance = var
        assert torch.allclose(node.covariance, var * torch.eye(nvariate))

    def test_covariance_0d(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        var = torch.rand((), generator=g)

        node.covariance = var
        assert torch.allclose(node.covariance, var * torch.eye(nvariate))

    def test_covariance_1d(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        var = torch.rand(nvariate, generator=g)

        node.covariance = var
        assert torch.allclose(node.covariance, var.diag())

    def test_covariance_2d(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        node.covariance = cov
        assert torch.allclose(node.covariance, cov)

    @pytest.mark.parametrize(
        "npred",
        (1, 4),
        ids=("npred=single", "npred=multiple"),
    )
    def test_prediction(
        self,
        npred: int,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        pred = [torch.rand(batch_shape, generator=g) for _ in range(npred)]

        mu = node.prediction(*pred)
        assert torch.allclose(mu, torch.stack(pred).mean(0))

    @pytest.mark.parametrize(
        "npred",
        (1, 4),
        ids=("npred=single", "npred=multiple"),
    )
    def test_energy(
        self,
        npred: int,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        zs = [torch.rand(batch_shape, generator=g) for _ in range(npred)]
        mus = [torch.rand(batch_shape, generator=g) for _ in range(npred)]

        node.covariance = cov
        node.initialize(*zs)

        # reference energy
        L = torch.linalg.cholesky(cov)
        z = sum(zs) / len(zs)
        mu_bar = sum(mus) / len(mus)

        m = len(mus)  # number of predictions
        n = nvariate  # number of variables
        p = z.numel() / (z.size(0) * n)  # number of points per variable # type: ignore

        d, pragma = node.shape.coalesce(z - mu_bar)  # type: ignore
        y = torch.cholesky_solve(d.unsqueeze(-1), L).squeeze(-1)

        d = node.shape.disperse(d, pragma).flatten(1)
        y = node.shape.disperse(y, pragma).flatten(1)

        q = 0.5 * m * (d.unsqueeze(-2) @ y.unsqueeze(-1))

        trs = []
        for mu in mus:
            d, pragma = node.shape.coalesce(mu - mu_bar)
            y = torch.cholesky_solve(d.unsqueeze(-1), L).squeeze(-1)

            d = node.shape.disperse(d, pragma).flatten(1)
            y = node.shape.disperse(y, pragma).flatten(1)

            trs.append(d.unsqueeze(-2) @ y.unsqueeze(-1))

        tr = 0.5 * ein.reduce(trs, "d ... -> ...", "sum")

        logdet = 2.0 * L.diag().log().sum()
        norm = 0.5 * m * p * (logdet + n * math.log(2.0 * math.pi))

        E = (q + tr + norm).flatten()

        # compare
        print(torch.abs(node.energy(*mus) - E).max())
        assert torch.allclose(node.energy(*mus), E)

    @pytest.mark.parametrize(
        "npred",
        (1, 4),
        ids=("npred=single", "npred=multiple"),
    )
    def test_error(
        self,
        npred: int,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        zs = [torch.rand(batch_shape, generator=g) for _ in range(npred)]
        mus = [torch.rand(batch_shape, generator=g) for _ in range(npred)]

        node.covariance = cov
        node.initialize(*zs)

        # derived error
        E = node.energy(*mus)

        (grad,) = torch.autograd.grad(
            outputs=E,
            inputs=node.activity(),
            grad_outputs=torch.ones_like(E),
        )

        # compare
        assert torch.allclose(node.error(*mus), grad)

    @pytest.mark.parametrize(
        "npred",
        (1, 4),
        ids=("npred=single", "npred=multiple"),
    )
    def test_sample(
        self,
        npred: int,
        shape: tuple[int | None, ...],
        sample_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        mus = [torch.rand(sample_shape, generator=g) for _ in range(npred)]

        node.covariance = cov

        # reference samples
        g = torch.Generator().manual_seed(seed + 1)
        m = len(mus)  # number of predictions

        mu = ein.reduce(mus, "m ... -> ...", "mean")
        mu, pragma = node.shape.coalesce(mu)

        L = torch.linalg.cholesky(cov)
        eps = torch.randn(mu.shape, generator=g, out=torch.empty_like(mu))
        eps = eps @ (L / math.sqrt(m)).t()

        z = mu + eps
        z = node.shape.disperse(z, pragma)

        # compare
        g = torch.Generator().manual_seed(seed + 1)
        sample = node.sample(*mus, generator=g)
        assert torch.allclose(sample, z)

        # empirical testing
        x, _ = node.shape.coalesce(sample)
        x_mean = x.mean(0, keepdim=True)
        x_cov = x - x_mean
        x_cov = (x_cov.t() @ x_cov) / (x_cov.size(0) - 1)
        x_mean = x_mean.squeeze(0)

        mean_tol = 5.0 / math.sqrt(x_cov.size(1))
        cov_tol_rel = 0.15 / math.sqrt(x_cov.size(0)) + 1e-3
        cov_tol_abs = 5e-5

        print(cov)
        print(x_cov)

        assert torch.allclose(x_mean, torch.zeros_like(x_mean), atol=mean_tol)
        assert torch.allclose(x_cov, cov / m, rtol=cov_tol_rel, atol=cov_tol_abs)
