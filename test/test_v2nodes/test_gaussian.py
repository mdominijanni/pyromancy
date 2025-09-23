from typing import Literal
import math
import pytest
import torch

from pyromancy import get_named_estep_params, get_named_mstep_params
from pyromancy.v2nodes import (
    StandardGaussianNode,
    IsotropicGaussianNode,
    FactorizedGaussianNode,
    MultivariateGaussianNode,
)


class TestStandardGaussianNode:

    @pytest.fixture
    def shape(self) -> tuple[int | None, ...]:
        return (3, None, 4, None, 2)

    @pytest.fixture
    def batch_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (8, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    @pytest.fixture
    def fullshape(self, nvariate: int) -> tuple[int | None, ...]:
        return (nvariate,)

    @pytest.fixture
    def batch_fullshape(self, fullshape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (
            8,
            *(s if s is not None else d + 1 for d, s in enumerate(fullshape)),
        )

    @pytest.fixture
    def sample_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (10000, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    @pytest.fixture
    def nvariate(self, shape: tuple[int | None, ...]) -> int:
        return math.prod(filter(lambda s: s is not None, shape))  # type: ignore

    @pytest.fixture
    def seed(self) -> int:
        return 42

    def test_covariance(self, shape: tuple[int | None, ...], nvariate: int) -> None:
        node = IsotropicGaussianNode(*shape)
        assert torch.allclose(node.covariance, torch.eye(nvariate))

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_energy(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = StandardGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.eye(nvariate)

        z = torch.rand(batch_shape, generator=g)
        mu = torch.rand(batch_shape, generator=g)

        node.initialize(z)

        # reference energy
        L = torch.linalg.cholesky(cov)
        diff, pragma = node.shapeobj.coalesce(z - mu)

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True).t()

        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()
        quad = node.shapeobj.disperse(quad, pragma, "plate")

        match energyfn:
            case "nll":
                norm = 0.5 * (L.diag().log().sum().mul(2) + L.size(0) * node._ln2pi)
            case "kld":
                norm = 0.0
            case "ce":
                norm = 0.5 * (
                    L.diag().log().sum().mul(2) + L.size(0) * (1.0 + node._ln2pi)
                )

        E = (quad + norm).flatten(1).sum(1)

        # compare
        print(torch.abs(node.energy(mu, energyfn) - E).max())
        assert torch.allclose(node.energy(mu, energyfn), E)

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_energy_fullshape(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        fullshape: tuple[int | None, ...],
        batch_fullshape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = StandardGaussianNode(*fullshape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.eye(nvariate)

        z = torch.rand(batch_fullshape, generator=g)
        mu = torch.rand(batch_fullshape, generator=g)

        node.initialize(z)

        # reference energy
        L = torch.linalg.cholesky(cov)
        diff, pragma = node.shapeobj.coalesce(z - mu)

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True).t()

        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()
        quad = node.shapeobj.disperse(quad, pragma, "plate")

        match energyfn:
            case "nll":
                norm = 0.5 * (L.diag().log().sum().mul(2) + L.size(0) * node._ln2pi)
            case "kld":
                norm = 0.0
            case "ce":
                norm = 0.5 * (
                    L.diag().log().sum().mul(2) + L.size(0) * (1.0 + node._ln2pi)
                )

        E = (quad + norm).flatten(1).sum(1)

        # compare
        print(torch.abs(node.energy(mu, energyfn) - E).max())
        assert torch.allclose(node.energy(mu, energyfn), E)

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_error(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = StandardGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        z = torch.rand(batch_shape, generator=g)
        mu = torch.rand(batch_shape, generator=g)

        node.initialize(z)

        # derived error
        E = node.energy(mu, energyfn)

        (grad,) = torch.autograd.grad(
            outputs=E,
            inputs=node.activity(),
            grad_outputs=torch.ones_like(E),
        )

        # compare
        assert torch.allclose(node.error(mu), grad)

    @pytest.mark.parametrize(
        "usepred",
        (True, False),
        ids=("usepred=True", "usepred=False"),
    )
    def test_sample(
        self,
        usepred: bool,
        shape: tuple[int | None, ...],
        sample_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = StandardGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.eye(nvariate)

        mean = torch.rand(sample_shape, generator=g)

        # reference samples
        g = torch.Generator().manual_seed(seed + 1)

        mu, pragma = node.shapeobj.coalesce(mean)

        L = torch.linalg.cholesky(cov)
        eps = torch.randn(mu.shape, generator=g, out=torch.empty_like(mu))
        eps = eps @ L.t()

        z = mu + eps
        z = node.shapeobj.disperse(z, pragma)

        # compare
        g = torch.Generator().manual_seed(seed + 1)

        if usepred:
            sample = node.sample(mean, generator=g)
        else:
            node.initialize(mean, sample=False)
            sample = node.sample(generator=g)

        assert torch.allclose(sample, z, atol=1e-7)

    def test_initialize_deterministic(
        self,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        mu = torch.rand(batch_shape, generator=g)
        z = node.initialize(mu)

        assert torch.all(z == mu)
        assert torch.all(node.activity() == mu)

    def test_initialize_sampled(
        self,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        mu = torch.rand(batch_shape, generator=g)

        g = torch.Generator().manual_seed(seed + 1)
        z = node.initialize(mu, sample=True, generator=g)

        g = torch.Generator().manual_seed(seed + 1)
        zsample = node.sample(mu, generator=g)

        assert torch.all(z == zsample)
        assert torch.all(node.activity() == zsample)

    def test_estep_params(self, shape: tuple[int | None, ...]) -> None:
        node = StandardGaussianNode(*shape)
        estep = dict(get_named_estep_params(node))
        sol = {"mean": node.mean}
        assert estep == sol

    def test_mstep_params(self, shape: tuple[int | None, ...]) -> None:
        node = StandardGaussianNode(*shape)
        mstep = dict(get_named_mstep_params(node))
        sol = dict()
        assert mstep == sol


class TestIsotropicGaussianNode:

    @pytest.fixture
    def shape(self) -> tuple[int | None, ...]:
        return (3, None, 4, None, 2)

    @pytest.fixture
    def batch_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (8, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    @pytest.fixture
    def fullshape(self, nvariate: int) -> tuple[int | None, ...]:
        return (nvariate,)

    @pytest.fixture
    def batch_fullshape(self, fullshape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (
            8,
            *(s if s is not None else d + 1 for d, s in enumerate(fullshape)),
        )

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
        node = IsotropicGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        var = torch.rand((), generator=g).item()

        node.covariance = var
        assert torch.allclose(node.covariance, var * torch.eye(nvariate))

    def test_covariance_0d(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = IsotropicGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        var = torch.rand((), generator=g)

        node.covariance = var
        assert torch.allclose(node.covariance, var * torch.eye(nvariate))

    def test_covariance_1d(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = IsotropicGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        var = torch.rand(nvariate, generator=g)

        node.covariance = var
        assert torch.allclose(node.covariance, torch.eye(nvariate) * var.mean())

    def test_covariance_2d(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = IsotropicGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        node.covariance = cov
        assert torch.allclose(node.covariance, torch.eye(nvariate) * cov.diag().mean())

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_energy(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = IsotropicGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.eye(nvariate) * (torch.rand(1, generator=g) + 0.5)

        z = torch.rand(batch_shape, generator=g)
        mu = torch.rand(batch_shape, generator=g)

        node.covariance = cov
        node.initialize(z)

        # reference energy
        L = torch.linalg.cholesky(cov)
        diff, pragma = node.shapeobj.coalesce(z - mu)

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True).t()

        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()
        quad = node.shapeobj.disperse(quad, pragma, "plate")

        match energyfn:
            case "nll":
                norm = 0.5 * (L.diag().log().sum().mul(2) + L.size(0) * node._ln2pi)
            case "kld":
                norm = 0.0
            case "ce":
                norm = 0.5 * (
                    L.diag().log().sum().mul(2) + L.size(0) * (1.0 + node._ln2pi)
                )

        E = (quad + norm).flatten(1).sum(1)

        # compare
        print(torch.abs(node.energy(mu, energyfn) - E).max())
        assert torch.allclose(node.energy(mu, energyfn), E)

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_energy_fullshape(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        fullshape: tuple[int | None, ...],
        batch_fullshape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = IsotropicGaussianNode(*fullshape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.eye(nvariate) * (torch.rand(1, generator=g) + 0.5)

        z = torch.rand(batch_fullshape, generator=g)
        mu = torch.rand(batch_fullshape, generator=g)

        node.covariance = cov
        node.initialize(z)

        # reference energy
        L = torch.linalg.cholesky(cov)
        diff, pragma = node.shapeobj.coalesce(z - mu)

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True).t()

        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()
        quad = node.shapeobj.disperse(quad, pragma, "plate")

        match energyfn:
            case "nll":
                norm = 0.5 * (L.diag().log().sum().mul(2) + L.size(0) * node._ln2pi)
            case "kld":
                norm = 0.0
            case "ce":
                norm = 0.5 * (
                    L.diag().log().sum().mul(2) + L.size(0) * (1.0 + node._ln2pi)
                )

        E = (quad + norm).flatten(1).sum(1)

        # compare
        print(torch.abs(node.energy(mu, energyfn) - E).max())
        assert torch.allclose(node.energy(mu, energyfn), E)

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_error(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = IsotropicGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.eye(nvariate) * (torch.rand(1, generator=g) + 0.5)

        z = torch.rand(batch_shape, generator=g)
        mu = torch.rand(batch_shape, generator=g)

        node.covariance = cov
        node.initialize(z)

        # derived error
        E = node.energy(mu, energyfn)

        (grad,) = torch.autograd.grad(
            outputs=E,
            inputs=node.activity(),
            grad_outputs=torch.ones_like(E),
        )

        # compare
        assert torch.allclose(node.error(mu), grad)

    @pytest.mark.parametrize(
        "usepred",
        (True, False),
        ids=("usepred=True", "usepred=False"),
    )
    def test_sample(
        self,
        usepred: bool,
        shape: tuple[int | None, ...],
        sample_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = IsotropicGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.eye(nvariate) * (torch.rand(1, generator=g) + 0.5)

        mean = torch.rand(sample_shape, generator=g)

        node.covariance = cov

        # reference samples
        g = torch.Generator().manual_seed(seed + 1)

        mu, pragma = node.shapeobj.coalesce(mean)

        L = torch.linalg.cholesky(cov)
        eps = torch.randn(mu.shape, generator=g, out=torch.empty_like(mu))
        eps = eps @ L.t()

        z = mu + eps
        z = node.shapeobj.disperse(z, pragma)

        # compare
        g = torch.Generator().manual_seed(seed + 1)

        if usepred:
            sample = node.sample(mean, generator=g)
        else:
            node.initialize(mean, sample=False)
            sample = node.sample(generator=g)

        assert torch.allclose(sample, z, atol=1e-7)

    def test_initialize_deterministic(
        self,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        node.covariance = cov

        mu = torch.rand(batch_shape, generator=g)
        z = node.initialize(mu)

        assert torch.all(z == mu)
        assert torch.all(node.activity() == mu)

    def test_initialize_sampled(
        self,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        node.covariance = cov

        mu = torch.rand(batch_shape, generator=g)

        g = torch.Generator().manual_seed(seed + 1)
        z = node.initialize(mu, sample=True, generator=g)

        g = torch.Generator().manual_seed(seed + 1)
        zsample = node.sample(mu, generator=g)

        assert torch.all(z == zsample)
        assert torch.all(node.activity() == zsample)

    def test_estep_params(self, shape: tuple[int | None, ...]) -> None:
        node = IsotropicGaussianNode(*shape)
        estep = dict(get_named_estep_params(node))
        sol = {"mean": node.mean}
        assert estep == sol

    def test_mstep_params(self, shape: tuple[int | None, ...]) -> None:
        node = IsotropicGaussianNode(*shape)
        mstep = dict(get_named_mstep_params(node))
        sol = {"logvar": node.logvar}
        assert mstep == sol


class TestFactorizedGaussianNode:

    @pytest.fixture
    def shape(self) -> tuple[int | None, ...]:
        return (3, None, 4, None, 2)

    @pytest.fixture
    def batch_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (8, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    @pytest.fixture
    def fullshape(self, nvariate: int) -> tuple[int | None, ...]:
        return (nvariate,)

    @pytest.fixture
    def batch_fullshape(self, fullshape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (
            8,
            *(s if s is not None else d + 1 for d, s in enumerate(fullshape)),
        )

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
        node = FactorizedGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        var = torch.rand((), generator=g).item()

        node.covariance = var
        assert torch.allclose(node.covariance, var * torch.eye(nvariate))

    def test_covariance_0d(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = FactorizedGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        var = torch.rand((), generator=g)

        node.covariance = var
        assert torch.allclose(node.covariance, var * torch.eye(nvariate))

    def test_covariance_1d(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = FactorizedGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        var = torch.rand(nvariate, generator=g)

        node.covariance = var
        assert torch.allclose(node.covariance, var.diag())

    def test_covariance_2d(
        self, shape: tuple[int | None, ...], nvariate: int, seed: int
    ) -> None:
        node = FactorizedGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        node.covariance = cov
        assert torch.allclose(node.covariance, cov.diag().diag())

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_energy(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = FactorizedGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = (torch.rand(nvariate, generator=g) + 0.5).diag()

        z = torch.rand(batch_shape, generator=g)
        mu = torch.rand(batch_shape, generator=g)

        node.covariance = cov
        node.initialize(z)

        # reference energy
        L = torch.linalg.cholesky(cov)
        diff, pragma = node.shapeobj.coalesce(z - mu)

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True).t()

        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()
        quad = node.shapeobj.disperse(quad, pragma, "plate")

        match energyfn:
            case "nll":
                norm = 0.5 * (L.diag().log().sum().mul(2) + L.size(0) * node._ln2pi)
            case "kld":
                norm = 0.0
            case "ce":
                norm = 0.5 * (
                    L.diag().log().sum().mul(2) + L.size(0) * (1.0 + node._ln2pi)
                )

        E = (quad + norm).flatten(1).sum(1)

        # compare
        print(torch.abs(node.energy(mu, energyfn) - E).max())
        assert torch.allclose(node.energy(mu, energyfn), E)

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_energy_fullshape(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        fullshape: tuple[int | None, ...],
        batch_fullshape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = FactorizedGaussianNode(*fullshape)
        g = torch.Generator().manual_seed(seed)

        cov = (torch.rand(nvariate, generator=g) + 0.5).diag()

        z = torch.rand(batch_fullshape, generator=g)
        mu = torch.rand(batch_fullshape, generator=g)

        node.covariance = cov
        node.initialize(z)

        # reference energy
        L = torch.linalg.cholesky(cov)
        diff, pragma = node.shapeobj.coalesce(z - mu)

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True).t()

        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()
        quad = node.shapeobj.disperse(quad, pragma, "plate")

        match energyfn:
            case "nll":
                norm = 0.5 * (L.diag().log().sum().mul(2) + L.size(0) * node._ln2pi)
            case "kld":
                norm = 0.0
            case "ce":
                norm = 0.5 * (
                    L.diag().log().sum().mul(2) + L.size(0) * (1.0 + node._ln2pi)
                )

        E = (quad + norm).flatten(1).sum(1)

        # compare
        print(torch.abs(node.energy(mu, energyfn) - E).max())
        assert torch.allclose(node.energy(mu, energyfn), E)

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_error(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = FactorizedGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = (torch.rand(nvariate, generator=g) + 0.5).diag()

        z = torch.rand(batch_shape, generator=g)
        mu = torch.rand(batch_shape, generator=g)

        node.covariance = cov
        node.initialize(z)

        # derived error
        E = node.energy(mu, energyfn)

        (grad,) = torch.autograd.grad(
            outputs=E,
            inputs=node.activity(),
            grad_outputs=torch.ones_like(E),
        )

        # compare
        assert torch.allclose(node.error(mu), grad)

    @pytest.mark.parametrize(
        "usepred",
        (True, False),
        ids=("usepred=True", "usepred=False"),
    )
    def test_sample(
        self,
        usepred: bool,
        shape: tuple[int | None, ...],
        sample_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = FactorizedGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = (torch.rand(nvariate, generator=g) + 0.5).diag()

        mean = torch.rand(sample_shape, generator=g)

        node.covariance = cov

        # reference samples
        g = torch.Generator().manual_seed(seed + 1)

        mu, pragma = node.shapeobj.coalesce(mean)

        L = torch.linalg.cholesky(cov)
        eps = torch.randn(mu.shape, generator=g, out=torch.empty_like(mu))
        eps = eps @ L.t()

        z = mu + eps
        z = node.shapeobj.disperse(z, pragma)

        # compare
        g = torch.Generator().manual_seed(seed + 1)

        if usepred:
            sample = node.sample(mean, generator=g)
        else:
            node.initialize(mean, sample=False)
            sample = node.sample(generator=g)

        assert torch.allclose(sample, z)

    def test_initialize_deterministic(
        self,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        node.covariance = cov

        mu = torch.rand(batch_shape, generator=g)
        z = node.initialize(mu)

        assert torch.all(z == mu)
        assert torch.all(node.activity() == mu)

    def test_initialize_sampled(
        self,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        node.covariance = cov

        mu = torch.rand(batch_shape, generator=g)

        g = torch.Generator().manual_seed(seed + 1)
        z = node.initialize(mu, sample=True, generator=g)

        g = torch.Generator().manual_seed(seed + 1)
        zsample = node.sample(mu, generator=g)

        assert torch.all(z == zsample)
        assert torch.all(node.activity() == zsample)

    def test_estep_params(self, shape: tuple[int | None, ...]) -> None:
        node = FactorizedGaussianNode(*shape)
        estep = dict(get_named_estep_params(node))
        sol = {"mean": node.mean}
        assert estep == sol

    def test_mstep_params(self, shape: tuple[int | None, ...]) -> None:
        node = FactorizedGaussianNode(*shape)
        mstep = dict(get_named_mstep_params(node))
        sol = {"logvar": node.logvar}
        assert mstep == sol


class TestMultivariateGaussianNode:

    @pytest.fixture
    def shape(self) -> tuple[int | None, ...]:
        return (3, None, 4, None, 2)

    @pytest.fixture
    def batch_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (8, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    @pytest.fixture
    def fullshape(self, nvariate: int) -> tuple[int | None, ...]:
        return (nvariate,)

    @pytest.fixture
    def batch_fullshape(self, fullshape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (
            8,
            *(s if s is not None else d + 1 for d, s in enumerate(fullshape)),
        )

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
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_energy(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        z = torch.rand(batch_shape, generator=g)
        mu = torch.rand(batch_shape, generator=g)

        node.covariance = cov
        node.initialize(z)

        # reference energy
        L = torch.linalg.cholesky(cov)
        diff, pragma = node.shapeobj.coalesce(z - mu)

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True).t()

        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()
        quad = node.shapeobj.disperse(quad, pragma, "plate")

        match energyfn:
            case "nll":
                norm = 0.5 * (L.diag().log().sum().mul(2) + L.size(0) * node._ln2pi)
            case "kld":
                norm = 0.0
            case "ce":
                norm = 0.5 * (
                    L.diag().log().sum().mul(2) + L.size(0) * (1.0 + node._ln2pi)
                )

        E = (quad + norm).flatten(1).sum(1)

        # compare
        print(torch.abs(node.energy(mu, energyfn) - E).max())
        assert torch.allclose(node.energy(mu, energyfn), E)

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_energy_fullshape(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        fullshape: tuple[int | None, ...],
        batch_fullshape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*fullshape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        z = torch.rand(batch_fullshape, generator=g)
        mu = torch.rand(batch_fullshape, generator=g)

        node.covariance = cov
        node.initialize(z)

        # reference energy
        L = torch.linalg.cholesky(cov)
        diff, pragma = node.shapeobj.coalesce(z - mu)

        u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
        y = torch.linalg.solve_triangular(L.t(), u, upper=True).t()

        quad = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2)).flatten()
        quad = node.shapeobj.disperse(quad, pragma, "plate")

        match energyfn:
            case "nll":
                norm = 0.5 * (L.diag().log().sum().mul(2) + L.size(0) * node._ln2pi)
            case "kld":
                norm = 0.0
            case "ce":
                norm = 0.5 * (
                    L.diag().log().sum().mul(2) + L.size(0) * (1.0 + node._ln2pi)
                )

        E = (quad + norm).flatten(1).sum(1)

        # compare
        print(torch.abs(node.energy(mu, energyfn) - E).max())
        assert torch.allclose(node.energy(mu, energyfn), E)

    @pytest.mark.parametrize(
        "energyfn",
        ("nll", "kld", "ce"),
        ids=("energyfn=NLL", "energyfn=KLD", "energyfn=CE"),
    )
    def test_error(
        self,
        energyfn: Literal["nll", "kld", "ce"],
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        z = torch.rand(batch_shape, generator=g)
        mu = torch.rand(batch_shape, generator=g)

        node.covariance = cov
        node.initialize(z)

        # derived error
        E = node.energy(mu, energyfn)

        (grad,) = torch.autograd.grad(
            outputs=E,
            inputs=node.activity(),
            grad_outputs=torch.ones_like(E),
        )

        # compare
        assert torch.allclose(node.error(mu), grad)

    @pytest.mark.parametrize(
        "usepred",
        (True, False),
        ids=("usepred=True", "usepred=False"),
    )
    def test_sample(
        self,
        usepred: bool,
        shape: tuple[int | None, ...],
        sample_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        mean = torch.rand(sample_shape, generator=g)

        node.covariance = cov

        # reference samples
        g = torch.Generator().manual_seed(seed + 1)

        mu, pragma = node.shapeobj.coalesce(mean)

        L = torch.linalg.cholesky(cov)
        eps = torch.randn(mu.shape, generator=g, out=torch.empty_like(mu))
        eps = eps @ L.t()

        z = mu + eps
        z = node.shapeobj.disperse(z, pragma)

        # compare
        g = torch.Generator().manual_seed(seed + 1)

        if usepred:
            sample = node.sample(mean, generator=g)
        else:
            node.initialize(mean, sample=False)
            sample = node.sample(generator=g)

        assert torch.allclose(sample, z)

        # empirical testing
        x, _ = node.shapeobj.coalesce(sample)
        x_mean = x.mean(0, keepdim=True)
        x_cov = x - x_mean
        x_cov = (x_cov.t() @ x_cov) / (x_cov.size(0) - 1)
        x_mean = x_mean.squeeze(0)

        mean_tol = 5.0 / math.sqrt(x_cov.size(1))
        cov_tol_rel = 0.15 / math.sqrt(x_cov.size(0)) + 1e-3
        cov_tol_abs = 5e-5

        assert torch.allclose(x_mean, torch.zeros_like(x_mean), atol=mean_tol)
        assert torch.allclose(x_cov, cov, rtol=cov_tol_rel, atol=cov_tol_abs)

    def test_initialize_deterministic(
        self,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        node.covariance = cov

        mu = torch.rand(batch_shape, generator=g)
        z = node.initialize(mu)

        assert torch.all(z == mu)
        assert torch.all(node.activity() == mu)

    def test_initialize_sampled(
        self,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = MultivariateGaussianNode(*shape)
        g = torch.Generator().manual_seed(seed)

        cov = torch.rand(nvariate, nvariate, generator=g)
        cov = cov @ cov.t()

        node.covariance = cov

        mu = torch.rand(batch_shape, generator=g)

        g = torch.Generator().manual_seed(seed + 1)
        z = node.initialize(mu, sample=True, generator=g)

        g = torch.Generator().manual_seed(seed + 1)
        zsample = node.sample(mu, generator=g)

        assert torch.all(z == zsample)
        assert torch.all(node.activity() == zsample)

    def test_estep_params(self, shape: tuple[int | None, ...]) -> None:
        node = MultivariateGaussianNode(*shape)
        estep = dict(get_named_estep_params(node))
        sol = {"mean": node.mean}
        assert estep == sol

    def test_mstep_params(self, shape: tuple[int | None, ...]) -> None:
        node = MultivariateGaussianNode(*shape)
        mstep = dict(get_named_mstep_params(node))
        sol = {
            "cov_cf_logdiag": node.cov_cf_logdiag,
            "cov_cf_offtril": node.cov_cf_offtril,
        }
        assert mstep == sol
