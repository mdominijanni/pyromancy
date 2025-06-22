import math
import random

import pytest
import torch
import torch.nn as nn

from pyromancy import get_named_estep_params, get_named_mstep_params
from pyromancy.nodes import (
    StandardGaussianNode,
    IsotropicGaussianNode,
    FactorizedGaussianNode,
    MultivariateGaussianNode,
)

from ..common import randshape


class TestStandardGaussianNode:

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shapeobj(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = StandardGaussianNode(*shape)

        assert node.shapeobj == (None, *shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = StandardGaussianNode(*shape)

        assert node.shape == shape

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_bshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = StandardGaussianNode(*shape)

        assert node.bshape == tuple(d if d is not None else 1 for d in shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_size(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = StandardGaussianNode(*shape)

        assert node.size == math.prod(filter(lambda d: d is not None, shape))  # type: ignore

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_init_reset(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = StandardGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        assert node.value.shape == (0,)
        initres = node.init(data)
        assert initres.shape == data.shape
        assert torch.allclose(initres, data)
        node.reset()
        assert node.value.shape == (0,)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some"),
        ids=("virtual_dims='none'", "virtual_dims='some'"),
    )
    def test_init_badshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = (3, 2, 5)
            case "some":
                shape = (3, None, 5)
            case _:
                raise AssertionError

        node = StandardGaussianNode(*shape)
        data = torch.rand(
            2, *(d if d is not None else random.randint(1, 4) for d in shape), 4
        )

        with pytest.raises(ValueError) as excinfo:
            _ = node.init(data)
        assert "shape of `value`" in str(excinfo.value)
        assert "is incompatible with node shape" in str(excinfo.value)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_error(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = StandardGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        pred = torch.rand_like(data)

        with torch.no_grad():
            cov = torch.eye(node.size)
            diff, pragma = node.shapeobj.coalesce(data - pred)
            sol = node.shapeobj.disperse(torch.linalg.solve(cov, diff.t()).t(), pragma)
            res = node.error(pred)

        assert res.shape == data.shape
        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_energy(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = StandardGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        pred = torch.rand_like(data)

        with torch.no_grad():
            cov = torch.eye(node.size)
            diff, pragma = node.shapeobj.coalesce(data - pred)
            y = torch.linalg.solve(cov, diff.t()).t()

            diff = node.shapeobj.disperse(diff, pragma).flatten(1)
            y = node.shapeobj.disperse(y, pragma).flatten(1)
            logdet = torch.zeros(())

            sol = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()
            res = node.energy(pred)

        assert res.shape[:1] == data.shape[:1]
        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    @pytest.mark.parametrize(
        "training",
        (True, False),
        ids=("training=True", "training='False"),
    )
    def test_forward(self, virtual_dims, training):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = StandardGaussianNode(*shape).train(training)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        with torch.no_grad():
            res = node(data)

        assert res.shape == data.shape
        assert torch.allclose(res, data)

        if training:
            assert node.value.shape == data.shape
            assert torch.allclose(node.value, res)

    def test_estep_params(self):
        m = nn.ModuleList((StandardGaussianNode(4, None, 3, 3),))
        assert dict(get_named_estep_params(m)) == {"0.value": m[0].value}

    def test_mstep_params(self):
        m = nn.ModuleList((StandardGaussianNode(4, None, 3, 3),))
        assert dict(get_named_mstep_params(m)) == {}


class TestIsotropicGaussianNode:

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shapeobj(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape)

        assert node.shapeobj == (None, *shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape)

        assert node.shape == shape

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_bshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape)

        assert node.bshape == tuple(d if d is not None else 1 for d in shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_size(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape)

        assert node.size == math.prod(filter(lambda d: d is not None, shape))  # type: ignore

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_init_reset(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        assert node.value.shape == (0,)
        initres = node.init(data)
        assert initres.shape == data.shape
        assert torch.allclose(initres, data)
        node.reset()
        assert node.value.shape == (0,)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some"),
        ids=("virtual_dims='none'", "virtual_dims='some'"),
    )
    def test_init_badshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = (3, 2, 5)
            case "some":
                shape = (3, None, 5)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape)
        data = torch.rand(
            2, *(d if d is not None else random.randint(1, 4) for d in shape), 4
        )

        with pytest.raises(ValueError) as excinfo:
            _ = node.init(data)
        assert "shape of `value`" in str(excinfo.value)
        assert "is incompatible with node shape" in str(excinfo.value)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some"),
        ids=("virtual_dims='none'", "virtual_dims='some'"),
    )
    @pytest.mark.parametrize(
        "cov_dim",
        ("scalar", 0, 1, 2),
        ids=("cov_dim='scalar'", "cov_dim=0", "cov_dim=1", "cov_dim=2"),
    )
    def test_prop_covariance(self, virtual_dims, cov_dim):
        match virtual_dims:
            case "none":
                shape = (3, 2, 5)
            case "some":
                shape = (3, None, 5)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape)

        match cov_dim:
            case "scalar":
                cov = torch.rand(()).item()
                sol = torch.full((node.size,), cov).diag()
            case 0:
                cov = torch.rand(())
                sol = torch.full((node.size,), cov.item()).diag()
            case 1:
                cov = torch.rand(node.size)
                sol = torch.full((node.size,), cov.mean().item()).diag()
            case 2:
                cov = torch.rand(node.size, node.size)
                cov = cov @ cov.t()
                sol = torch.full((node.size,), cov.diag().mean().item()).diag()
            case _:
                raise AssertionError

        node.covariance = cov
        res = node.covariance

        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_error(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        var = torch.rand(())
        node.covariance = var

        pred = torch.rand_like(data)

        with torch.no_grad():
            cov = torch.full((node.size,), var.item()).diag()
            diff, pragma = node.shapeobj.coalesce(data - pred)
            sol = node.shapeobj.disperse(torch.linalg.solve(cov, diff.t()).t(), pragma)
            res = node.error(pred)

        assert res.shape == data.shape
        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_energy(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        var = torch.rand(())
        node.covariance = var

        pred = torch.rand_like(data)

        with torch.no_grad():
            cov = torch.full((node.size,), var.item()).diag()
            diff, pragma = node.shapeobj.coalesce(data - pred)
            y = torch.linalg.solve(cov, diff.t()).t()

            diff = node.shapeobj.disperse(diff, pragma).flatten(1)
            y = node.shapeobj.disperse(y, pragma).flatten(1)
            logdet = node.size * var.log()

            sol = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()
            res = node.energy(pred)

        assert res.shape[:1] == data.shape[:1]
        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    @pytest.mark.parametrize(
        "training",
        (True, False),
        ids=("training=True", "training='False"),
    )
    def test_forward(self, virtual_dims, training):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = IsotropicGaussianNode(*shape).train(training)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        with torch.no_grad():
            res = node(data)

        assert res.shape == data.shape
        assert torch.allclose(res, data)

        if training:
            assert node.value.shape == data.shape
            assert torch.allclose(node.value, res)

    def test_estep_params(self):
        m = nn.ModuleList((IsotropicGaussianNode(4, None, 3, 3),))
        assert dict(get_named_estep_params(m)) == {"0.value": m[0].value}

    def test_mstep_params(self):
        m = nn.ModuleList((IsotropicGaussianNode(4, None, 3, 3),))
        assert dict(get_named_mstep_params(m)) == {"0.logvar": m[0].logvar}


class TestFactorizedGaussianNode:

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shapeobj(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape)

        assert node.shapeobj == (None, *shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape)

        assert node.shape == shape

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_bshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape)

        assert node.bshape == tuple(d if d is not None else 1 for d in shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_size(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape)

        assert node.size == math.prod(filter(lambda d: d is not None, shape))  # type: ignore

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_init_reset(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        assert node.value.shape == (0,)
        initres = node.init(data)
        assert initres.shape == data.shape
        assert torch.allclose(initres, data)
        node.reset()
        assert node.value.shape == (0,)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some"),
        ids=("virtual_dims='none'", "virtual_dims='some'"),
    )
    def test_init_badshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = (3, 2, 5)
            case "some":
                shape = (3, None, 5)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape)
        data = torch.rand(
            2, *(d if d is not None else random.randint(1, 4) for d in shape), 4
        )

        with pytest.raises(ValueError) as excinfo:
            _ = node.init(data)
        assert "shape of `value`" in str(excinfo.value)
        assert "is incompatible with node shape" in str(excinfo.value)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some"),
        ids=("virtual_dims='none'", "virtual_dims='some'"),
    )
    @pytest.mark.parametrize(
        "cov_dim",
        ("scalar", 0, 1, 2),
        ids=("cov_dim='scalar'", "cov_dim=0", "cov_dim=1", "cov_dim=2"),
    )
    def test_prop_covariance(self, virtual_dims, cov_dim):
        match virtual_dims:
            case "none":
                shape = (3, 2, 5)
            case "some":
                shape = (3, None, 5)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape)

        match cov_dim:
            case "scalar":
                cov = torch.rand(()).item()
                sol = torch.full((node.size,), cov).diag()
            case 0:
                cov = torch.rand(())
                sol = torch.full((node.size,), cov.item()).diag()
            case 1:
                cov = torch.rand(node.size)
                sol = cov.diag()
            case 2:
                cov = torch.rand(node.size, node.size)
                cov = cov @ cov.t()
                sol = cov.diag().diag()
            case _:
                raise AssertionError

        node.covariance = cov
        res = node.covariance

        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_error(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        var = torch.rand((node.size,))
        node.covariance = var

        pred = torch.rand_like(data)

        with torch.no_grad():
            cov = var.diag()
            diff, pragma = node.shapeobj.coalesce(data - pred)
            sol = node.shapeobj.disperse(torch.linalg.solve(cov, diff.t()).t(), pragma)
            res = node.error(pred)

        assert res.shape == data.shape
        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_energy(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        var = torch.rand((node.size,))
        node.covariance = var

        pred = torch.rand_like(data)

        with torch.no_grad():
            cov = var.diag()
            diff, pragma = node.shapeobj.coalesce(data - pred)
            y = torch.linalg.solve(cov, diff.t()).t()

            diff = node.shapeobj.disperse(diff, pragma).flatten(1)
            y = node.shapeobj.disperse(y, pragma).flatten(1)
            logdet = var.log().sum()

            sol = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()
            res = node.energy(pred)

        assert res.shape[:1] == data.shape[:1]
        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    @pytest.mark.parametrize(
        "training",
        (True, False),
        ids=("training=True", "training='False"),
    )
    def test_forward(self, virtual_dims, training):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = FactorizedGaussianNode(*shape).train(training)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        with torch.no_grad():
            res = node(data)

        assert res.shape == data.shape
        assert torch.allclose(res, data)

        if training:
            assert node.value.shape == data.shape
            assert torch.allclose(node.value, res)

    def test_estep_params(self):
        m = nn.ModuleList((FactorizedGaussianNode(4, None, 3, 3),))
        assert dict(get_named_estep_params(m)) == {"0.value": m[0].value}

    def test_mstep_params(self):
        m = nn.ModuleList((FactorizedGaussianNode(4, None, 3, 3),))
        assert dict(get_named_mstep_params(m)) == {"0.logvar": m[0].logvar}


class TestMultivariateGaussianNode:

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shapeobj(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape)

        assert node.shapeobj == (None, *shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape)

        assert node.shape == shape

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_bshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape)

        assert node.bshape == tuple(d if d is not None else 1 for d in shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_size(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape)

        assert node.size == math.prod(filter(lambda d: d is not None, shape))  # type: ignore

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_init_reset(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        assert node.value.shape == (0,)
        initres = node.init(data)
        assert initres.shape == data.shape
        assert torch.allclose(initres, data)
        node.reset()
        assert node.value.shape == (0,)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some"),
        ids=("virtual_dims='none'", "virtual_dims='some'"),
    )
    def test_init_badshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = (3, 2, 5)
            case "some":
                shape = (3, None, 5)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape)
        data = torch.rand(
            2, *(d if d is not None else random.randint(1, 4) for d in shape), 4
        )

        with pytest.raises(ValueError) as excinfo:
            _ = node.init(data)
        assert "shape of `value`" in str(excinfo.value)
        assert "is incompatible with node shape" in str(excinfo.value)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some"),
        ids=("virtual_dims='none'", "virtual_dims='some'"),
    )
    @pytest.mark.parametrize(
        "cov_dim",
        ("scalar", 0, 1, 2),
        ids=("cov_dim='scalar'", "cov_dim=0", "cov_dim=1", "cov_dim=2"),
    )
    def test_prop_covariance(self, virtual_dims, cov_dim):
        match virtual_dims:
            case "none":
                shape = (3, 2, 5)
            case "some":
                shape = (3, None, 5)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape)

        match cov_dim:
            case "scalar":
                cov = torch.rand(()).item()
                sol = torch.full((node.size,), cov).diag()
            case 0:
                cov = torch.rand(())
                sol = torch.full((node.size,), cov.item()).diag()
            case 1:
                cov = torch.rand(node.size)
                sol = cov.diag()
            case 2:
                cov = torch.rand(node.size, node.size)
                cov = cov @ cov.t()
                sol = cov
            case _:
                raise AssertionError

        node.covariance = cov
        res = node.covariance

        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_error(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        sqrtcov = torch.rand((node.size, node.size))
        cov = sqrtcov @ sqrtcov.t()
        node.covariance = cov

        pred = torch.rand_like(data)

        with torch.no_grad():
            diff, pragma = node.shapeobj.coalesce(data - pred)
            sol = node.shapeobj.disperse(torch.linalg.solve(cov, diff.t()).t(), pragma)
            res = node.error(pred)

        assert res.shape == data.shape
        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_energy(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        sqrtcov = torch.rand((node.size, node.size))
        cov = sqrtcov @ sqrtcov.t()
        node.covariance = cov

        pred = torch.rand_like(data)

        with torch.no_grad():
            L, info = torch.linalg.cholesky_ex(cov)
            assert info.item() == 0

            diff, pragma = node.shapeobj.coalesce(data - pred)
            y = torch.linalg.solve(cov, diff.t()).t()
            #u = torch.linalg.solve_triangular(L, diff.t(), upper=False)
            #y = torch.linalg.solve_triangular(L.t(), u, upper=True).t()

            diff = node.shapeobj.disperse(diff, pragma).flatten(1)
            y = node.shapeobj.disperse(y, pragma).flatten(1)
            logdet = 2.0 * L.diag().log().sum()

            sol = 0.5 * (diff.unsqueeze(1) @ y.unsqueeze(2) + logdet).flatten()
            res = node.energy(pred)

        assert res.shape[:1] == data.shape[:1]
        assert res.shape == sol.shape
        assert torch.allclose(res, sol)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    @pytest.mark.parametrize(
        "training",
        (True, False),
        ids=("training=True", "training='False"),
    )
    def test_forward(self, virtual_dims, training):
        match virtual_dims:
            case "none":
                shape = randshape(5, 2, 5, 0)
            case "some":
                shape = randshape(5, 2, 5, 2)
            case "all":
                shape = randshape(5, 2, 5, 4)
            case _:
                raise AssertionError

        node = MultivariateGaussianNode(*shape).train(training)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        with torch.no_grad():
            res = node(data)

        assert res.shape == data.shape
        assert torch.allclose(res, data)

        if training:
            assert node.value.shape == data.shape
            assert torch.allclose(node.value, res)

    def test_estep_params(self):
        m = nn.ModuleList((MultivariateGaussianNode(4, None, 3, 3),))
        assert dict(get_named_estep_params(m)) == {"0.value": m[0].value}

    def test_mstep_params(self):
        m = nn.ModuleList((MultivariateGaussianNode(4, None, 3, 3),))
        assert dict(get_named_mstep_params(m)) == {
            "0.covar_cf_logdiag": m[0].covar_cf_logdiag,
            "0.covar_cf_offtril": m[0].covar_cf_offtril,
        }
