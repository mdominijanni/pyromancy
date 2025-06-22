import math
import random

import pytest
import torch
import torch.nn as nn

from pyromancy import get_named_estep_params, get_named_mstep_params
from pyromancy.nodes import BiasNode, FixedNode, FloatNode

from ..common import randshape


class TestBiasNode:

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_init(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = BiasNode(*shape)

        assert tuple(node.bias.shape) == tuple(d if d else 1 for d in shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shapeobj(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = BiasNode(*shape)

        assert node.shapeobj == (None, *shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = BiasNode(*shape)

        assert node.shape == shape

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_bshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = BiasNode(*shape)

        assert node.bshape == tuple(d if d is not None else 1 for d in shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_size(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = BiasNode(*shape)

        assert node.size == math.prod(filter(lambda d: d is not None, shape))  # type: ignore

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_error(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = BiasNode(*shape)
        with torch.no_grad():
            node.bias.copy_(torch.rand_like(node.bias))
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        with torch.no_grad():
            sol = node.bias.unsqueeze(0) - data
            res = node.error(data)

        assert sol.shape == res.shape
        assert torch.allclose(sol, res)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_forward(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = BiasNode(*shape)
        with torch.no_grad():
            node.bias.copy_(torch.rand_like(node.bias))
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )

        with torch.no_grad():
            sol = node.bias.unsqueeze(0).expand_as(data)
            res = node(data)

        assert sol.shape == res.shape
        assert torch.allclose(sol, res)

    def test_estep_params(self):
        m = nn.ModuleList((BiasNode(4, None, 3, 3),))
        assert dict(get_named_estep_params(m)) == {}

    def test_mstep_params(self):
        m = nn.ModuleList((BiasNode(4, None, 3, 3),))
        assert dict(get_named_mstep_params(m)) == {"0.bias": m[0].bias}


class TestFixedNode:

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shapeobj(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FixedNode(*shape)

        assert node.shapeobj == (None, *shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FixedNode(*shape)

        assert node.shape == shape

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_bshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FixedNode(*shape)

        assert node.bshape == tuple(d if d is not None else 1 for d in shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_size(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FixedNode(*shape)

        assert node.size == math.prod(filter(lambda d: d is not None, shape))  # type: ignore

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_init_reset(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FixedNode(*shape)
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

        node = FixedNode(*shape)
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
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FixedNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        pred = torch.rand_like(data)

        sol = data - pred
        res = node.error(pred)

        assert sol.shape == res.shape
        assert torch.allclose(sol, res)

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
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FixedNode(*shape).train(training)
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
        m = nn.ModuleList((FixedNode(4, None, 3, 3),))
        assert dict(get_named_estep_params(m)) == {}

    def test_mstep_params(self):
        m = nn.ModuleList((FixedNode(4, None, 3, 3),))
        assert dict(get_named_mstep_params(m)) == {}


class TestFloatNode:

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shapeobj(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FloatNode(*shape)

        assert node.shapeobj == (None, *shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_shape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FloatNode(*shape)

        assert node.shape == shape

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_bshape(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FloatNode(*shape)

        assert node.bshape == tuple(d if d is not None else 1 for d in shape)

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_prop_size(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FloatNode(*shape)

        assert node.size == math.prod(filter(lambda d: d is not None, shape))  # type: ignore

    @pytest.mark.parametrize(
        "virtual_dims",
        ("none", "some", "all"),
        ids=("virtual_dims='none'", "virtual_dims='some'", "virtual_dims='all'"),
    )
    def test_init_reset(self, virtual_dims):
        match virtual_dims:
            case "none":
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FloatNode(*shape)
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

        node = FloatNode(*shape)
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
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FloatNode(*shape)
        data = torch.rand(
            random.randint(2, 5),
            *(d if d is not None else random.randint(1, 4) for d in shape),
        )
        _ = node.init(data)

        pred = torch.rand_like(data)

        sol = data - pred
        res = node.error(pred)

        assert sol.shape == res.shape
        assert torch.allclose(sol, res)

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
                shape = randshape(4, 2, 5, 0)
            case "some":
                shape = randshape(4, 2, 5, 2)
            case "all":
                shape = randshape(4, 2, 5, 4)
            case _:
                raise AssertionError

        node = FloatNode(*shape).train(training)
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
        m = nn.ModuleList((FloatNode(4, None, 3, 3),))
        assert dict(get_named_estep_params(m)) == {"0.value": m[0].value}

    def test_mstep_params(self):
        m = nn.ModuleList((FloatNode(4, None, 3, 3),))
        assert dict(get_named_mstep_params(m)) == {}
