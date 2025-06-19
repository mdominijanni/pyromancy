import math
import pytest
import random
import torch

from pyromancy import Shape


class TestShape:

    def test_init_empty(self):
        with pytest.raises(ValueError) as excinfo:
            _ = Shape()
        assert "`shape` must contain at least one element" in str(excinfo.value)

    @pytest.mark.parametrize("shape", ((1, None, 0.0), (2, "3", 4)))
    def test_init_badtype(self, shape):
        with pytest.raises(TypeError) as excinfo:
            _ = Shape(*shape)
        assert "all elements of `shape` must be of type `int` or `None`" in str(
            excinfo.value
        )

    @pytest.mark.parametrize("shape", ((10, 0, None), (2, -3, 4)))
    def test_init_badvalue(self, shape):
        with pytest.raises(ValueError) as excinfo:
            _ = Shape(*shape)
        assert "all integer elements of `shape` must be positive" in str(excinfo.value)

    @pytest.mark.parametrize(
        "shape",
        ((5, None, None, 3, None), (None, None), (3, 4)),
        ids=("mixed", "all_virtual", "all_concrete"),
    )
    def test_prop_rawshape(self, shape):
        shp = Shape(*shape)

        sol = shape
        res = shp.rawshape

        assert sol == res

    @pytest.mark.parametrize(
        "shape",
        ((5, None, None, 3, None), (None, None), (3, 4)),
        ids=("mixed", "all_virtual", "all_concrete"),
    )
    def test_prop_shape(self, shape):
        shp = Shape(*shape)

        sol = tuple(map(lambda d: d if isinstance(d, int) else 1, shape))
        res = shp.shape

        assert sol == res

    @pytest.mark.parametrize(
        "shape",
        ((5, None, None, 3, None), (None, None), (3, 4)),
        ids=("mixed", "all_virtual", "all_concrete"),
    )
    def test_prop_size(self, shape):
        shp = Shape(*shape)

        sol = math.prod(map(lambda d: d if isinstance(d, int) else 1, shape))
        res = shp.size

        assert sol == res

    @pytest.mark.parametrize(
        "shape",
        ((5, None, None, 3, None), (None, None), (3, 4)),
        ids=("mixed", "all_virtual", "all_concrete"),
    )
    def test_prop_ndim(self, shape):
        shp = Shape(*shape)

        sol = len(shape)
        res = shp.ndim

        assert sol == res

    @pytest.mark.parametrize(
        "shape",
        ((5, None, None, 3, None), (None, None), (3, 4)),
        ids=("mixed", "all_virtual", "all_concrete"),
    )
    def test_prop_nconcrete(self, shape):
        shp = Shape(*shape)

        sol = sum(map(lambda d: 1 if isinstance(d, int) else 0, shape))
        res = shp.nconcrete

        assert sol == res

    @pytest.mark.parametrize(
        "shape",
        ((5, None, None, 3, None), (None, None), (3, 4)),
        ids=("mixed", "all_virtual", "all_concrete"),
    )
    def test_prop_nvirtual(self, shape):
        shp = Shape(*shape)

        sol = sum(map(lambda d: 0 if isinstance(d, int) else 1, shape))
        res = shp.nvirtual

        assert sol == res

    @pytest.mark.parametrize("shape", ((1, None), (2, "3", 4)))
    def test_compat_badtype(self, shape):
        shp = Shape(*(5, None, None, 3, None))
        with pytest.raises(TypeError) as excinfo:
            _ = shp.compat(*shape)
        assert "all elements of `shape` must be of type `int`" in str(excinfo.value)

    @pytest.mark.parametrize("shape", ((10, 0, 1), (2, -3, 4)))
    def test_compat_badvalue(self, shape):
        shp = Shape(*(5, None, None, 3, None))
        with pytest.raises(ValueError) as excinfo:
            _ = shp.compat(*shape)
        assert "all elements of `shape` must be positive" in str(excinfo.value)

    @pytest.mark.parametrize(
        "shape, cmpshp",
        (
            ((None, 5, None), (5, 1, 5)),
            ((None, 5, None), (1, 5, 1, 1)),
            ((None, 5, None), (1, 5)),
        ),
        ids=("bad_dim", "too_many_dims", "too_few_dims"),
    )
    def test_compat_false(self, shape, cmpshp):
        shp = Shape(*shape)
        assert not shp.compat(*cmpshp)

    @pytest.mark.parametrize(
        "shape, cmpshp",
        (
            ((None, 5, None), (3, 5, 4)),
            ((None, None), (3, 4)),
            ((5, 4), (5, 4)),
        ),
        ids=("mixed", "all_virtual", "all_concrete"),
    )
    def test_compat_true(self, shape, cmpshp):
        shp = Shape(*shape)
        assert shp.compat(*cmpshp)

    @pytest.mark.parametrize("fill", ((1, 2), (2, 3, 4, 5)))
    def test_filled_badlen(self, fill):
        shp = Shape(*(5, None, None, 3, None))
        with pytest.raises(ValueError) as excinfo:
            _ = shp.filled(*fill)
        assert (
            "`fill` must contain exactly the required number of placeholder elements"
            in str(excinfo.value)
        )

    @pytest.mark.parametrize("fill", ((1, 2.0, 3), (2, "3", 4)))
    def test_filled_badtype(self, fill):
        shp = Shape(*(5, None, None, 3, None))
        with pytest.raises(TypeError) as excinfo:
            _ = shp.filled(*fill)
        assert "all elements of `fill` must be of type `int`" in str(excinfo.value)

    @pytest.mark.parametrize("fill", ((10, 0, 1), (2, -3, 4)))
    def test_filled_badvalue(self, fill):
        shp = Shape(*(5, None, None, 3, None))
        with pytest.raises(ValueError) as excinfo:
            _ = shp.filled(*fill)
        assert "all elements of `fill` must be positive" in str(excinfo.value)

    @pytest.mark.parametrize(
        "shape",
        ((5, None, None, 3, None), (None, None), (3, 4)),
        ids=("mixed", "all_virtual", "all_concrete"),
    )
    def test_coalesce(self, shape):
        shp = Shape(*shape)

        data = torch.rand(
            *map(lambda d: d if isinstance(d, int) else random.randint(1, 9), shape)
        )
        nv = math.prod(
            1 if isinstance(r, int) else d for d, r in zip(data.shape, shape)
        )
        nc = math.prod(
            d if isinstance(r, int) else 1 for d, r in zip(data.shape, shape)
        )

        res, _ = shp.coalesce(data)

        assert tuple(res.shape) == (nv, nc)

    @pytest.mark.parametrize(
        "shape",
        ((5, None, None, 3, None), (None, None), (3, 4)),
        ids=("mixed", "all_virtual", "all_concrete"),
    )
    def test_disperse_coalesce(self, shape):
        shp = Shape(*shape)

        data = torch.rand(
            *map(lambda d: d if isinstance(d, int) else random.randint(1, 9), shape)
        )

        tmp, pragma = shp.coalesce(data)
        res = shp.disperse(tmp, pragma)

        assert res.shape == data.shape
        assert torch.all(res == data)
