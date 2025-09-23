import pytest
import torch

from pyromancy import get_named_estep_params, get_named_mstep_params
from pyromancy.v2nodes import BiasNode, InputNode


class TestBiasNode:

    @pytest.fixture
    def shape(self) -> tuple[int | None, ...]:
        return (3, None, 4, None, 2)

    @pytest.fixture
    def batch_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (8, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    def test_activity_uninitialized(self, shape: tuple[int | None, ...]) -> None:
        node = BiasNode(*shape)
        bshape = tuple(s if s is not None else 1 for s in shape)
        assert node.activity().shape == (1, *bshape)

    def test_initialize(
        self, shape: tuple[int | None, ...], batch_shape: tuple[int, ...]
    ) -> None:
        node = BiasNode(*shape)
        bshape = tuple(s if s is not None else 1 for s in shape)
        node.initialize(torch.empty(batch_shape, device="meta"))
        assert node.activity().shape == batch_shape
        assert node.bias.shape == bshape

    def test_prediction(
        self, shape: tuple[int | None, ...], batch_shape: tuple[int, ...]
    ) -> None:
        node = BiasNode(*shape)
        pred = node.prediction(torch.empty(batch_shape, device="meta"))
        assert torch.all(pred == node.activity())

    def test_reset(
        self, shape: tuple[int | None, ...], batch_shape: tuple[int, ...]
    ) -> None:
        node = BiasNode(*shape)
        bshape = tuple(s if s is not None else 1 for s in shape)
        node.initialize(torch.empty(batch_shape, device="meta"))
        node.reset()
        assert node.activity().shape == (1, *bshape)
        assert node.bias.shape == bshape

    def test_estep_params(self, shape: tuple[int | None, ...]) -> None:
        node = BiasNode(*shape)
        estep = dict(get_named_estep_params(node))
        assert len(estep) == 0

    def test_mstep_params(self, shape: tuple[int | None, ...]) -> None:
        node = BiasNode(*shape)
        mstep = dict(get_named_mstep_params(node))
        assert set(mstep.keys()) == set(("bias",))
        assert mstep["bias"] is node.bias


class TestInputNode:

    @pytest.fixture
    def shape(self) -> tuple[int | None, ...]:
        return (3, None, 4, None, 2)

    @pytest.fixture
    def batch_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (8, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    @pytest.fixture
    def seed(self) -> int:
        return 42

    @pytest.mark.parametrize(
        "trainable",
        (True, False),
        ids=("trainable=True", "trainable=False"),
    )
    def test_initialize(
        self, trainable: bool, shape: tuple[int | None, ...], batch_shape: tuple[int, ...], seed: int
    ) -> None:
        node = InputNode(*shape, trainable=trainable)
        g = torch.Generator().manual_seed(seed)

        z = torch.rand(batch_shape, generator=g)
        node.initialize(z)
        assert torch.all(node.activity() == z)

    @pytest.mark.parametrize(
        "trainable",
        (True, False),
        ids=("trainable=True", "trainable=False"),
    )
    def test_prediction(
        self, trainable: bool, shape: tuple[int | None, ...], batch_shape: tuple[int, ...], seed: int
    ) -> None:
        node = InputNode(*shape, trainable=trainable)
        g = torch.Generator().manual_seed(seed)

        z = torch.rand(batch_shape, generator=g)
        pred = node.prediction(z)
        assert torch.all(pred == z)

    @pytest.mark.parametrize(
        "trainable",
        (True, False),
        ids=("trainable=True", "trainable=False"),
    )
    def test_reset(
        self, trainable: bool, shape: tuple[int | None, ...], batch_shape: tuple[int, ...]
    ) -> None:
        node = InputNode(*shape, trainable=trainable)
        z = torch.rand(batch_shape)

        node.initialize(z)
        node.reset()
        assert node.activity().numel() == 0

    @pytest.mark.parametrize(
        "trainable",
        (True, False),
        ids=("trainable=True", "trainable=False"),
    )
    def test_estep_params(self, trainable: bool, shape: tuple[int | None, ...]) -> None:
        node = InputNode(*shape, trainable=trainable)
        estep = dict(get_named_estep_params(node))
        if trainable:
            assert set(estep.keys()) == set(("value",))
            assert estep["value"] is node.value
        else:
            assert len(estep) == 0

    @pytest.mark.parametrize(
        "trainable",
        (True, False),
        ids=("trainable=True", "trainable=False"),
    )
    def test_mstep_params(self, trainable: bool, shape: tuple[int | None, ...]) -> None:
        node = InputNode(*shape, trainable=trainable)
        mstep = dict(get_named_mstep_params(node))
        assert len(mstep) == 0
