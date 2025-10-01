from typing import Literal
import math
import pytest
import torch
import torch.nn.functional as F

from pyromancy import get_named_estep_params, get_named_mstep_params
from pyromancy.v2nodes import CategoricalNode


class TestCategoricalNode:

    @pytest.fixture
    def shape(self) -> tuple[int | None, ...]:
        return (3, None, 4, None, 2)

    @pytest.fixture
    def batch_shape(self, shape: tuple[int | None, ...]) -> tuple[int, ...]:
        return (8, *(s if s is not None else d + 1 for d, s in enumerate(shape)))

    @pytest.fixture
    def small_shape(self) -> tuple[int, ...]:
        return (3, 2)

    @pytest.fixture
    def small_batch_shape(self, small_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (6, *small_shape)

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
    def small_nvariate(self, small_shape: tuple[int, ...]) -> int:
        return math.prod(small_shape)

    @pytest.fixture
    def seed(self) -> int:
        return 42

    @pytest.mark.parametrize(
        "from_logits",
        (True, False),
        ids=("from_logits=True", "from_logits=False"),
    )
    def test_initialize_deterministic(
        self,
        from_logits: bool,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = CategoricalNode(*shape)
        g = torch.Generator().manual_seed(seed)

        if from_logits:
            qlogits = 100 * torch.rand(batch_shape, generator=g)
            qprobs, pragma = node.shapeobj.coalesce(qlogits)
            qprobs = F.softmax(qprobs, dim=1)
            qprobs = node.shapeobj.disperse(qprobs, pragma)
        else:
            pragma = node.shapeobj.pragma(torch.empty(batch_shape, device="meta"))
            qprobs = torch.randint(
                0, nvariate, (math.prod(batch_shape) // nvariate,), generator=g
            )
            qprobs = F.one_hot(qprobs, nvariate).float()
            qprobs = node.shapeobj.disperse(qprobs, pragma)
            qlogits = qprobs.clamp_min(1e-12).log()

        if from_logits:
            node.initialize(qlogits, from_logits=True)
        else:
            node.initialize(qprobs, from_logits=False)

        assert torch.all(node.logits == qlogits)

    @pytest.mark.parametrize(
        "as_logits",
        (True, False),
        ids=("as_logits=True", "as_logits=False"),
    )
    def test_activity(
        self,
        as_logits: bool,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        seed: int,
    ) -> None:
        node = CategoricalNode(*shape)
        g = torch.Generator().manual_seed(seed)

        qlogits = 100 * torch.rand(batch_shape, generator=g)
        qprobs, pragma = node.shapeobj.coalesce(qlogits)
        qprobs = F.softmax(qprobs, dim=1)
        qprobs = node.shapeobj.disperse(qprobs, pragma)

        node.initialize(qlogits)
        res = node.activity(as_logits=as_logits)

        if as_logits:
            assert torch.all(res == qlogits)
        else:
            assert torch.all(res == qprobs)

    @pytest.mark.parametrize(
        "energyfn",
        ("kld", "ce"),
        ids=("energyfn=kld", "energyfn=ce"),
    )
    @pytest.mark.parametrize(
        "q_from_logits",
        (True, False),
        ids=("q_from_logits=True", "q_from_logits=False"),
    )
    @pytest.mark.parametrize(
        "p_from_logits",
        (True, False),
        ids=("p_from_logits=True", "p_from_logits=False"),
    )
    def test_energy(
        self,
        energyfn: Literal["kld", "ce"],
        q_from_logits: bool,
        p_from_logits: bool,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        nvariate: int,
        seed: int,
    ) -> None:
        node = CategoricalNode(*shape)
        g = torch.Generator().manual_seed(seed)

        if q_from_logits:
            qlogits = 100 * torch.rand(batch_shape, generator=g)
            qprobs, pragma = node.shapeobj.coalesce(qlogits)
            qprobs = F.softmax(qprobs, dim=1)
            qprobs = node.shapeobj.disperse(qprobs, pragma)
            qlogits = qprobs.log()
        else:
            pragma = node.shapeobj.pragma(torch.empty(batch_shape, device="meta"))
            qprobs = torch.randint(
                0, nvariate, (math.prod(batch_shape) // nvariate,), generator=g
            )
            qprobs = F.one_hot(qprobs, nvariate).float()
            qprobs = node.shapeobj.disperse(qprobs, pragma)
            qlogits = qprobs.clamp_min(1e-12).log()

        if p_from_logits:
            plogits = 100 * torch.rand(batch_shape, generator=g)
            pprobs, pragma = node.shapeobj.coalesce(plogits)
            pprobs = F.softmax(pprobs, dim=1)
            pprobs = node.shapeobj.disperse(pprobs, pragma)
            plogits = pprobs.log()
        else:
            pragma = node.shapeobj.pragma(torch.empty(batch_shape, device="meta"))
            pprobs = torch.randint(
                0, nvariate, (math.prod(batch_shape) // nvariate,), generator=g
            )
            pprobs = F.one_hot(pprobs, nvariate).float()
            pprobs = node.shapeobj.disperse(pprobs, pragma)
            plogits = pprobs.clamp_min(1e-12).log()

        if q_from_logits:
            node.initialize(qlogits, from_logits=True)
        else:
            node.initialize(qprobs, from_logits=False)

        # normalize and coalesce probabilities
        if q_from_logits:
            q, pragma = node.shapeobj.coalesce(qprobs)
        else:
            q, pragma = node.shapeobj.coalesce(qprobs.clamp_min(1e-12))

        if p_from_logits:
            p, _ = node.shapeobj.coalesce(pprobs)
        else:
            p, _ = node.shapeobj.coalesce(pprobs.clamp_min(1e-12))

        # cross entropy term
        Hqp = -torch.special.xlogy(q, p).sum(1)

        # entropy term
        Hq = -torch.special.xlogy(q, q).sum(1)

        # reference energy
        match energyfn:
            case "kld":
                E = Hqp - Hq
            case "ce":
                E = Hqp

        E = node.shapeobj.disperse(E, pragma, "plate")
        E = E.flatten(1).sum(1)

        # compare
        if p_from_logits:
            testE = node.energy(plogits, from_logits=True, fn=energyfn)
        else:
            testE = node.energy(pprobs, from_logits=False, fn=energyfn)

        assert torch.allclose(testE, E)
        assert torch.all(torch.isfinite(testE))

    @pytest.mark.parametrize(
        "energyfn",
        ("kld", "ce"),
        ids=("energyfn=kld", "energyfn=ce"),
    )
    @pytest.mark.parametrize(
        "q_from_logits",
        (True, False),
        ids=("q_from_logits=True", "q_from_logits=False"),
    )
    @pytest.mark.parametrize(
        "p_from_logits",
        (True, False),
        ids=("p_from_logits=True", "p_from_logits=False"),
    )
    def test_error(
        self,
        energyfn: Literal["kld", "ce"],
        q_from_logits: bool,
        p_from_logits: bool,
        shape: tuple[int | None, ...],
        batch_shape: tuple[int, ...],
        seed: int,
    ) -> None:
        node = CategoricalNode(*shape)
        g = torch.Generator().manual_seed(seed)

        zq = 10 * torch.rand(batch_shape, generator=g)
        _zq, pragma = node.shapeobj.coalesce(zq)
        q = F.softmax(_zq, dim=1)
        q = node.shapeobj.disperse(q, pragma)
        logq = F.log_softmax(_zq, dim=1)
        logq = node.shapeobj.disperse(logq, pragma)

        zp = 10 * torch.rand(batch_shape, generator=g)
        _zp, pragma = node.shapeobj.coalesce(zp)
        p = F.softmax(_zp, dim=1)
        p = node.shapeobj.disperse(p, pragma)
        logp = F.log_softmax(_zp, dim=1)
        logp = node.shapeobj.disperse(logp, pragma)

        if q_from_logits:
            node.initialize(zq, from_logits=True)
        else:
            node.initialize(q, from_logits=False)

        # derived error
        if p_from_logits:
            E = node.energy(zp, from_logits=True, fn=energyfn)
        else:
            E = node.energy(p, from_logits=False, fn=energyfn)

        (grad,) = torch.autograd.grad(
            outputs=E,
            inputs=node.logits,
            grad_outputs=torch.ones_like(E),
        )

        # computed error
        if p_from_logits:
            err = node.error(zp, from_logits=True, fn=energyfn)
        else:
            err = node.error(p, from_logits=False, fn=energyfn)

        # compare
        assert torch.allclose(err, grad, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize(
        "source",
        ("logits", "probs", "internal"),
        ids=("source=logits", "source=probs", "source=internal"),
    )
    def test_sample_cdf_uniform(
        self,
        source: Literal["logits", "probs", "internal"],
        small_shape: tuple[int | None, ...],
        small_batch_shape: tuple[int, ...],
        small_nvariate: int,
        seed: int,
    ) -> None:
        node = CategoricalNode(*small_shape, eps_probs=0.0)

        pragma = node.shapeobj.pragma(torch.empty(small_batch_shape, device="meta"))
        base = torch.ones(small_nvariate) / 2 ** torch.arange(small_nvariate)
        probs = base / base.sum()
        probs = torch.stack([probs.roll(k, 0) for k in range(small_batch_shape[0])])
        probs = node.shapeobj.disperse(probs, pragma)
        logits = probs.log()

        # reference
        g = torch.Generator().manual_seed(seed)
        probs_, pragma = node.shapeobj.coalesce(probs)

        cdf = probs_.cumsum(1)
        cdf[:, -1] = 1.0

        uniforms = cdf.new_empty(cdf.size(0), 1).uniform_(generator=g)
        idx = torch.searchsorted(cdf, uniforms, right=True)

        y = torch.zeros_like(probs_).scatter_(1, idx, 1.0)
        y = node.shapeobj.disperse(y, pragma)

        # check equivalence
        g = torch.Generator().manual_seed(seed)

        match source:
            case "logits":
                ytest = node.sample(
                    logits, from_logits=True, generator=g, procedure="cdf-uniform"
                )
            case "probs":
                ytest = node.sample(
                    probs, from_logits=False, generator=g, procedure="cdf-uniform"
                )
            case "internal":
                node.initialize(logits)
                ytest = node.sample(None, generator=g, procedure="cdf-uniform")

        assert torch.all(ytest == y)

        # check sanity
        testres = torch.zeros_like(y)
        for _ in range(1000):
            match source:
                case "logits":
                    y = node.sample(
                        logits, from_logits=True, generator=g, procedure="cdf-uniform"
                    )
                case "probs":
                    y = node.sample(
                        probs, from_logits=False, generator=g, procedure="cdf-uniform"
                    )
                case "internal":
                    node.initialize(logits)
                    y = node.sample(None, generator=g, procedure="cdf-uniform")
            testres += y

        testres, _ = node.shapeobj.coalesce(testres)
        testres = torch.sort(testres, dim=1).indices

        res, _ = node.shapeobj.coalesce(probs)
        res = torch.sort(res, dim=1).indices

        assert torch.all(testres == res)

    @pytest.mark.parametrize(
        "source",
        ("logits", "probs", "internal"),
        ids=("source=logits", "source=probs", "source=internal"),
    )
    def test_sample_continuous_gumbel_softmax(
        self,
        source: Literal["logits", "probs", "internal"],
        small_shape: tuple[int | None, ...],
        small_batch_shape: tuple[int, ...],
        small_nvariate: int,
        seed: int,
    ) -> None:
        node = CategoricalNode(*small_shape, eps_probs=0.0)

        pragma = node.shapeobj.pragma(torch.empty(small_batch_shape, device="meta"))
        base = torch.ones(small_nvariate) / 2 ** torch.arange(small_nvariate)
        probs = base / base.sum()
        probs = torch.stack([probs.roll(k, 0) for k in range(small_batch_shape[0])])
        probs = node.shapeobj.disperse(probs, pragma)
        logits = probs.log()

        # reference
        g = torch.Generator().manual_seed(seed)
        logits_, pragma = node.shapeobj.coalesce(logits)

        gumbels = -torch.empty_like(logits_).exponential_(generator=g).log()
        gumbels = (logits_ + gumbels)

        y = F.softmax(gumbels, dim=1)

        y = node.shapeobj.disperse(y, pragma)

        # check equivalence
        g = torch.Generator().manual_seed(seed)

        match source:
            case "logits":
                ytest = node.sample(
                    logits,
                    from_logits=True,
                    generator=g,
                    procedure="gumbel-softmax",
                    discrete=False,
                )
            case "probs":
                ytest = node.sample(
                    probs,
                    from_logits=False,
                    generator=g,
                    procedure="gumbel-softmax",
                    discrete=False,
                )
            case "internal":
                node.initialize(logits)
                ytest = node.sample(
                    None, generator=g, procedure="gumbel-softmax", discrete=False
                )

        assert torch.all(ytest == y)

        # check sanity
        testres = torch.zeros_like(y)
        for _ in range(1000):
            match source:
                case "logits":
                    y = node.sample(
                        logits,
                        from_logits=True,
                        generator=g,
                        procedure="gumbel-softmax",
                        discrete=False,
                    )
                case "probs":
                    y = node.sample(
                        probs,
                        from_logits=False,
                        generator=g,
                        procedure="gumbel-softmax",
                        discrete=False,
                    )
                case "internal":
                    node.initialize(logits)
                    y = node.sample(
                        None, generator=g, procedure="gumbel-softmax", discrete=False
                    )
            testres += y

        testres, _ = node.shapeobj.coalesce(testres)
        testres = torch.sort(testres, dim=1).indices

        res, _ = node.shapeobj.coalesce(probs)
        res = torch.sort(res, dim=1).indices

        assert torch.all(testres == res)

    @pytest.mark.parametrize(
        "source",
        ("logits", "probs", "internal"),
        ids=("source=logits", "source=probs", "source=internal"),
    )
    def test_sample_discrete_gumbel_softmax(
        self,
        source: Literal["logits", "probs", "internal"],
        small_shape: tuple[int | None, ...],
        small_batch_shape: tuple[int, ...],
        small_nvariate: int,
        seed: int,
    ) -> None:
        node = CategoricalNode(*small_shape, eps_probs=0.0)

        pragma = node.shapeobj.pragma(torch.empty(small_batch_shape, device="meta"))
        base = torch.ones(small_nvariate) / 2 ** torch.arange(small_nvariate)
        probs = base / base.sum()
        probs = torch.stack([probs.roll(k, 0) for k in range(small_batch_shape[0])])
        probs = node.shapeobj.disperse(probs, pragma)
        logits = probs.log()

        # reference
        g = torch.Generator().manual_seed(seed)
        logits_, pragma = node.shapeobj.coalesce(logits)

        gumbels = -torch.empty_like(logits_).exponential_(generator=g).log()
        gumbels = (logits_ + gumbels)

        y = F.softmax(gumbels, dim=1)

        idx = y.argmax(1, keepdim=True)
        yd = torch.zeros_like(logits_).scatter_(1, idx, 1.0)
        y = yd - y.detach() + y

        y = node.shapeobj.disperse(y, pragma)

        # check equivalence
        g = torch.Generator().manual_seed(seed)

        match source:
            case "logits":
                ytest = node.sample(
                    logits,
                    from_logits=True,
                    generator=g,
                    procedure="gumbel-softmax",
                    discrete=True,
                )
            case "probs":
                ytest = node.sample(
                    probs,
                    from_logits=False,
                    generator=g,
                    procedure="gumbel-softmax",
                    discrete=True,
                )
            case "internal":
                node.initialize(logits)
                ytest = node.sample(
                    None, generator=g, procedure="gumbel-softmax", discrete=True
                )

        assert torch.all(ytest == y)

        # check sanity
        testres = torch.zeros_like(y)
        for _ in range(1000):
            match source:
                case "logits":
                    y = node.sample(
                        logits,
                        from_logits=True,
                        generator=g,
                        procedure="gumbel-softmax",
                        discrete=True,
                    )
                case "probs":
                    y = node.sample(
                        probs,
                        from_logits=False,
                        generator=g,
                        procedure="gumbel-softmax",
                        discrete=True,
                    )
                case "internal":
                    node.initialize(logits)
                    y = node.sample(
                        None, generator=g, procedure="gumbel-softmax", discrete=True
                    )
            testres += y

        testres, _ = node.shapeobj.coalesce(testres)
        testres = torch.sort(testres, dim=1).indices

        res, _ = node.shapeobj.coalesce(probs)
        res = torch.sort(res, dim=1).indices

        assert torch.all(testres == res)

    def test_estep_params(self, shape: tuple[int | None, ...]) -> None:
        node = CategoricalNode(*shape)
        estep = dict(get_named_estep_params(node))
        sol = {"logits": node.logits}
        assert estep == sol

    def test_mstep_params(self, shape: tuple[int | None, ...]) -> None:
        node = CategoricalNode(*shape)
        mstep = dict(get_named_mstep_params(node))
        sol = dict()
        assert mstep == sol
