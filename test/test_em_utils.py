import pytest
import torch
import torch.nn as nn

from pyromancy import get_named_estep_params


class UnspecifiedEM(nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


class SpecifiedE(nn.Module):
    _e_params_ = ["e1", "e2"]

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.e1 = nn.Parameter(torch.rand(()), False)
        self.e2 = nn.Parameter(torch.rand(()), False)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


class SpecifiedM(nn.Module):
    _m_params_ = ["m1", "m2"]

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.m1 = nn.Parameter(torch.rand(()), False)
        self.m2 = nn.Parameter(torch.rand(()), False)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


class SpecifiedEM(nn.Module):
    _e_params_ = ["e1", "e2"]
    _m_params_ = ["m1", "m2"]

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.e1 = nn.Parameter(torch.rand(()), False)
        self.e2 = nn.Parameter(torch.rand(()), False)
        self.m1 = nn.Parameter(torch.rand(()), False)
        self.m2 = nn.Parameter(torch.rand(()), False)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


class TestGetNamedEStepParams:

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_eparam(self, default):
        m = SpecifiedE()
        m.inner = SpecifiedE()

        sol = {"e1": m.e1, "e2": m.e2, "inner.e1": m.inner.e1, "inner.e2": m.inner.e2}
        res = {n: p for n, p in get_named_estep_params(m, default=default)}

        assert len(sol) == len(res)
        for k in sol:
            assert k in res
            assert res[k] is sol[k]

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_mparam(self, default):
        m = SpecifiedM()
        m.inner = SpecifiedM()

        if default:
            sol = {
                "p1": m.p1,
                "p2": m.p2,
                "inner.p1": m.inner.p1,
                "inner.p2": m.inner.p2,
            }
        else:
            sol = {}
        res = {n: p for n, p in get_named_estep_params(m, default=default)}

        assert len(sol) == len(res)
        for k in sol:
            assert k in res
            assert res[k] is sol[k]

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_emparam(self, default):
        m = SpecifiedEM()
        m.inner1 = SpecifiedE()
        m.inner2 = SpecifiedM()

        sol = {
            "e1": m.e1,
            "e2": m.e2,
            "inner1.e1": m.inner1.e1,
            "inner1.e2": m.inner1.e2,
        }
        if default:
            sol = sol | {"inner2.p1": m.inner2.p1, "inner2.p2": m.inner2.p2}
        res = {n: p for n, p in get_named_estep_params(m, default=default)}

        assert len(sol) == len(res)
        for k in sol:
            assert k in res
            assert res[k] is sol[k]

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_unspecified_emparam(self, default):
        m = UnspecifiedEM()
        m.inner = UnspecifiedEM()

        if default:
            sol = {
                "p1": m.p1,
                "p2": m.p2,
                "inner.p1": m.inner.p1,
                "inner.p2": m.inner.p2,
            }
        else:
            sol = {}
        res = {n: p for n, p in get_named_estep_params(m, default=default)}

        assert len(sol) == len(res)
        for k in sol:
            assert k in res
            assert res[k] is sol[k]
