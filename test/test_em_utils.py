import pytest
import torch
import torch.nn as nn

from pyromancy import (
    eparameters,
    mparameters,
    get_named_estep_params,
    get_estep_params,
    get_named_mstep_params,
    get_mstep_params,
)


class UnspecifiedEM(nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


@eparameters("e1", "e2")
class SpecifiedE(nn.Module):

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.e1 = nn.Parameter(torch.rand(()), False)
        self.e2 = nn.Parameter(torch.rand(()), False)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


@mparameters("m1", "m2")
class SpecifiedM(nn.Module):

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.m1 = nn.Parameter(torch.rand(()), False)
        self.m2 = nn.Parameter(torch.rand(()), False)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


@eparameters("e1", "e2")
@mparameters("m1", "m2")
class SpecifiedEM(nn.Module):

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.e1 = nn.Parameter(torch.rand(()), False)
        self.e2 = nn.Parameter(torch.rand(()), False)
        self.m1 = nn.Parameter(torch.rand(()), False)
        self.m2 = nn.Parameter(torch.rand(()), False)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


@eparameters()
@mparameters()
class InheritedEM(SpecifiedE, SpecifiedM):

    def __init__(self) -> None:

        SpecifiedE.__init__(self)
        SpecifiedM.__init__(self)


@eparameters("e1", "e2")
class SpecifiedEMixin(nn.Module):

    def __init__(self) -> None:
        self.e1 = nn.Parameter(torch.rand(()), False)
        self.e2 = nn.Parameter(torch.rand(()), False)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


@mparameters("m1", "m2")
class SpecifiedMMixin(nn.Module):

    def __init__(self) -> None:
        self.m1 = nn.Parameter(torch.rand(()), False)
        self.m2 = nn.Parameter(torch.rand(()), False)
        self.p1 = nn.Parameter(torch.rand(()), False)
        self.p2 = nn.Parameter(torch.rand(()), False)


class ImproperlyInheritedEM(SpecifiedE, SpecifiedM):

    def __init__(self) -> None:
        nn.Module.__init__(self)
        SpecifiedEMixin.__init__(self)
        SpecifiedMMixin.__init__(self)


class TestEParameters:

    def test_inherited_eparams(self):
        m = InheritedEM()

        sol = {"e1": None, "e2": None}
        res = m._e_params_

        assert sol == res

    def test_direct_eparams(self):
        m = SpecifiedEM()

        sol = {"e1": None, "e2": None}
        res = m._e_params_

        assert sol == res


class TestMParameters:

    def test_inherited_mparams(self):
        m = InheritedEM()

        sol = {"m1": None, "m2": None}
        res = m._m_params_

        assert sol == res

    def test_direct_mparams(self):
        m = SpecifiedEM()

        sol = {"m1": None, "m2": None}
        res = m._m_params_

        assert sol == res


class TestGetNamedEStepParams:

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_improperly_inherited_eparam(self, default):
        m = ImproperlyInheritedEM()

        sol = {"e1": m.e1, "e2": m.e2}
        res = {n: p for n, p in get_named_estep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_eparam(self, default):
        m = SpecifiedE()
        m.inner = SpecifiedE()

        sol = {"e1": m.e1, "e2": m.e2, "inner.e1": m.inner.e1, "inner.e2": m.inner.e2}
        res = {n: p for n, p in get_named_estep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_mparam(self, default):
        m = SpecifiedM()
        m.inner = SpecifiedM()

        sol = dict()
        if default:
            sol |= {
                "p1": m.p1,
                "p2": m.p2,
                "inner.p1": m.inner.p1,
                "inner.p2": m.inner.p2,
            }
        res = {n: p for n, p in get_named_estep_params(m, default=default)}

        assert sol == res

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
            sol |= {"inner2.p1": m.inner2.p1, "inner2.p2": m.inner2.p2}
        res = {n: p for n, p in get_named_estep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_unspecified_emparam(self, default):
        m = UnspecifiedEM()
        m.inner = UnspecifiedEM()

        sol = dict()
        if default:
            sol |= {
                "p1": m.p1,
                "p2": m.p2,
                "inner.p1": m.inner.p1,
                "inner.p2": m.inner.p2,
            }
        res = {n: p for n, p in get_named_estep_params(m, default=default)}

        assert sol == res


class TestGetEStepParams:

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_improperly_inherited_eparam(self, default):
        m = ImproperlyInheritedEM()

        sol = {m.e1, m.e2}
        res = {*get_estep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_eparam(self, default):
        m = SpecifiedE()
        m.inner = SpecifiedE()

        sol = {m.e1, m.e2, m.inner.e1, m.inner.e2}
        res = {*get_estep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_mparam(self, default):
        m = SpecifiedM()
        m.inner = SpecifiedM()

        sol = set()
        if default:
            sol |= {m.p1, m.p2, m.inner.p1, m.inner.p2}
        res = {*get_estep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_emparam(self, default):
        m = SpecifiedEM()
        m.inner1 = SpecifiedE()
        m.inner2 = SpecifiedM()

        sol = {m.e1, m.e2, m.inner1.e1, m.inner1.e2}
        if default:
            sol |= {m.inner2.p1, m.inner2.p2}
        res = {*get_estep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_unspecified_emparam(self, default):
        m = UnspecifiedEM()
        m.inner = UnspecifiedEM()

        sol = set()
        if default:
            sol |= {m.p1, m.p2, m.inner.p1, m.inner.p2}
        res = {*get_estep_params(m, default=default)}

        assert sol == res


class TestGetNamedMStepParams:

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_improperly_inherited_mparam(self, default):
        m = ImproperlyInheritedEM()

        sol = {"m1": m.m1, "m2": m.m2}
        res = {n: p for n, p in get_named_mstep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_eparam(self, default):
        m = SpecifiedE()
        m.inner = SpecifiedE()

        sol = dict()
        if default:
            sol |= {
                "p1": m.p1,
                "p2": m.p2,
                "inner.p1": m.inner.p1,
                "inner.p2": m.inner.p2,
            }
        res = {n: p for n, p in get_named_mstep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_mparam(self, default):
        m = SpecifiedM()
        m.inner = SpecifiedM()

        sol = {"m1": m.m1, "m2": m.m2, "inner.m1": m.inner.m1, "inner.m2": m.inner.m2}
        res = {n: p for n, p in get_named_mstep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_emparam(self, default):
        m = SpecifiedEM()
        m.inner1 = SpecifiedE()
        m.inner2 = SpecifiedM()

        sol = {
            "m1": m.m1,
            "m2": m.m2,
            "inner2.m1": m.inner2.m1,
            "inner2.m2": m.inner2.m2,
        }
        if default:
            sol |= {"inner1.p1": m.inner1.p1, "inner1.p2": m.inner1.p2}
        res = {n: p for n, p in get_named_mstep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_unspecified_emparam(self, default):
        m = UnspecifiedEM()
        m.inner = UnspecifiedEM()

        sol = dict()
        if default:
            sol |= {
                "p1": m.p1,
                "p2": m.p2,
                "inner.p1": m.inner.p1,
                "inner.p2": m.inner.p2,
            }
        res = {n: p for n, p in get_named_mstep_params(m, default=default)}

        assert sol == res


class TestGetMStepParams:

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_improperly_inherited_mparam(self, default):
        m = ImproperlyInheritedEM()

        sol = {m.m1, m.m2}
        res = {*get_mstep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_eparam(self, default):
        m = SpecifiedE()
        m.inner = SpecifiedE()

        sol = set()
        if default:
            sol |= {m.p1, m.p2, m.inner.p1, m.inner.p2}
        res = {*get_mstep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_mparam(self, default):
        m = SpecifiedM()
        m.inner = SpecifiedM()

        sol = {m.m1, m.m2, m.inner.m1, m.inner.m2}
        res = {*get_mstep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_specified_emparam(self, default):
        m = SpecifiedEM()
        m.inner1 = SpecifiedE()
        m.inner2 = SpecifiedM()

        sol = {m.m1, m.m2, m.inner2.m1, m.inner2.m2}
        if default:
            sol |= {m.inner1.p1, m.inner1.p2}
        res = {*get_mstep_params(m, default=default)}

        assert sol == res

    @pytest.mark.parametrize(
        "default", (True, False), ids=("default=True", "default=False")
    )
    def test_unspecified_emparam(self, default):
        m = UnspecifiedEM()
        m.inner = UnspecifiedEM()

        sol = set()
        if default:
            sol |= {m.p1, m.p2, m.inner.p1, m.inner.p2}
        res = {*get_mstep_params(m, default=default)}

        assert sol == res
