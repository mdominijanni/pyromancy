(manual-em)=
# Working with Expectation–Maximization

## Background
Unlike the traditional backprop training regimens used for feedforward neural networks, predictive coding networks are trained using expectation–maximization (EM) based procedures (typically referred to as either *inference learning* or *prospective configuration*).

The training objective with predictive coding is to minimize the *variational free energy*, $\mathcal{F}$, of the network. In a predictive coding network, the total free energy is equal to the sum of free energies for each node in the network. In general, we can describe this as a function relative to the state of a node $\mathbf{x}$ and parameterized by $\boldsymbol{\Theta}$.

Then, EM breaks the optimization task into two steps:
- E-Step (Inference): $\mathbf{x}^\ast = \operatorname{arg\,max}_{\mathbf{x}} \mathcal{F}(\mathbf{x}, \boldsymbol{\Theta})$
- M-Step (Learning): $\boldsymbol{\Theta}^\ast = \operatorname{arg\,max}_{\boldsymbol{\Theta}} \mathcal{F}(\mathbf{x}, \boldsymbol{\Theta})$

In standard inference learning (IL), for each batch, multiple E-steps are repeatedly performed, followed by a single M-step. In incremental IL, for each batch, each E-step is followed by an M-step, with multiple iterations being performed.

## EM in Pyromancy
To work with PyTorch's {py:class}`~torch.optim.Optimizer` class, the values of {py:class}`~pyromancy.nodes.Node` classes incorporate their trainable state using {py:class}`~torch.nn.Parameter` objects. In addition to this, model parameters from classes not included in Pyromancy need to be incorporated into this scheme.

To this end, Pyromancy defines some helper functions for managing these two kinds of parameters. First, it provides two decorator functions to register parameters as either E-step or M-step parameters: {py:func}`~pyromancy.eparameters` and {py:func}`~pyromancy.mparameters` respectively. These add attributes `_e_params_` and `_m_params_`, respectively to the class, and fill them with any added parameter names, plus any E-step or M-step parameters in the superclasses (resolved by traversing superclasses in the method resolution order with {py:attr}`~object.__mro__`).

For example, in the following inheritance chain, {py:class}`~pyromancy.nodes.MultivariateGaussianNode` has `value` as an E-step parameter and `covar_cf_logdiag` and `covar_cf_offtril` as M-step parameters.

```python
@eparameters("value")
class PredictiveNode(Node, ABC):

    value: nn.Parameter

    def __init__(self, ...) -> None:
        ...


class VariationalNode(PredictiveNode, ABC):

    def __init__(self, ...) -> None:
        ...

class AbstractGaussianNode(VariationalNode, ABC):

    def __init__(self, ...) -> None:
        ...

@mparameters("covar_cf_logdiag", "covar_cf_offtril")
class MultivariateGaussianNode(AbstractGaussianNode):

    covar_cf_logdiag: nn.Parameter
    covar_cf_offtril: nn.Parameter

    def __init__(self, ...) -> None:
        ...
```

Then, Pyromancy provides the functions {py:func}`~pyromancy.get_named_estep_params` and {py:func}`~pyromancy.get_estep_params` to retrieve E-step parameters, and the functions {py:func}`~pyromancy.get_named_mstep_params` and {py:func}`~pyromancy.get_mstep_params` to retrieve M-step parameters. These are modelled after the {py:meth}`~torch.nn.Module.named_parameters` and {py:meth}`~torch.nn.Module.parameters` methods provided by PyTorch's {py:class}`~torch.nn.Module` object. These methods also traverse the entire method resolution order for classes so even if E-step and/or M-step parameters are not directly registered for a class, they will still be detected.

By default, if a class and none of its parent classes specify E-step or M-step parameters, any parameters found will be assumed to be M-step parameters. For example, for the following module, the named E-step parameters will be `1.value` and the named M-step parameters will be `0.weight`, `0.bias`, and `1.logvar`.

```python
module = nn.ModuleList(nn.Linear(784, 256), FactorizedGaussianNode(256))
```