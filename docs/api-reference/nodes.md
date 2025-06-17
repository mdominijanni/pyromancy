# pyromancy.nodes

```{eval-rst}
.. currentmodule:: pyromancy.nodes
```

## Base Classes
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    Node
    PredictiveNode
    VariationalNode
```

## Gaussian Nodes
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    AbstractGaussianNode
    StandardGaussianNode
    IsotropicGaussianNode
    FactorizedGaussianNode
    MultivariateGaussianNode
```

## Special Nodes
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    BiasNode
    FixedNode
```

## Variant Nodes

Nodes that have the same functionality as the "standard" nodes, but with alternative
implementations.

```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: generated

    LDLMultivariateGaussianNode
```