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
To work with PyTorch's {py:class}`~torch.optim.Optimizer` class, the values of {py:class}`~pyromancy.nodes.Node` classes incorporate their trainable state using {py:class}`~torch.nn.Parameter` objects. In addition to having