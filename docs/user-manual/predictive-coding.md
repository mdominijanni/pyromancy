(manual-pc)=
# Fundamentals of Predictive Coding

## Background
One of the core problems in *deep learning*, machine learning where multiple layers of parameterized transformations are applied to the input data to generate the required output is the problem of *credit assignment*. The credit assignment problem refers to the manner in which we determine the contribution of a given parameter to the output of a model. In a shallow neural network, this task is trivial, but quickly becomes nontrivial for deep neural networks. The typical way this is performed with modern neural networks is *backpropagation* (often shortened to "backprop" or BP).

Backprop is a clever fusion of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) with [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) to perform credit assignment on deep neural networks with differentiable nonlinear activation functions. Gradient descent, along with its many relatives, use the gradient of the loss function to optimize parameters of the network. With backprop, the gradient of the loss function with respect to the trainable parameters is computed, and stored, from the deepest layer to the shallowest. This eliminates redundant computation and optimizes the entire network simultaneously, as a single model.

Certain elements required for backprop however render it biologically implausible. For instance, backprop requires a network to maintain a backwards set of transposed connections to carry error signals, entirely detached from the forward set of connections for carry predictions. Additionally, backprop requires global synchronization over the entire network for forward and backward passes, a behavior we do not observe in the brain. Predictive coding provides an alternative to the credit assignment problem, one which uses local update rules to train a deep neural network.

## Learning with Predictive Coding
Unlike backprop, which treats the entire network as a single model to optimize. Predictive coding breaks this down by describing the global objective as a sum of local objectives. Then, gradient-based updates to the parameters and states of a predictive coding network (PCN) are based only on local interactions, while still performing global optimization. In PCNs, quantity being minimizing is the *variational free energy*.

We assume that each node is modelling a variational distribution, and its local contribution to the free energy of the network is based on the log probability of the node's state and the prediction it receives. For a multivariate Gaussian distribution, with node state $\mathbf{x}$, prediction $\boldsymbol{\mu}$, and learned covariances $\boldsymbol{\Sigma}$, we use the following as the free energy.

$$\mathcal{F} = \frac{1}{2} \left(
(\mathbf{z} - \boldsymbol{\mu})
\boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})^\intercal
+ \log \lvert\boldsymbol{\Sigma}\rvert \right) + C$$

Here, the term $C$ is constant and left out of the energy calculations and optimization. Note that when $\boldsymbol{\Sigma} = \mathbf{I}$, then this reduces to $\mathcal{F} = \frac{1}{2} \lVert \mathbf{x} - \boldsymbol{\mu} \rVert_2^2$.

## Constructing a PCN with Pyromancy
Hierarchical PCNs take on a very similar form to feedforward neural networks (FNNs). Pyromancy uses the base class {py:class}`~pyromancy.nodes.Node` to represent these node states, and the variational distribution they are imposing. Normal trainable transformations (e.g. {py:class}`~torch.nn.Linear`, {py:class}`~torch.nn.Conv2d`) can then be used as edges between them, along with a nonlinear activation function. For a full working example, see the example: {ref}`tutorial-mnist-classifier-pcn`.

## Node Parameters and Shapes
One added piece of complexity is when nodes have a parameter that is dependent upon the size of the value they're modelling. For example, consider the Gaussian example where $\boldsymbol{\Sigma} = \operatorname{diag}(\sigma_1, \sigma_2, \ldots, \sigma_N)$. With convolutional layers, the exact number of outputs might change between batches (based on different image sizes).

Pyromancy solves this by allowing for placeholder dimensions with {py:class}`~pyromancy.Shape` objects, for example, the batch dimension is automatically added as a placeholder when initializing {py:class}`~pyromancy.nodes.Node` objects. For causes such as the above (from {py:class}`~pyromancy.nodes.FactorizedGaussianNode`), any additional placeholders are treated as batch dimensions for the necessary operations. For example, if the node is constructed as `FactorizedGaussianNode(32, None, None)`, then 32 separate variances will be learned, and each of those would be shared across all elements of its corresponding each channel.