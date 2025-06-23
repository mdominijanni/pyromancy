(manual-pc)=
# Fundamentals of Predictive Coding

## Background
One of the core problems in *deep learning*, machine learning where multiple layers of parameterized transformations are applied to the input data to generate the required output is the problem of *credit assignment*. The credit assignment problem refers to the manner in which we determine the contribution of a given parameter to the output of a model. In a shallow neural network, this task is trivial, but quickly becomes nontrivial for deep neural networks. The typical way this is performed with modern neural networks is *backpropagation* (often shortened to "backprop" or BP).

Backprop is a clever fusion of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) with [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) to perform credit assignment on deep neural networks with differentiable nonlinear activation functions. Gradient descent, along with its its many relatives, use the gradient of the loss function to optimize parameters of the network. With backprop, the gradient of the loss function with respect to the trainable parameters is computed, and stored, from the deepest layer to the shallowest. This eliminates redundant computation and optimizes the entire network simultaneously, as a single model.

Certain elements required for backprop however render it biologically implausible. For instance, backprop requires a network to maintain a backwards set of transposed connections to carry error signals, entirely detached from the forward set of connections for carry predictions. Additionally, backprop requires global synchronization over the entire network for forward and backward passes, a behavior we do not observe in the brain. Predictive coding provides an alternative to the credit assignment problem, one which uses local update rules to train a deep neural network.

## Learning with Predictive Coding


