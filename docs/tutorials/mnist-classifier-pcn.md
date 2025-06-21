(tutorial-mnist-classifier-pcn)=
# Classifying MNIST with a Hierarchical PCN

In this example, we construct a simple predictive coding network (PCN) for classifying the MNIST dataset.

```{tip}
Download this example as a [Jupyter notebook](../_static/notebooks/mnist-classifier-pcn.ipynb).
```

## Setting Up the Notebook

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets
from torchvision.transforms import v2
from tqdm.notebook import tqdm

import pyromancy as pyro
from pyromancy.nodes import StandardGaussianNode
```

In addition to the import statements for external libraries, we import `pyromancy` as the shorthand `pyro` and a node class: {py:class}`~pyromancy.nodes.StandardGaussianNode`.

Next, we need to configure the compute device on which operations are performed, and the datatype of tensors to use.

```python
device: str = "auto"
dtype: torch.dtype = torch.float32

if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

assert torch.empty([], device=device, dtype=dtype).is_floating_point()

iscpu = device.partition(":")[0].lower() == "cpu"
iscuda = device.partition(":")[0].lower() == "cuda"
ismps = device.partition(":")[0].lower() == "mps"

print(f"using {device} with {dtype} tensors")
```

Then, we use TorchVision to fetch the dataset, convert the byte tensors to floating-point, and rescale the values between 0 and 1.

```python
train_set = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(dtype, scale=True)]),
)

test_set = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(dtype, scale=True)]),
)
```

## Defining the Model

After we set up the notebook, we need to define the PCN model we're using. A hierarchical PCN is structured very similarly to a feedforward neural network (FNN), and inference is performed in the same way as with an FNN.

However, unlike with an FNN, a PCN is split into two major parts: *nodes* and *edges*. The nodes define the *states* of the model: input, latent, and output, as well as how the *energy* is computed for those states. The edges define the parameterized functions to predict the value of one node, given the value of another.

```{note}
There is a slight difference in convention between PCNs and FNNs. With nonlinearity $f$, the output $\boldsymbol{\mu}$ of a layer for an FNN is usually defined as:

$$\boldsymbol{\mu} = f(\mathbf{W} \mathbf{z} + \mathbf{b})$$

whereas for a PCN it is defined as:

$$\boldsymbol{\mu} = \mathbf{W} f(\mathbf{z}) + \mathbf{b}$$

where $\mathbf{z}$ is the input to the layer, and $\mathbf{W}$ and $\mathbf{b}$ are the trainable weights and biases, respectively.
```

We'll define a PCN with four nodes, of sizes 784 (the input), 256 (the first latent state), 256 (the second latent state), and 10 (the output). There are three edges connecting these, using {py:class}`~torch.nn.Linear` to model the trainable affine transformation with {py:class}`~torch.nn.ReLU` as the nonlinearity.

Just like the corresponding FNN, this model has 268,800 weight parameters and 522 bias parameters.

```python
class PCN(nn.Module):

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.nodes = nn.ModuleList(StandardGaussianNode(n) for n in (784, 256, 256, 10))
        self.edges = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(), nn.Linear(self.nodes[ell].size, self.nodes[ell + 1].size)
            )
            for ell in range(len(self.nodes) - 1)
        )

    def reset(self) -> None:
        self.zero_grad()
        for node in self.nodes:
            node.reset()

    @torch.no_grad()
    def init_x(self, x: torch.Tensor) -> None:
        self.reset()
        z = self.nodes[0].init(x)
        for node, edge in zip(self.nodes[1:], self.edges):
            z = node.init(edge(z))

    @torch.no_grad()
    def init_xy(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.reset()
        z = self.nodes[0].init(x)
        for node, edge in zip(self.nodes[1:-1], self.edges[:-1]):
            z = node.init(edge(z))
        _ = self.nodes[-1].init(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x
        for edge in self.edges:
            mu = edge(mu)
        return mu

    def energy(self) -> torch.Tensor:
        vfe = self.nodes[0].value.new_zeros(self.nodes[0].value.size(0))

        mu = self.nodes[0].value
        for node, edge in zip(self.nodes[1:], self.edges):
            mu = edge(mu)
            vfe.add_(node.energy(mu))

        return vfe


pcn = PCN().to(dtype=dtype, device=device)
```

We also defined a few methods for convenience:
- `reset()` clears the transient state of the nodes.
- `init_x()` initializes the values of the input node to `x`, and initializes the others using the edges to generate predictions.
- `init_xy()` does the same as `init_x()`, but also fixes the output node to `y`.
- `forward()` performs inference in a feedforward manner, just like with an FNN.
- `energy()` computes the energy of the network, proportional to the sum of squared errors between the states of nodes and the predictions of those states.

## Configuring the Training Procedure

Unlike with FNNs trained with backprop, where a loss $\mathcal{L}$ is computed between the network outputs and the target, PCNs are trained to minimize the *variational free energy*. Although the specific calculation for this depends on the variational distribution assumed by a given `Node`, `StandardGaussianNode` assumes a Gaussian distribution with unit variance, i.e. $\mathcal{N}(\boldsymbol{\mu}, \mathbf{I})$. The free energy for a node $\ell$ with a state $z$ is then computed relative to the prediction $\boldsymbol{\mu}$.

$$f(\mathbf{z}; \boldsymbol{\mu}) = \lVert \mathbf{z} - \boldsymbol{\mu} \rVert_2^2$$

The energy of the entire network is the sum of the individual energy terms.

$$\mathcal{F} = \sum_\ell f(\mathbf{z}_\ell; \boldsymbol{\mu}_\ell)$$

This total energy is the quantity minimized by training. The *inference learning* procedure for PCNs is a type of *expectation maximization* (EM). This divides the process in two: E-steps are repeatedly performed to compute states of the network that aren't fixed (the $\mathbf{z}$ terms), then M-steps perform an update to the trainable parameters of the network.

```python
epochs: int = 10
batch_size: int = 500
num_esteps: int = 32
nbatches = len(train_set) // batch_size

e_opt = optim.SGD(
    pyro.get_estep_params(pcn, exclude=(pcn.nodes[0], pcn.nodes[-1])), lr=0.2
)
m_opt = optim.Adam(pyro.get_mstep_params(pcn), lr=0.001)
```

Here we set up the training procedure to run for 10 epochs, where 32 E-steps are performed for each batch of 500, then the trainable parameters are updated with a single M-step. The functions {py:func}`~pyromancy.get_estep_params` and {py:func}`~pyromancy.get_mstep_params` are used to separate which parameters should be updated on which type of step (by default, if a {py:class}`~torch.nn.Module` doesn't specify these, the parameters are assumed to be updated on M-steps). Additionally, since we want to fix the values of the input and output node during training, we use the `exclude` argument to leave those out when retrieving the E-step parameters.

## Training/Testing Loop

Finally, we create the training/testing loop over the dataset. Unlike for an FNN, the training procedure is broken down into the following steps:
- Initialize states of the network with the input, output, and *predictions* of latent states.
- Repeatedly perform E-steps to refine the (unfixed) node state to reduce the network's energy.
- Perform an M-step to refine *how* the predictions are generated.

```python
accs = []

for _ in tqdm(range(epochs), desc="Epoch", initial=0, total=epochs, position=0):
    # set training mode
    pcn.train()

    # load and sample training set
    sampler = RandomSampler(
        train_set,
        replacement=False,
    )
    loader = DataLoader(
        train_set,
        batch_size,
        sampler=sampler,
        drop_last=True,
        pin_memory=iscuda,
        pin_memory_device="" if not iscuda else device,
    )

    # training loop
    for x, y in tqdm(
        loader, desc="Batch", initial=0, total=nbatches, leave=False, position=1
    ):
        # prepare data
        x = x.to(device=device).flatten(1)
        y = y.to(device=device)

        # initialize pcn with data
        pcn.init_xy(x, F.one_hot(y, 10).to(dtype=dtype))

        # perform E-steps
        for _ in range(num_esteps):
            pcn.zero_grad()
            pcn.energy().mean().backward(inputs=e_opt.param_groups[0]["params"])
            e_opt.step()

        # perform M-step
        pcn.zero_grad()
        pcn.energy().mean().backward(inputs=m_opt.param_groups[0]["params"])
        m_opt.step()

    # set inference mode
    pcn.eval()

    # load testing set
    loader = DataLoader(
        test_set,
        len(test_set),
        shuffle=False,
        pin_memory=iscuda,
        pin_memory_device="" if not iscuda else device,
    )
    x, y = next(iter(loader))

    # prepare data
    x = x.to(device=device).flatten(1)
    y = y.to(device=device)

    # forward inference
    ypred = pcn(x)
    accs.append((y == ypred.argmax(1)).float().mean().item())

# print results
print("Epoch    Accuracy")
for e, acc in enumerate(accs, 1):
    print(f"{e:>5}    {f'{acc:.5f}':<8}")
```