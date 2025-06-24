![Pyromancy Header](misc/assets/pyromancy-github-header.png)

[![PyPI - Version](https://img.shields.io/pypi/v/pyromancy-ai.svg)](https://pypi.org/project/pyromancy-ai)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyromancy-ai.svg)](https://pypi.org/project/pyromancy-ai)

-----

## About
Pyromancy is a compact library for [predictive coding](https://arxiv.org/abs/2407.04117), implemented using [PyTorch](https://github.com/pytorch/pytorch). It takes a minimal approach, providing the core components for building and training predictive coding networks.

## Installation
Pyromancy is available as a package on PyPI and can be installed as follows.

```bash
pip install pyromancy-ai
```

By default, this installs the `torch` and `torchvision` packages with *only* CPU support (Linux/Windows) or support for CPU and MPS (macOS). To include support for CUDA or ROCm, a corresponding ``extra-index-url`` must be specified.

```bash
pip install pyromancy-ai --extra-index-url https://download.pytorch.org/whl/cu128
```

Installing with this command includes support for CPU and for CUDA 12.8. The installation options can be found on PyTorch's [getting started](https://pytorch.org/get-started/locally/) page.

## Getting Started
See the example, [Classifying MNIST with a Hierarchical PCN](https://pyromancy.ai/en/latest/tutorials/mnist-classifier-pcn.html), for a complete worked-out example of how to build and train a predictive coding network.

## License
Pyromancy is distributed under the terms of the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
