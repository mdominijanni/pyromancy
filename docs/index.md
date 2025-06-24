# Pyromancy Documentation

```{toctree}
:hidden:

api-reference/index
user-manual/index
tutorials/index
```

Pyromancy is a compact library for predictive coding built on top of [PyTorch](https://github.com/pytorch/pytorch).

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
See the example, {ref}`tutorial-mnist-classifier-pcn`, for a complete worked-out example of how to build and train a predictive coding network.

## Site Navigation

### {ref}`api-reference`
Reference documentation for modules, classes, and functions.

### {ref}`user-manual`
Information on the basics of predictive coding and how to use Pyromancy.

### {ref}`tutorials`
Tutorials and examples written using the Pyromancy library.

## License

Pyromancy is distributed under the terms of the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
