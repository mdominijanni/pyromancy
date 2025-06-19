![Pyromancy Header](misc/assets/pyromancy-github-header.png)

[![PyPI - Version](https://img.shields.io/pypi/v/pyromancy-ai.svg)](https://pypi.org/project/pyromancy-ai)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyromancy-ai.svg)](https://pypi.org/project/pyromancy-ai)

-----

## About

Pyromancy is a compact library for [predictive coding](https://arxiv.org/abs/2407.04117), implemented using [PyTorch](https://github.com/pytorch/pytorch). It takes a minimal approach, providing the core components for building and training predictive coding networks.

## Installation
Pyromancy is available as a package on PyPI and can be installed without PyTorch as follows.

```console
pip install pyromancy-ai
```

Installation including PyTorch (a mandatory dependency) can be specified by using the `torch` feature.

```console
pip install pyromancy-ai[torch]
```

By default, this installs the `torch`, `torchvision`, and `torchaudio` packages with *only* CPU support (Linux/Windows) or support for CPU and MPS (macOS). To include support for CUDA or ROCm, a corresponding ``index-url`` must be specified.

```console
pip install pyromancy-ai[torch] --index-url https://download.pytorch.org/whl/cu128
```

Installing with this command includes support for CPU and for CUDA 12.8. All installation options can be found on PyTorch's [getting started](https://pytorch.org/get-started/locally/) page.

## License

Pyromancy is distributed under the terms of the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
