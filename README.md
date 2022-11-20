![Tests](https://github.com/Simbold/HawkesPyLib/actions/workflows/tests.yml/badge.svg?branch=master)
![Build](https://github.com/Simbold/HawkesPyLib/actions/workflows/build_master.yml/badge.svg?branch=master)
[![PyPI package](https://img.shields.io/pypi/v/HawkesPyLib?color=green&label=pypi%20package)](https://pypi.org/project/HawkesPyLib/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/HawkesPyLib)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/pypi/l/HawkesPyLib?color=blue)](https://opensource.org/licenses/MIT)

# HawkesPyLib
A simple Python Package for simulation and inference of Hawkes processes. The library is currently under active development. More methods and functionality will be introduced shortly.

## Installation

    $ pip install HawkesPyLib

## Documentation

A detailed description of the package can be found in the [documentation](https://simbold.github.io/HawkesPyLib/).

## Description
The library allows for simulation and fitting of Hawkes processes. Hawkes processes are self-exciting point processes and can be used to model or analyse event arrivals. Hawkes processes can be defined in terms of the conditional intensity function:

$$ \lambda(t) = \mu + \sum_{t_i < t} g(t-t_i) $$

where $\mu$ is a constant background intensity and the memory kernel function $g(t)$ specifies how past event arrivals influence the current state of the process. 

The following Hawkes process models are currently available:
- Univariate Hawkes process with single exponential memory kernel
- Univariate Hawkes process with P-sum exponential memory kernel
- Univariate Hawkes process with approximate power-law memory kernel
- Univariate Hawkes process with approximate power-law memory kernel with smooth cutoff
- Homogenous Poisson process

For each of the models there is a simulator class for generating Hawkes process samples using Ogata's thining algorithm.
The estimator class allows for maximum likelihood estimation of the model as well as the calculation of the corresponding compensator, and evaluation of the conditional intensity function.

A quick example of simulating and estimating Hawkes processes can be found in the Examples folder.

The core simulation and estimation algorithms are optimized for speed
by recursive calculation of the model and further accelerated by
[numba's](https://numba.pydata.org/) JIT compiler.

## License

HawkesPyLib is distributed under the terms of the [MIT](https://opensource.org/licenses/MIT) license.
