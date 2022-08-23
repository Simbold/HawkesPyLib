[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# HawkesPyLib
A simple Python Library for simulation and inference of univariate Hawkes processes. The library is currently under active development. More methods and functionality will be introduced shortly.

A quick example of simulating and estimating a Hawkes process can be found in the Example.ipynb notebook.

The following Hawkes process models are currently available:
- Univariate Hawkes process with single exponential kernel
- Univariate Hawkes process with P-sum exponential kernel
- Univariate Hawkes process with approximate power-law kernel
- Univariate Hawkes process with approximate power-law with short time cutoff kernel

For each of the models there is a simulator class for generating Hawkes process samples using Ogata's thining algorithm.
The estimator class allows for maximum likelihood estimation of the model as well as the calculation of the corresponding compensator.

## Installation
`numba` (>=0.55) and `numpy` (>=1.21) and `scipy` and `matplotlib` must be installed for the library to function.