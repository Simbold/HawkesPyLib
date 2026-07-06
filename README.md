[![Tests](https://github.com/Simbold/HawkesPyLib/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/Simbold/HawkesPyLib/actions/workflows/tests.yml)
[![Release (PyPI)](https://github.com/Simbold/HawkesPyLib/actions/workflows/build_master.yml/badge.svg)](https://github.com/Simbold/HawkesPyLib/actions/workflows/build_master.yml)
[![PyPI package](https://img.shields.io/pypi/v/HawkesPyLib?color=green&label=pypi%20package)](https://pypi.org/project/HawkesPyLib/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/HawkesPyLib)](https://pypi.org/project/HawkesPyLib/)
[![License: MIT](https://img.shields.io/pypi/l/HawkesPyLib?color=blue)](https://opensource.org/licenses/MIT)

# HawkesPyLib

A fast, lightweight Python library for **simulation and inference of univariate Hawkes
processes**. The core simulation and estimation routines are optimized by recursive
computation and JIT-compiled with [numba](https://numba.pydata.org/).

## Installation

```bash
pip install HawkesPyLib
```

Requires **Python ≥ 3.10**. Runtime dependencies (`numpy`, `scipy`, `numba`) are
installed automatically.

## Quickstart

```python
from HawkesPyLib.simulation import ApproxPowerlawHawkesProcessSimulation
from HawkesPyLib.inference import ApproxPowerlawHawkesProcessInference

# Simulate an approximate power-law Hawkes process
sim = ApproxPowerlawHawkesProcessSimulation("powlaw", mu=2.0, eta=0.5, alpha=0.4,
                                            tau0=0.05, m=5.0, M=5)
timestamps = sim.simulate(T=1000, seed=42)
print(f"{sim.n_jumps} events simulated")

# Recover the parameters by maximum likelihood
est = ApproxPowerlawHawkesProcessInference("powlaw", m=5.0, M=5)
mu, eta, alpha, tau0 = est.estimate_grid(timestamps, T=timestamps[-1], return_params=True)
print(f"mu={mu:.2f}, eta={eta:.2f}, alpha={alpha:.2f}, tau0={tau0:.2f}")
```

A fuller walk-through — plotting the conditional intensity and memory kernel, and fitting
under model misspecification — is in
[`Examples/SimpleExample.ipynb`](Examples/SimpleExample.ipynb).

## Documentation

Full API documentation is available at
[simbold.github.io/HawkesPyLib](https://simbold.github.io/HawkesPyLib/).

## Description

Hawkes processes are self-exciting point processes used to model or analyse event
arrivals. A univariate Hawkes process is defined through its conditional intensity
function:

$$ \lambda(t) = \mu + \sum_{t_i < t} g(t - t_i) $$

where $\mu$ is a constant background intensity and the memory kernel $g(t)$ specifies how
past arrivals excite the current intensity.

The following models are available, each with a **simulation** class (sampling via Ogata's
thinning algorithm) and an **inference** class (maximum likelihood estimation, plus the
corresponding compensator and conditional-intensity evaluation):

- Univariate Hawkes process with single exponential memory kernel
- Univariate Hawkes process with P-sum exponential memory kernel
- Univariate Hawkes process with approximate power-law memory kernel
- Univariate Hawkes process with approximate power-law memory kernel with smooth cutoff
- Homogeneous Poisson process

## License

HawkesPyLib is distributed under the terms of the
[MIT](https://opensource.org/licenses/MIT) license.
