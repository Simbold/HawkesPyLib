r"""
`HawkesPyLib` is a python package for simulation, and inference of Hawkes processes.

The source code of the package can be found on [github](https://github.com/Simbold/HawkesPyLib).

The package is structured into three main modules:

- `HawkesPyLib.simulation` for simulation of Hawkes processes

- `HawkesPyLib.inference` for fitting Hawkes processes

- `HawkesPyLib.processes` for analysing fully specified Hawkes processes

The module `HawkesPyLib.core` contains the functions underlying the simulation and fitting methods.

----------------------------------------------------------------------------------------------------------

### Notes

The inference module currently only supports model fitting using maximum likelihood estimation,
and is using scipy's L-BFGS-B algorithm as its optimization routine.

Hawkes process simulation is done via Ogata's modified thinning algorithm.
For more details on the simulation algorithm see algorithm 2 in (Ogata 1981).

The core simulation and estimation algorithms are optimized for speed
by recursively calculating the state of the process and further accelerated by using
[numba's](https://numba.pydata.org/) JIT compiler.

For more information on Hawkes processes, their simulation and inference find the references below.

    References:

- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point
    processes. Biometrika, 58(1):83 90.

- Ogata, Y. (1981). On lewis simulation method for point processes. IEEE transactions on information theory, 27(1):23 31.

- Ozaki, T. (1979). Maximum likelihood estimation of hawkes self-exciting point
    processes. Annals of the Institute of Statistical Mathematics, 31(1):145 155.

- C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines
         for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.

- Filimonov, V. and Sornette, D. (2015). Apparent criticality and calibration issues in
    the hawkes self-excited point process model: application to high-frequency finan-
    cial data. Quantitative Finance, 15(8):1293 1314.
"""
import importlib.metadata

__version__ = importlib.metadata.version("HawkesPyLib")
__pdoc__ = {}
__pdoc__["HawkesPyLib.simulation.ExpHawkesProcessSimulation.mu"] = False
__pdoc__["HawkesPyLib.simulation.ExpHawkesProcessSimulation.eta"] = False
__pdoc__["HawkesPyLib.simulation.ExpHawkesProcessSimulation.theta"] = False
__pdoc__["HawkesPyLib.simulation.ExpHawkesProcessSimulation.T"] = False

__pdoc__["HawkesPyLib.simulation.SumExpHawkesProcessSimulation.mu"] = False
__pdoc__["HawkesPyLib.simulation.SumExpHawkesProcessSimulation.eta"] = False
__pdoc__["HawkesPyLib.simulation.SumExpHawkesProcessSimulation.theta_vec"] = False
__pdoc__["HawkesPyLib.simulation.SumExpHawkesProcessSimulation.T"] = False

__pdoc__["HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.mu"] = False
__pdoc__["HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.eta"] = False
__pdoc__["HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.alpha"] = False
__pdoc__["HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.tau0"] = False
__pdoc__["HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.m"] = False
__pdoc__["HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.M"] = False
__pdoc__["HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.T"] = False
__pdoc__["HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.kernel"] = False

__pdoc__["HawkesPyLib.simulation.PoissonProcessSimulation.mu"] = False
__pdoc__["HawkesPyLib.simulation.PoissonProcessSimulation.T"] = False

__pdoc__["HawkesPyLib.util"] = False

__pdoc__["HawkesPyLib.inference.ExpHawkesProcessInference.set_params"] = False
__pdoc__["HawkesPyLib.inference.SumExpHawkesProcessInference.set_params"] = False
__pdoc__["HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.set_params"] = False
__pdoc__["HawkesPyLib.inference.ExpHawkesProcessInference.set_arrival_times"] = False
__pdoc__["HawkesPyLib.inference.SumExpHawkesProcessInference.set_arrival_times"] = False
__pdoc__["HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.set_arrival_times"] = False
__pdoc__["HawkesPyLib.inference.PoissonProcessInference.timestamps"] = False

__pdoc__["HawkesPyLib.processes.UnivariateHawkesProcess.mu"] = False
__pdoc__["HawkesPyLib.processes.UnivariateHawkesProcess.eta"] = False
__pdoc__["HawkesPyLib.processes.UnivariateHawkesProcess.theta"] = False
__pdoc__["HawkesPyLib.processes.UnivariateHawkesProcess.timestamps"] = False
__pdoc__["HawkesPyLib.processes.UnivariateHawkesProcess.theta_vec"] = False
__pdoc__["HawkesPyLib.processes.UnivariateHawkesProcess.alpha"] = False
__pdoc__["HawkesPyLib.processes.UnivariateHawkesProcess.tau0"] = False
