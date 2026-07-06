# Graph Report - .  (2026-07-06)

## Corpus Check
- Corpus is ~22,853 words - fits in a single context window. You may not need a graph.

## Summary
- 582 nodes · 919 edges · 47 communities (42 shown, 5 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 155 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_MLE Inference & Poisson|MLE Inference & Poisson]]
- [[_COMMUNITY_Intensity Functions|Intensity Functions]]
- [[_COMMUNITY_Log-Likelihood Computation|Log-Likelihood Computation]]
- [[_COMMUNITY_Process Base Class|Process Base Class]]
- [[_COMMUNITY_Parameter Validators|Parameter Validators]]
- [[_COMMUNITY_Poisson Process Simulation|Poisson Process Simulation]]
- [[_COMMUNITY_Concepts & Documentation|Concepts & Documentation]]
- [[_COMMUNITY_Exponential Hawkes Simulation|Exponential Hawkes Simulation]]
- [[_COMMUNITY_ApproxPowlaw Estimation Tests|ApproxPowlaw Estimation Tests]]
- [[_COMMUNITY_ExpSumExp Simulation Methods|Exp/SumExp Simulation Methods]]
- [[_COMMUNITY_Sum-Exp Simulation|Sum-Exp Simulation]]
- [[_COMMUNITY_Core Simulation Algorithms|Core Simulation Algorithms]]
- [[_COMMUNITY_Memory Kernel Functions|Memory Kernel Functions]]
- [[_COMMUNITY_Kernel Tests|Kernel Tests]]
- [[_COMMUNITY_ApproxPowlaw Grid Estimation Tests|ApproxPowlaw Grid Estimation Tests]]
- [[_COMMUNITY_Exponential Estimation Tests|Exponential Estimation Tests]]
- [[_COMMUNITY_Sum-Exp Estimation Tests|Sum-Exp Estimation Tests]]
- [[_COMMUNITY_Compensator Functions|Compensator Functions]]
- [[_COMMUNITY_ApproxPowlaw GetterLogL Tests|ApproxPowlaw Getter/LogL Tests]]
- [[_COMMUNITY_Expo Grid Estimation Tests|Expo Grid Estimation Tests]]
- [[_COMMUNITY_Sum-Exp Grid Estimation Tests|Sum-Exp Grid Estimation Tests]]
- [[_COMMUNITY_Sum-Exp Inference Class|Sum-Exp Inference Class]]
- [[_COMMUNITY_ApproxPowlaw Intensity Tests|ApproxPowlaw Intensity Tests]]
- [[_COMMUNITY_Expo Intensity Tests|Expo Intensity Tests]]
- [[_COMMUNITY_PowlawCutoff Intensity Tests|PowlawCutoff Intensity Tests]]
- [[_COMMUNITY_ApproxPowlaw Inference Class|ApproxPowlaw Inference Class]]
- [[_COMMUNITY_Exponential Inference Class|Exponential Inference Class]]
- [[_COMMUNITY_Expo Kernel-Value Tests|Expo Kernel-Value Tests]]
- [[_COMMUNITY_Sum-Exp Compensator Tests|Sum-Exp Compensator Tests]]
- [[_COMMUNITY_Sum-Exp Intensity Tests|Sum-Exp Intensity Tests]]
- [[_COMMUNITY_ApproxPowlaw Compensator Tests|ApproxPowlaw Compensator Tests]]
- [[_COMMUNITY_ApproxPowlaw Intensity Method Tests|ApproxPowlaw Intensity Method Tests]]
- [[_COMMUNITY_ApproxPowlaw Kernel-Value Tests|ApproxPowlaw Kernel-Value Tests]]
- [[_COMMUNITY_Expo Compensator Tests|Expo Compensator Tests]]
- [[_COMMUNITY_Expo Intensity Method Tests|Expo Intensity Method Tests]]
- [[_COMMUNITY_Expo Compute-LogL Tests|Expo Compute-LogL Tests]]
- [[_COMMUNITY_Sum-Exp Intensity Method Tests|Sum-Exp Intensity Method Tests]]
- [[_COMMUNITY_Sum-Exp Kernel-Value Tests|Sum-Exp Kernel-Value Tests]]
- [[_COMMUNITY_Sum-Exp Compute-LogL Tests|Sum-Exp Compute-LogL Tests]]
- [[_COMMUNITY_CI & Publishing Pipeline|CI & Publishing Pipeline]]
- [[_COMMUNITY_Integer Range Validator|Integer Range Validator]]
- [[_COMMUNITY_Expo Getter Tests|Expo Getter Tests]]
- [[_COMMUNITY_Docs Build Script|Docs Build Script]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Ruff Linting|Ruff Linting]]

## God Nodes (most connected - your core abstractions)
1. `ApproxPowerlawHawkesProcessInference` - 39 edges
2. `ExpHawkesProcessInference` - 36 edges
3. `SumExpHawkesProcessInference` - 36 edges
4. `UnivariateHawkesProcess` - 32 edges
5. `ApproxPowerlawHawkesProcessSimulation` - 20 edges
6. `generate_eval_grid()` - 19 edges
7. `ndarray` - 18 edges
8. `ExpHawkesProcessSimulation` - 17 edges
9. `SumExpHawkesProcessSimulation` - 17 edges
10. `OneOf` - 17 edges

## Surprising Connections (you probably didn't know these)
- `TestExpo_compensator` --uses--> `ExpHawkesProcessInference`  [INFERRED]
  tests/test_estim_expo.py → src/HawkesPyLib/inference.py
- `TestExpo_compute_logL` --uses--> `ExpHawkesProcessInference`  [INFERRED]
  tests/test_estim_expo.py → src/HawkesPyLib/inference.py
- `TestExpo_estimate` --uses--> `ExpHawkesProcessInference`  [INFERRED]
  tests/test_estim_expo.py → src/HawkesPyLib/inference.py
- `TestExpo_estimate_grid` --uses--> `ExpHawkesProcessInference`  [INFERRED]
  tests/test_estim_expo.py → src/HawkesPyLib/inference.py
- `TestExpo_intensity` --uses--> `ExpHawkesProcessInference`  [INFERRED]
  tests/test_estim_expo.py → src/HawkesPyLib/inference.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Hawkes Process Memory Kernel Models** — readme_single_exponential_kernel, readme_psum_exponential_kernel, readme_approx_powerlaw_kernel, readme_approx_powerlaw_smooth_cutoff_kernel [EXTRACTED 1.00]
- **CI Testing & Publishing Pipeline** — tests_workflow, build_master_workflow, build_develop_workflow, claude_tox [INFERRED 0.75]

## Communities (47 total, 5 thin omitted)

### Community 0 - "MLE Inference & Poisson"
Cohesion: 0.06
Nodes (34): PoissonProcessInference, Estimates process parameters using maximum likelihood estimation.          Args:, # TODO: Improve the standard starting values: eta0 using the model-free branchin, Provides maximum liklihood estimation for the single exponential kernel., Estimates the Hawkes process parameters using maximum likelihood estimation., # TODO: Improve the standard starting values: eta0 using the model-free branchin, Estimates the Hawkes process parameters using maximum likelihood estimation., # TODO: Improve the standard starting values: eta0 using the model-free branchin (+26 more)

### Community 1 - "Intensity Functions"
Cohesion: 0.07
Nodes (26): generate_eval_grid(), Evaluation of the intensity function of a univariate Hawkes process         with, Evaluation of the intensity function of a univariate Hawkes process with approxi, Evaluation of the intensity function of a univariate Hawkes process with single, Generates an equidistant grid in the closed interval [0, T]     with step size g, Evaluation of the intensity function of a univariate Hawkes process         with, uvhp_approx_powl_cutoff_intensity(), uvhp_approx_powl_intensity() (+18 more)

### Community 2 - "Log-Likelihood Computation"
Cohesion: 0.07
Nodes (23): Log-likelihood function for a Hawkes Process with P-sum exponential kernel, Gradient of the log-likelihood function for a Hawkes Process with P-sum exponent, Log-likelihood function for a Hawkes Process with approximate power-law kernel w, Log-likelihood function for a Hawkes Process with approximate power-law kernel w, Gradient of the log-likelihood function for a Hawkes Process with single exponen, Log-likelihood function for a Hawkes Process with single exponential kernel, uvhp_approx_powl_cut_logL(), uvhp_approx_powl_logL() (+15 more)

### Community 3 - "Process Base Class"
Cohesion: 0.09
Nodes (13): r"""Set Hawkes process model parameters.             Arguments `mu` and `eta` ar, Set arrival times of the process.          Args:             timestamps (np.ndar, Returns the model parameters:          Returns:             tuple: Tuple of para, Checks if input 'T' is valid          Args:             T (float): End time of t, r""" Implements multiple univariate Hawkes process models and         allows for, r"""         Args:             kernel (str): Type of Hawkes process memory kerne, UnivariateHawkesProcess, Test the ApproxPowerlawProcess Class. (+5 more)

### Community 4 - "Parameter Validators"
Cohesion: 0.11
Nodes (12): ABC, FloatInExRange, OneOf, PositiveFloatNdarray, PositiveOrderedFloatNdarray, Validator class that validates if given array contains only positive floats and, Validator Class which validates if string is part of multiple options, Validator class that validates if a given value is of type float or int and in t (+4 more)

### Community 5 - "Poisson Process Simulation"
Cohesion: 0.12
Nodes (11): PoissonProcessSimulation, # TODO: Add simulation options: max_n, multiple paths, branching sampler and con, r""" Class for simulation a homogenous Poisson process with rate parameter `mu`., # TODO: Add simulation options: max_n, multiple paths, branching sampler and con, Args:             mu (float): Constant intensity rate parameter, mu > 0., Generates a realization of the specified homogenous Poisson process.          Ar, # TODO: Add simulation options: max_n, multiple paths, branching sampler and con, Test calls correct simulator (+3 more)

### Community 6 - "Concepts & Documentation"
Cohesion: 0.12
Nodes (18): core module, inference module, Explicit numba @njit Type Signatures, processes module, simulation module, Approximate Power-law Memory Kernel, Approximate Power-law Memory Kernel with Smooth Cutoff, Compensator (+10 more)

### Community 7 - "Exponential Hawkes Simulation"
Cohesion: 0.15
Nodes (9): ExpHawkesProcessSimulation, r""" Class for simulation of univariate Hawkes processes with single exponential, r""" To initlize the class provide the following parameters that specify the sin, Check if non positive mu and non float mu raise error, Check if values for eta outside the interval (0,1) are refused, Check if non positive theta and non float theta raise error, Check if non positive T and non float int raise error, Check if intensity function called correct and with the correct parameters (+1 more)

### Community 8 - "ApproxPowlaw Estimation Tests"
Cohesion: 0.12
Nodes (9): Test the ApproxPowerlawInference Class., Test if .estimate() calls correct mle with custom param_vec0, Test if .estimate() calls correct mle with custom param_vec0, Test if .estimate() calls correct mle with custom param_vec0, Test if .estimate() calls correct mle with custom param_vec0, Check if invalid timestamp input raises error, Check if invalid T input raises error, Check if all attributes set after succeseful estimation (+1 more)

### Community 9 - "Exp/SumExp Simulation Methods"
Cohesion: 0.13
Nodes (8): r""" Evaluates the intensity function \(\lambda(t)\) on a grid of equidistant ti, r""" Returns the value of the memory kernel at given time values.             Th, r""" To initlize the class provide the following parameters that specify the P-s, Generates a realization of the specified Hawkes process.          Args:, r""" Evaluates the intensity function \(\lambda(t)\) on a grid of equidistant ti, r""" Returns the value of the memory kernel at given time values. The memory ker, Generates a realization of the specified Hawkes process.          Args:, ndarray

### Community 10 - "Sum-Exp Simulation"
Cohesion: 0.18
Nodes (8): r""" Class for simulation of univariate Hawkes processes with P-sum exponential, SumExpHawkesProcessSimulation, Check if intensity function called correct and the correct with correct params, Check if non positive mu and non float mu raise error, Check if values for eta outside the interval (0,1) are refused, Check if non positive theta and non float theta raise error, Check if non positive T and non float int raise error, TestSumExpHawkesSimulation

### Community 11 - "Core Simulation Algorithms"
Cohesion: 0.20
Nodes (12): homogenous_poisson_simulator(), Simulates a Hawkes process with single exponential memory kernel.         Implem, Simulates a Hawkes process with approximate power-law memory kernel.         Imp, Simulates a Hawkes process with P-sum expoential memory kernel.         Implemen, Simulates a Hawkes process with approximate power-law memory kernel.         Imp, Simulates a homogenous Poisson process with constant intensity mu.      Args:, uvhp_approx_powl_cutoff_simulator(), uvhp_approx_powl_simulator() (+4 more)

### Community 12 - "Memory Kernel Functions"
Cohesion: 0.22
Nodes (9): Computes values of the Approximate power-law memory kernel with smooth cutoff co, Computes values of the single exponential Hawkes process memory kernel.      Arg, Computes values of the Approximate power-law memory kernel.      Args:         t, uvhp_approx_powl_cutoff_kernel(), uvhp_approx_powl_kernel(), uvhp_expo_kernel(), r""" Returns the value of the memory kernel at given time values.          The s, r""" Returns the value of the memory kernel at given time values. The memory ker (+1 more)

### Community 13 - "Kernel Tests"
Cohesion: 0.17
Nodes (7): Computes values of the P-sum exponential Hawkes process memory kernel.      Args, uvhp_sum_expo_kernel(), test that sum expo with P=1 equal single expo, test powlaw kernel at t=0, test powlaw cutoff kernel at t=0, Test the the memory kernel functions, TestHawkesKernels

### Community 14 - "ApproxPowlaw Grid Estimation Tests"
Cohesion: 0.15
Nodes (7): Check if mle function called correct and the correct number of times, Check if invalid timestamp input raises error, Check if invalid T input raises error, Check if all attributes set after succeseful estimation, Class for testing the estimate_grid method, Check if mle function called correct and the correct number of times, TestApproxPowlaw_estimate_grid

### Community 15 - "Exponential Estimation Tests"
Cohesion: 0.15
Nodes (7): Test the .estimate method., Test if .estimate() calls correct mle with custom param_vec0, Test if .estimate() calls correct mle with custom param_vec0, Check if invalid timestamp input raises error, Check if invalid T input raises error, Check if all attributes set after succeseful estimation, TestExpo_estimate

### Community 16 - "Sum-Exp Estimation Tests"
Cohesion: 0.15
Nodes (7): Test the SumExpHawkesProcessInference .estimate method., Test if .estimate() calls correct mle with custom param_vec0, Test if .estimate() calls correct mle with custom param_vec0, Check if invalid timestamp input raises error, Check if invalid T input raises error, Check if all attributes set after succeseful estimation, TestSumExpo_estimate

### Community 17 - "Compensator Functions"
Cohesion: 0.26
Nodes (10): Computes the compensator for a Hawkes procss with P-sum exponential kernel., Computes the compensator for a Hawkes procss with approximate power-law kernel., Computes the compensator for a Hawkes procss with single exponential kernel., Computes the compensator for a Hawkes procss with approximate power-law kernel w, uvhp_approx_powl_compensator(), uvhp_approx_powl_cut_compensator(), uvhp_expo_compensator(), uvhp_sum_expo_compensator() (+2 more)

### Community 18 - "ApproxPowlaw Getter/LogL Tests"
Cohesion: 0.17
Nodes (6): Tests the compute logL method, test if method refuses if model parameters not set, Check if compute_logL function called correct and the  correct params, Tests the getter equals estimate return and attribute, TestApproxPowlaw_getter, TestAproxPowl_compute_logL

### Community 19 - "Expo Grid Estimation Tests"
Cohesion: 0.18
Nodes (6): Check if invalid timestamp input raises error, Check if invalid T input raises error, Check if all attributes set after succeseful estimation, Class for testing the estimate_grid method, Check if mle function called correct and the correct number of times, TestExpo_estimate_grid

### Community 20 - "Sum-Exp Grid Estimation Tests"
Cohesion: 0.18
Nodes (6): Check if invalid timestamp input raises error, Check if invalid T input raises error, Check if all attributes set after succeseful estimation, Class for testing the estimate_grid method, Check if mle function called correct and the correct number of times, TestSumExpo_estimate_grid

### Community 21 - "Sum-Exp Inference Class"
Cohesion: 0.22
Nodes (6): r""" Fitting of unvivariate Hawkes processes with P-sum exponentials memory kern, Args:             P (int): The number of exponentials that make up the P-sum exp, Returns the fitted model parameters:          Returns:             tuple: Tuple, SumExpHawkesProcessInference, Tests the getter equals estimate return and attribute, TestSumExpoInference_getter

### Community 22 - "ApproxPowlaw Intensity Tests"
Cohesion: 0.22
Nodes (6): TestCase, Tests the intensity evaluation functions, Check if intensity is equal to background intensity until first event arrival., Check if all jumpsizes have the desired size., Check decay speed after first event arrival, TestApproxPowlaw_intensity

### Community 23 - "Expo Intensity Tests"
Cohesion: 0.22
Nodes (5): Tests the intensity evaluation functions, Check if intensity is equal to background intensity until first event arrival., Check if all jumpsizes have the desired size., Check decay speed after first event arrival, TestExpo_intensity

### Community 24 - "PowlawCutoff Intensity Tests"
Cohesion: 0.22
Nodes (5): Check if intensity is equal to background intensity until first event arrival., Check if all jumpsizes have the desired size., Check decay speed after first event arrival, Tests the intensity evaluation functions, TestApproxPowlawCutoff_intensity

### Community 25 - "ApproxPowlaw Inference Class"
Cohesion: 0.25
Nodes (5): ApproxPowerlawHawkesProcessInference, r""" Fitting of unvivariate Hawkes processes with approximate power-law memory k, Args:             kernel (str): Must be one of: 'powlaw', 'powlaw-cutoff'. Speci, Set parameters, timestamps and the end time T of the model manually., Returns the fitted model parameters:          Returns:             tuple: Tuple

### Community 26 - "Exponential Inference Class"
Cohesion: 0.25
Nodes (5): ExpHawkesProcessInference, r""" Fitting of unvivariate Hawkes processes with single exponentials memory ker, Args:             rng (optional): numpy random number generator.             For, Set parameters manually.         This function is overriden and disabled for the, Returns the fitted model parameters:          Returns:             tuple: Tuple

### Community 27 - "Expo Kernel-Value Tests"
Cohesion: 0.25
Nodes (4): Tests the kernel_values method, test if method refuses if model parameters not set, Check if kernel values function called correct and the with correct params, TestExpo_kernel_values

### Community 28 - "Sum-Exp Compensator Tests"
Cohesion: 0.25
Nodes (4): Tests the compensator method, test if method refuses if model parameters not set, Check if compensator function called correct and the with correct params, TestSumExpo_compensator

### Community 29 - "Sum-Exp Intensity Tests"
Cohesion: 0.25
Nodes (5): Tests the intensity evaluation functions, Check if intensity is equal to background intensity until first event arrival., Check if all jumpsizes have the desired size., Check decay speed after first event arrival, TestSumExpo_intensity

### Community 30 - "ApproxPowlaw Compensator Tests"
Cohesion: 0.29
Nodes (4): Tests the method compensator, test if method refuses if model not yet succesfully estimated, Check if compensator function called correct and the correct with correct params, TestApproxPowlaw_compensator

### Community 31 - "ApproxPowlaw Intensity Method Tests"
Cohesion: 0.29
Nodes (4): Tests the method intensity, test if method refuses if model paramters not set, Check if intensity function called correct and the correct with correct params, TestApproxPowlaw_intensity

### Community 32 - "ApproxPowlaw Kernel-Value Tests"
Cohesion: 0.29
Nodes (4): Tests the kernel_values method, test if method refuses if model parameters not set, Check if kernel values function called correct and the with correct params, TestApproxPowl_kernel_values

### Community 33 - "Expo Compensator Tests"
Cohesion: 0.29
Nodes (4): Tests the method compensator, test if method refuses if model paramters not set, Check if compensator function called correct and the correct with correct params, TestExpo_compensator

### Community 34 - "Expo Intensity Method Tests"
Cohesion: 0.29
Nodes (4): Tests the method intensity, test if method refuses if model paramters not set, Check if intensity function called correct and the correct with correct params, TestExpo_intensity

### Community 35 - "Expo Compute-LogL Tests"
Cohesion: 0.29
Nodes (4): Tests the compute logL method, test if method refuses if model parameters not set, Check if compute_logL function called correct and the  correct params, TestExpo_compute_logL

### Community 36 - "Sum-Exp Intensity Method Tests"
Cohesion: 0.29
Nodes (4): Tests the intensity method, test if method refuses if model parameters not set, Check if intensity function called correct and the with correct params, TestSumExpo_intensity

### Community 37 - "Sum-Exp Kernel-Value Tests"
Cohesion: 0.29
Nodes (4): Tests the kernel_values method, test if method refuses if model parameters not set, Check if kernel values function called correct and the with correct params, TestSumExpo_kernel_values

### Community 38 - "Sum-Exp Compute-LogL Tests"
Cohesion: 0.29
Nodes (4): Tests the compute logL method, test if method refuses if model parameters not set, Check if compute_logL function called correct and the  correct params, TestSumExpo_compute_logL

### Community 39 - "CI & Publishing Pipeline"
Cohesion: 0.33
Nodes (6): Develop Build & TestPyPI Publish Workflow, Migrate to PyPI Trusted Publishing (OIDC), Master Build & PyPI Publish Workflow, pytest, tox Test Runner, Tests GitHub Actions Workflow

## Knowledge Gaps
- **10 isolated node(s):** `build_docs.sh script`, `P-sum Exponential Memory Kernel`, `Homogenous Poisson Process`, `Compensator`, `simulation module` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UnivariateHawkesProcess` connect `Process Base Class` to `MLE Inference & Poisson`, `Intensity Functions`, `Log-Likelihood Computation`, `Parameter Validators`, `Integer Range Validator`, `Memory Kernel Functions`, `Compensator Functions`, `Sum-Exp Inference Class`, `ApproxPowlaw Inference Class`, `Exponential Inference Class`?**
  _High betweenness centrality (0.202) - this node is a cross-community bridge._
- **Why does `ApproxPowerlawHawkesProcessInference` connect `ApproxPowlaw Inference Class` to `MLE Inference & Poisson`, `ApproxPowlaw Kernel-Value Tests`, `Process Base Class`, `Parameter Validators`, `Integer Range Validator`, `ApproxPowlaw Estimation Tests`, `ApproxPowlaw Grid Estimation Tests`, `ApproxPowlaw Getter/LogL Tests`, `ApproxPowlaw Compensator Tests`, `ApproxPowlaw Intensity Method Tests`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `ExpHawkesProcessInference` connect `Exponential Inference Class` to `MLE Inference & Poisson`, `Expo Compensator Tests`, `Expo Intensity Method Tests`, `Process Base Class`, `Parameter Validators`, `Expo Compute-LogL Tests`, `Integer Range Validator`, `Expo Getter Tests`, `Exponential Estimation Tests`, `Expo Grid Estimation Tests`, `Expo Kernel-Value Tests`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `ApproxPowerlawHawkesProcessInference` (e.g. with `UnivariateHawkesProcess` and `IntInExRange`) actually correct?**
  _`ApproxPowerlawHawkesProcessInference` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `ExpHawkesProcessInference` (e.g. with `UnivariateHawkesProcess` and `IntInExRange`) actually correct?**
  _`ExpHawkesProcessInference` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `SumExpHawkesProcessInference` (e.g. with `UnivariateHawkesProcess` and `IntInExRange`) actually correct?**
  _`SumExpHawkesProcessInference` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `UnivariateHawkesProcess` (e.g. with `ApproxPowerlawHawkesProcessInference` and `ExpHawkesProcessInference`) actually correct?**
  _`UnivariateHawkesProcess` has 13 INFERRED edges - model-reasoned connections that need verification._