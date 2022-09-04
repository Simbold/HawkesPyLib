URLS=[
"HawkesPyLib/index.html",
"HawkesPyLib/simulation.html",
"HawkesPyLib/core/index.html",
"HawkesPyLib/core/simulation.html",
"HawkesPyLib/core/logll.html",
"HawkesPyLib/core/kernel.html",
"HawkesPyLib/core/intensity.html",
"HawkesPyLib/core/compensator.html",
"HawkesPyLib/processes.html",
"HawkesPyLib/inference.html"
];
INDEX=[
{
"ref":"HawkesPyLib",
"url":0,
"doc":" HawkesPyLib is a python package for simulation, and inference of Hawkes processes."
},
{
"ref":"HawkesPyLib.simulation",
"url":1,
"doc":""
},
{
"ref":"HawkesPyLib.simulation.ExpHawkesProcessSimulation",
"url":1,
"doc":"Class for simulation of univariate Hawkes processes with single exponential memory kernel. The conditional intensity function is defined as:  \\lambda(t) = \\mu + \\dfrac{\\eta}{\\theta} \\sum_{t_i  0\\). eta (float): The branching ratio, \\(0  1\\). theta (float): Exponential decay parameter, \\(\\theta > 0\\). Attributes: mu (float): The background intensity, \\(\\mu > 0\\). eta (float): The branching ratio, \\(0  1\\). theta (float): Exponential decay parameter, \\(\\theta > 0\\). T (float): Maximum time until the Hawkes process was simulated. timestamps (np.ndarray): 1d array of simulated arrival times. n_jumps (int): Number of simulated arrival times."
},
{
"ref":"HawkesPyLib.simulation.ExpHawkesProcessSimulation.simulate",
"url":1,
"doc":"Generates a realization of the specified Hawkes process. Args: T (float): Maximum time until which the Hawkes process will be simulated. seed (int, optional): Seed the random number generator. Returns: np.ndarray: 1d array of simulated arrival times.",
"func":1
},
{
"ref":"HawkesPyLib.simulation.ExpHawkesProcessSimulation.intensity",
"url":1,
"doc":"Evaluates the intensity function \\(\\lambda(t)\\) on a grid of equidistant timestamps over the closed interval [0, T] with step size  step_size .  ! note - Additionally to the equidistant time grid the intensity is evaluated at each simulated arrival time. - If the process end time T is not perfectly divisible by the step size the 'effective' step size deviates slightly from the given one. Args: step_size (float): Step size of the time grid. Returns: np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.simulation.ExpHawkesProcessSimulation.kernel_values",
"url":1,
"doc":"Returns the value of the memory kernel at given time values. The memory kernel of the single exponential memory kernel is defined as:  g(t) = \\dfrac{\\eta}{\\theta} e^{(-t/\\theta)}  Args: times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel. Returns: np.ndarray: 1d array containing the values of the memory kernel value at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.simulation.SumExpHawkesProcessSimulation",
"url":1,
"doc":"Class for simulation of univariate Hawkes processes with single exponential memory kernel. The conditional intensity function is defined as:  \\lambda(t) = \\mu + \\dfrac{\\eta}{P} \\sum_{t_i  0\\). eta (float): The branching ratio, \\(0  1\\). theta_vec (np.ndarray): 1d array of exponential decay parameters, \\(\\theta_k > 0\\). Attributes: mu (float): The background intensity, \\(\\mu > 0\\). eta (float): The branching ratio, \\(0  1\\). theta_vec (np.ndarray): 1d array of exponential decay parameters, \\(\\theta_k > 0\\). T (float): Maximum time until the Hawkes process was simulated. timestamps (np.ndarray): 1d array of simulated arrival times. n_jumps (int): Number of simulated arrival times."
},
{
"ref":"HawkesPyLib.simulation.SumExpHawkesProcessSimulation.simulate",
"url":1,
"doc":"Generates a realization of the specified Hawkes process. Args: T (float): Maximum time until which the Hawkes process will be simulated. seed (int, optional): Seed the random number generator. Returns: np.ndarray: 1d array of simulated arrival times.",
"func":1
},
{
"ref":"HawkesPyLib.simulation.SumExpHawkesProcessSimulation.intensity",
"url":1,
"doc":"Evaluates the intensity function \\(\\lambda(t)\\) on a grid of equidistant timestamps over the closed interval [0, T] with step size  step_size .  ! note - Additionally to the equidistant time grid the intensity is evaluated at each simulated arrival time. - If the process end time T is not perfectly divisible by the step size the 'effective' step size deviates slightly from the given one. Args: step_size (float): Step size of the time grid. Returns: np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.simulation.SumExpHawkesProcessSimulation.kernel_values",
"url":1,
"doc":"Returns the value of the memory kernel at given time values. The memory kernel of the P-sum exponential memory kernel is defined as:  g(t) = \\dfrac{\\eta}{P} \\sum_{k=1}^{P} \\dfrac{1}{\\theta_k} e^{(-t/\\theta_k)}  Args: times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel. Returns: np.ndarray: 1d array containing the values of the memory kernel value at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation",
"url":1,
"doc":"Class for simulation of univariate Hawkes processes with approximate power-law memory kernel. The conditional intensity function for the approximate power-law kernel is defined as: \\[\\lambda(t) = \\mu + \\sum_{t_i  0\\). eta (float): The branching ratio, \\(0  1\\). alpha (float): Tail index of the power-law decay, \\(\\alpha > 0\\). tau0 (float): Timescale parameter, \\(\\tau_0 > 0\\). m (float): Scale parameter of the power-law weights, \\(m > 0 \\). M (int): Number of weighted exponential functions that approximate the power-law. Attributes: mu (float): The background intensity, \\(\\mu > 0\\). eta (float): The branching ratio, \\(0  1\\). alpha (float): Tail index of the power-law decay, \\(\\alpha > 0\\). tau0 (float): Timescale parameter, \\(\\tau_0 > 0\\). m (float): Scale parameter of the power-law weights, \\(m > 0 \\). M (int): Number of weighted exponential functions that approximate the power-law. T (float): Maximum time until the Hawkes process was simulated. timestamps (np.ndarray): 1d array of simulated arrival times. n_jumps (int): Number of simulated arrival times."
},
{
"ref":"HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.simulate",
"url":1,
"doc":"Generates a realization of the specified Hawkes process. Args: T (float): Maximum time until which the Hawkes process will be simulated. seed (int, optional): Seed the random number generator. Returns: np.ndarray: 1d array of simulated arrival times.",
"func":1
},
{
"ref":"HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.intensity",
"url":1,
"doc":"Evaluates the intensity function \\(\\lambda(t)\\) on a grid of equidistant timestamps over the closed interval [0, T] with step size  step_size .  ! note - Additionally to the equidistant time grid the intensity is evaluated at each simulated arrival time. - If the process end time T is not perfectly divisible by the step size the 'effective' step size deviates slightly from the given one. Args: step_size (float): Step size of the time grid. Returns: np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.simulation.ApproxPowerlawHawkesProcessSimulation.kernel_values",
"url":1,
"doc":"Returns the value of the memory kernel at given time values. The memory kernel of the approximate power-law memory kernel is defined as:  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} \\bigg]  and the memory kernel of the approximate power-law kernel with smooth cutoff is defined as:  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \\bigg]  Args: times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel. Returns: np.ndarray: 1d array containing the values of the memory kernel value at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.core",
"url":2,
"doc":""
},
{
"ref":"HawkesPyLib.core.simulation",
"url":3,
"doc":""
},
{
"ref":"HawkesPyLib.core.simulation.uvhp_approx_powl_cutoff_simulator",
"url":3,
"doc":"Simulates a Hawkes process with approximate power-law memory kernel. Implements Ogata's modified thinning algorithm as described in algorithm 2 in (Ogata 1981) . References: - Ogata, Y. (1981). On lewis simulation method for point processes. IEEE transactions on information theory, 27(1):2331. Args: T (float): Time until which the Hawkes process will be simulated. T > 0 mu (float): Background intensity of the Hawkes process. mu > 0 eta (float): Branching ratio of the Hawkes process. 0  0 tau (float): Memory kernel parameter. tau > 0 m (float): Memory kernel variable. m > 0 M (int): Memory kernel variable. M > 0 seed (int, optional): Seed for numba's random number generator. Defaults to None. Returns: np.ndarray: Array of simulated timestamps",
"func":1
},
{
"ref":"HawkesPyLib.core.simulation.uvhp_expo_simulator",
"url":3,
"doc":"Simulates a Hawkes process with single exponential memory kernel. Implements Ogata's modified thinning algorithm as described in algorithm 2 in (Ogata 1981). References: - Ogata, Y. (1981). On lewis simulation method for point processes. IEEE transactions on information theory, 27(1):2331. Args: T (float): Time until which the Hawkes process will be simulated. T > 0 mu (float): Background intensity of the Hawkes process. mu > 0 eta (float): Branching ratio of the Hawkes process. 0  0 seed (_type_, optional): Seed numbda's random number generator. Defaults to None. Returns: np.ndarray: Array of simulated timestamps",
"func":1
},
{
"ref":"HawkesPyLib.core.simulation.uvhp_approx_powl_simulator",
"url":3,
"doc":"Simulates a Hawkes process with approximate power-law memory kernel. Implements Ogata's modified thinning algorithmas described in algorithm 2 in (Ogata 1981). References: - Ogata, Y. (1981). On lewis simulation method for point processes. IEEE transactions on information theory, 27(1):2331. Args: T (float): Time until which the Hawkes process will be simulated. T > 0 mu (float): Background intensity of the Hawkes process. mu > 0 eta (float): Branching ratio of the Hawkes process. 0  0 tau (float): Memory kernel parameter. tau > 0 m (float): Memory kernel variable. m > 0 M (int): Memory kernel variable. M > 0 seed (int, optional): Seed for numba's random number generator. Defaults to None. Returns: np.ndarray: Array of simulated timestamps",
"func":1
},
{
"ref":"HawkesPyLib.core.simulation.uvhp_sum_expo_simulator",
"url":3,
"doc":"Simulates a Hawkes process with P-sum expoential memory kernel. Implements Ogata's modified thinning algorithmas described in algorithm 2 in (Ogata 1981). References: - Ogata, Y. (1981). On lewis simulation method for point processes. IEEE transactions on information theory, 27(1):2331. Args: T (float): Time until which the Hawkes process will be simulated. T > 0 mu (float): Background intensity of the Hawkes process. mu > 0 eta (float): Branching ratio of the Hawkes process. 0  0 seed (int, optional): Seed for numba's random number generator. Defaults to None. Returns: np.ndarray: Array of simulated timestamps",
"func":1
},
{
"ref":"HawkesPyLib.core.logll",
"url":4,
"doc":""
},
{
"ref":"HawkesPyLib.core.logll.uvhp_expo_logL",
"url":4,
"doc":"Log-likelihood function for a Hawkes Process with single exponential kernel Args: param_vec (np.ndarray): array of parameter: [mu, eta, theta] sample_vec (np.ndarray): array of timestamps. Must be sorted in ascending order. All timestamps must be positive. tn (float): End time of the Hawkes process. Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1] Returns: float: The negative log-likelihood value",
"func":1
},
{
"ref":"HawkesPyLib.core.logll.uvhp_expo_logL_grad",
"url":4,
"doc":"Gradient of the log-likelihood function for a Hawkes Process with single exponential kernel Args: param_vec (np.ndarray): array of parameter: [mu, eta, theta] sample_vec (np.ndarray): array of timestamps. Must be sorted in ascending order. All timestamps must be positive. tn (float): End time of the Hawkes process. Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1] Returns: np.ndarray: The negative value of each gradient: [mu gradient, eta gradient, theta gradient]",
"func":1
},
{
"ref":"HawkesPyLib.core.logll.uvhp_sum_expo_logL",
"url":4,
"doc":"Log-likelihood function for a Hawkes Process with P-sum exponential kernel Args: param_vec (np.ndarray): array of parameter: [mu, eta, theta1, theta2,  ., thetaP] sample_vec (np.ndarray): array of timestamps. Must be sorted in ascending order. All timestamps must be positive. tn (float): End time of the Hawkes process. Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1] Returns: float: The negative log-likelihood value",
"func":1
},
{
"ref":"HawkesPyLib.core.logll.uvhp_sum_expo_logL_grad",
"url":4,
"doc":"Gradient of the log-likelihood function for a Hawkes Process with P-sum exponential kernel Args: param_vec (np.ndarray): array of parameter: [mu, eta, theta1, theta2,  ., thetaP] sample_vec (np.ndarray): array of timestamps. Must be sorted in ascending order. All timestamps must be positive. tn (float): End time of the Hawkes process. Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1] Returns: np.ndarray: The negative value of each gradient: [mu gradient, eta gradient, theta1 gradient,  ., thetaP gradient]",
"func":1
},
{
"ref":"HawkesPyLib.core.logll.uvhp_approx_powl_cut_logL",
"url":4,
"doc":"Log-likelihood function for a Hawkes Process with approximate power-law kernel with cutoff component Args: param_vec (np.ndarray): array of parameter: [mu, eta, theta1, theta2,  ., thetaP] sample_vec (np.ndarray): array of timestamps. Must be sorted in ascending order. All timestamps must be positive. tn (float): End time of the Hawkes process. Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1] m (float): Fixed kernel variable M (int): Fixed kernel variable Returns: float: negative log-likelihood value",
"func":1
},
{
"ref":"HawkesPyLib.core.logll.uvhp_approx_powl_logL",
"url":4,
"doc":"Log-likelihood function for a Hawkes Process with approximate power-law kernel without cutoff component Args: param_vec (np.ndarray): array of parameter: [mu, eta, theta1, theta2,  ., thetaP] sample_vec (np.ndarray): array of timestamps. Must be sorted in ascending order. All timestamps must be positive. tn (float): End time of the Hawkes process. Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1] m (float): Fixed kernel variable M (int): Fixed kernel variable Returns: float: negative log-likelihood value",
"func":1
},
{
"ref":"HawkesPyLib.core.kernel",
"url":5,
"doc":""
},
{
"ref":"HawkesPyLib.core.kernel.uvhp_expo_kernel",
"url":5,
"doc":"Computes values of the single exponential Hawkes process memory kernel. Args: t (np.ndarray): Single time value or 1d array containing all the times at which the kernel value will be computed. Must be positive. eta (float): The branching ratio, 0  1 theta (float): 1d array of theta decay parameters, theta > 0 Returns: np.ndarray: Single value or 1d array containing the kernel values at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.core.kernel.uvhp_sum_expo_kernel",
"url":5,
"doc":"Computes values of the P-sum exponential Hawkes process memory kernel. Args: t (np.ndarray): Single time value or 1d array containing all the times at which the kernel value will be computed. Must be positive. eta (float): The branching ratio, 0  1 theta_vec (float): 1d array of theta decay parameters, theta_k > 0 Returns: np.ndarray: Single value or 1d array containing the kernel values at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.core.kernel.uvhp_approx_powl_cutoff_kernel",
"url":5,
"doc":"Computes values of the Approximate power-law memory kernel with smooth cutoff component. Args: t (np.ndarray): Single time value or 1d array containing all the times at which the kernel value will be computed. Must be positive. eta (float): Branching ratio of the Hawkes process, 0 > eta  0 tau (float): Approximate location of cutoff, tau > 0 m (float): Approximate power-law parameter, m > 0 M (int): Number of weighted exponential kernels that approximate the power-law Returns: np.ndarray: Single value or 1d array containing the kernel values at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.core.kernel.uvhp_approx_powl_kernel",
"url":5,
"doc":"Computes values of the Approximate power-law memory kernel. Args: t (np.ndarray): Single time value or 1d array containing all the times at which the kernel value will be computed. Must be positive. eta (float): Branching ratio of the Hawkes process, 0 > eta  0 tau (float): Approximate location of cutoff, tau > 0 m (float): Approximate power-law parameter, m > 0 M (int): Number of weighted exponential kernels that approximate the power-law Returns: np.ndarray: Single value or 1d array containing the kernel values at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.core.intensity",
"url":6,
"doc":""
},
{
"ref":"HawkesPyLib.core.intensity.generate_eval_grid",
"url":6,
"doc":"Generates an equidistant grid in the closed interval [0, T] with step size given by step_size. Args: step_size (float): Step size of the equidistant grid. T (float): End point of the equidistant grid (the last value). Returns: np.ndarray: 1d array equidistant grid between from 0 to t with step size.",
"func":1
},
{
"ref":"HawkesPyLib.core.intensity.uvhp_expo_intensity",
"url":6,
"doc":"Evaluation of the intensity function of a univariate Hawkes process with single exponential kernel Args: sample_vec (np.ndarray): Jump times of the Hawkes process. Must be non-negative and in ascending order! grid (np.ndarray): Times at which the intensity function is evaluated at. Must be non-negative and in ascending order mu (float): Background intensity of the Hawkes process, mu > 0 eta (float): Branching ratio of the Hawkes process, 0 > eta  0 Returns: np.ndarray: A 2d numpy array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.core.intensity.uvhp_sum_expo_intensity",
"url":6,
"doc":"Evaluation of the intensity function of a univariate Hawkes process with P-sum exponential kernel Args: sample_vec (np.ndarray): Jump times of the Hawkes process. Must be non-negative and in ascending order! grid (np.ndarray): Times at which the intensity function is evaluated at. Must be non-negative and in ascending order mu (float): Background intensity of the Hawkes process, mu > 0 eta (float): Branching ratio of the Hawkes process, 0 > eta  0 Returns: np.ndarray: A 2d numpy array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.core.intensity.uvhp_approx_powl_cutoff_intensity",
"url":6,
"doc":"Evaluation of the intensity function of a univariate Hawkes process with approximate power-law kernel with smooth cutoff component Args: sample_vec (np.ndarray): Jump times of the Hawkes process. Must be non-negative and in ascending order! grid (np.ndarray): Times at which the intensity function is evaluated at. Must be non-negative and in ascending order mu (float): Background intensity of the Hawkes process, mu > 0 eta (float): Branching ratio of the Hawkes process, 0 > eta  0 tau (float): Approximate location of cutoff, tau > 0 m (float): Approximate power-law parameter, m > 0 M (int): Number of weighted exponential kernels that approximate the power-law Returns: np.ndarray: A 2d numpy array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.core.intensity.uvhp_approx_powl_intensity",
"url":6,
"doc":"Evaluation of the intensity function of a univariate Hawkes process with approximate power-law kernel. Args: sample_vec (np.ndarray): Jump times of the Hawkes process. Must be non-negative and in ascending order! grid (np.ndarray): Times at which the intensity function is evaluated at. Must be non-negative and in ascending order mu (float): Background intensity of the Hawkes process, mu > 0 eta (float): Branching ratio of the Hawkes process, 0 > eta  0 tau (float): Approximate location of cutoff, tau > 0 m (float): Approximate power-law parameter, m > 0 M (int): Number of weighted exponential kernels that approximate the power-law Returns: np.ndarray: A 2d numpy array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.core.compensator",
"url":7,
"doc":""
},
{
"ref":"HawkesPyLib.core.compensator.uvhp_expo_compensator",
"url":7,
"doc":"Computes the compensator for a Hawkes procss with single exponential kernel. Args: sample_vec (np.ndarray): numpy array of timestamps. Must be non-negative and sorted in ascending order. mu (float): Background intensity of the Hawkes process. mu > 0 eta (float): Branching ratio of the Hawkes process. 0  0 Returns: np.ndarray: numpy array of timestamps",
"func":1
},
{
"ref":"HawkesPyLib.core.compensator.uvhp_approx_powl_compensator",
"url":7,
"doc":"Computes the compensator for a Hawkes procss with approximate power-law kernel. Args: sample_vec (np.ndarray): numpy array of timestamps. Must be non-negative and sorted in ascending order. mu (float): Background intensity of the Hawkes process. mu > 0 eta (float): Branching ratio of the Hawkes process. 0  0 tau (float): Decay speed of the kernel. tau > 0 m (float): Memory kernel parameter, m > 0 M (int): Memory kernel parameter, M > 0 Returns: np.ndarray: numpy array of timestamps",
"func":1
},
{
"ref":"HawkesPyLib.core.compensator.uvhp_approx_powl_cut_compensator",
"url":7,
"doc":"Computes the compensator for a Hawkes procss with approximate power-law kernel with cutoff. Args: sample_vec (np.ndarray): numpy array of timestamps. Must be non-negative and sorted in ascending order. mu (float): background intensity of the Hawkes process eta (float): Branching ratio of the Hawkes process. 0  0 tau (float): Decay speed of the kernel. tau > 0 m (float): Memory kernel parameter, m > 0 M (int): Memory kernel parameter, M > 0 Returns: np.ndarray: numpy array of timestamps",
"func":1
},
{
"ref":"HawkesPyLib.core.compensator.uvhp_sum_expo_compensator",
"url":7,
"doc":"Computes the compensator for a Hawkes procss with P-sum exponential kernel. Args: sample_vec (np.ndarray): numpy array of timestamps. Must be non-negative and sorted in ascending order. mu (float): background intensity of the Hawkes process. mu > 0 eta (float): Branching ratio of the Hawkes process. 0  0 all values Returns: np.ndarray: numpy array of timestamps",
"func":1
},
{
"ref":"HawkesPyLib.processes",
"url":8,
"doc":""
},
{
"ref":"HawkesPyLib.processes.UnivariateHawkesProcess",
"url":8,
"doc":"Implements multiple univariate Hawkes process models and allows for the calculation of basic process characteristics. The following Hawkes process models are implemented: - Hawkes process with single exponential memory kernel ('expo'  kernel ), defined by the conditional intensity function: \\[\\lambda(t) = \\mu + \\dfrac{\\eta}{\\theta} \\sum_{t_i  0\\). eta (float): The branching ratio, \\(0  1\\). theta (float): Decay speed of single exponential kernel, theta > 0. Only set if  kernel is set to 'expo'. theta_vec (np.ndarray, optional): Decay speed of P-sum, exponential kernel, theta_k > 0. Only set if  kernel is set to 'sum-expo'. alpha (float): Tail index of the power-law decay, \\(\\alpha > 0\\). Only set if  kernel is set to 'powlaw' or 'powlaw-cutoff'. tau0 (float): Timescale parameter, \\(\\tau_0 > 0\\). Only set if  kernel is set to 'powlaw' or 'powlaw-cutoff'. T (float): End time of the Hawkes process. timestamps (np.ndarray): 1d array of arrival times."
},
{
"ref":"HawkesPyLib.processes.UnivariateHawkesProcess.set_params",
"url":8,
"doc":"Set Hawkes process model parameters. Args: mu (float): The background intensity, \\(\\mu > 0\\). eta (float): The branching ratio, \\(0  1\\). theta (float): Decay speed of single exponential kernel, theta > 0. Must be set if  kernel is set to 'expo'. theta_vec (np.ndarray, optional): Decay speed of P-sum, exponential kernel, theta_k > 0. Must be set if  kernel is set to 'sum-expo'. alpha (float): Tail index of the power-law decay, \\(\\alpha > 0\\). Must be set if  kernel is set to 'powlaw' or 'powlaw-cutoff'. tau0 (float): Timescale parameter, \\(\\tau_0 > 0\\). Must be set if  kernel is set to 'powlaw' or 'powlaw-cutoff'. m (float, optional): Scale parameter of the power-law weights, m > 0. Must be set if  kernel is set to 'powlaw' or 'powlaw-cutoff'. M (int, optional): Number of weighted exponential kernels used in the approximate power-law kernel. Must be set if  kernel is set to 'powlaw' or 'powlaw-cutoff'.",
"func":1
},
{
"ref":"HawkesPyLib.processes.UnivariateHawkesProcess.set_arrival_times",
"url":8,
"doc":"Set arrival times of the process. Args: timestamps (np.ndarray): 1d array of arrival times of the process. Arrival times must be positive and ordered. T (float): End time of the process. T must be larger or equal to the last arrival time.",
"func":1
},
{
"ref":"HawkesPyLib.processes.UnivariateHawkesProcess.get_params",
"url":8,
"doc":"Returns the model parameters: Returns: tuple: Tuple of parameter values such as: If  kernel set to 'expo': -  mu (float),  eta (float),  theta (float) If  kernel set to 'sum-expo': -  mu (float),  eta (float),  theta_vec (np.ndarray) If  kernel set to 'powlaw' or 'powlaw-cutoff': -  mu (float),  eta (float),  alpha (float),  tau0 (float),  m (float),  M (int)",
"func":1
},
{
"ref":"HawkesPyLib.processes.UnivariateHawkesProcess.intensity",
"url":8,
"doc":"Evaluates the intensity function \\(\\lambda(t)\\) on a grid of equidistant timestamps over the closed interval [0, T] with step size  step_size .  ! note - Additionally to the equidistant time grid the intensity is evaluated at each simulated arrival time. - If the process end time T is not perfectly divisible by the step size the 'effective' step size deviates slightly from the given one. Args: step_size (float): Step size of the time grid. Returns: np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.processes.UnivariateHawkesProcess.compensator",
"url":8,
"doc":"Computes the compensator of the process at the set timestamps given the set parameter values. The compensator of a point process is defined as the integral of the conditional intensity:  \\Lambda(t) = \\int_0^t \\lambda(u) du   ! tip If the set arrival times are a realization of the Hawkes process specified by the set parameters, the resulting compensator will be a Poisson process with unit intensity (assuming a large enough sample size). Returns: np.ndarray: 1d array of timestamps (the compensator)",
"func":1
},
{
"ref":"HawkesPyLib.processes.UnivariateHawkesProcess.kernel_values",
"url":8,
"doc":"Returns the value of the memory kernel at given time values. The single exponential memory kernel:  g(t) = \\dfrac{\\eta}{\\theta} e^{(-t/\\theta)}  The P-sum exponential memory kernel:  g(t) = \\dfrac{\\eta}{P} \\sum_{k=1}^{P} \\dfrac{1}{\\theta_k} e^{(-t/\\theta_k)}  The approximate power-law memory kernel:  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} \\bigg]  And the approximate power-law kernel with smooth cutoff :  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \\bigg]  Args: times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel. Returns: np.ndarray: 1d array containing the values of the memory kernel value at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.processes.UnivariateHawkesProcess.compute_logL",
"url":8,
"doc":"Computes the log-likelihood function given the set parameters, arrival times and process end time T. The log-likelihood function of a Hawkes process is given by:  logL(t_1, ., t_n) = -\\int_0^T \\lambda(t) du + \\sum_{t=1}^n log(\\lambda(t_i  Returns: float: The log-likelihood value",
"func":1
},
{
"ref":"HawkesPyLib.processes.UnivariateHawkesProcess.ic",
"url":8,
"doc":"Calculates one of the following information criteria given the set parameters and arrival times: - 'aic': Akaike information criteria - 'bic': Baysian information criteria - 'hq': Hannan-Quinn information criteria Args: type (str): Which information criterion to return, must be one of: 'aic', 'bic', 'hq' Returns: float: The information criterions value",
"func":1
},
{
"ref":"HawkesPyLib.inference",
"url":9,
"doc":""
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference",
"url":9,
"doc":"Inference of unvivariate Hawkes processes with single exponentials memory kernel. Hawkes process with single exponential memory kernel ('expo' kernel), is defined by the conditional intensity function: \\[\\lambda(t) = \\mu + \\dfrac{\\eta}{\\theta} \\sum_{t_i  0. timestamps (np.ndarray): 1d array of arrival times that were used in the most recent fitting routine. T: (float): End time of the estimated process. logL (Float): The log-likelihood value of the fitted process."
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference.estimate",
"url":9,
"doc":"Estimates the Hawkes process parameters using maximum likelihood estimation. Args: timestamps (np.ndarray): 1d array of arrival times. Must be sorted and only positive timestamps are allowed! T (float): The end time of the Hawkes process. return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0 max_attempts (int, optional): Number of times the maximum likelihood estimation repeats with new starting values if the optimization routine does not exit successfully. Defaults to 5. custom_param_vec0 (bool, optional): If custom initial values should be used. Defaults to False. If true you must supply an addtional variable 'param_vec0' a 1d numpy array containing the initial starting values: param_vec0 = [mu0 > 0, 0  0]. Returns: tuple: (optional) if  return_params =True returns tuple of fitted parameters: mu, eta, theta",
"func":1
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference.estimate_grid",
"url":9,
"doc":"Estimates the Hawkes process parameters using maximum likelihood estimation. Fits the model multiple times using different initial parameter values. Subsequently the fitted model with the largest logL value is returned. The grid applied only to the initial value of eta in the mle routine, mu0 is set in using the unconditonal mean of the process and theta is always chosed randomly. There are three options for the type of eta0 starting value grid: - ('random') starting values for eta0 are chosen randomly. - ('equidistant') starting values for eta0 are equidistant between 0 and 1 - ('custom') A custom eta0 grid of starting value is supplied to 'custom-grid' variable: You must supply an addtional variable 'custom-grid' a 1d numpy array containing a grid of initial starting values with constraint 0 < eta0 < 1. Args: timestamps (np.ndarray): 1d array of arrival times. Must be sorted and only positive timestamps are allowed! T (float): The end time of the Hawkes process. grid_type (str, optional): Type of grid for eta0 starting values, one of: 'random', 'equidistant' or 'custom'. Default to 'equidistant' grid_size (int, optional): The number of optimizations to run. Defaults to 20. return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0 Returns: tuple: (optional) if  return_params =True returns tuple of fitted parameters: mu, eta, theta",
"func":1
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference.get_params",
"url":9,
"doc":"Returns the fitted model parameters: Returns: tuple: Tuple of parameter values such as: -  mu (float),  eta (float),  theta (float)",
"func":1
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference.set_arrival_times",
"url":8,
"doc":"Set arrival times of the process. Args: timestamps (np.ndarray): 1d array of arrival times of the process. Arrival times must be positive and ordered. T (float): End time of the process. T must be larger or equal to the last arrival time.",
"func":1
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference.intensity",
"url":8,
"doc":"Evaluates the intensity function \\(\\lambda(t)\\) on a grid of equidistant timestamps over the closed interval [0, T] with step size  step_size .  ! note - Additionally to the equidistant time grid the intensity is evaluated at each simulated arrival time. - If the process end time T is not perfectly divisible by the step size the 'effective' step size deviates slightly from the given one. Args: step_size (float): Step size of the time grid. Returns: np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference.compensator",
"url":8,
"doc":"Computes the compensator of the process at the set timestamps given the set parameter values. The compensator of a point process is defined as the integral of the conditional intensity:  \\Lambda(t) = \\int_0^t \\lambda(u) du   ! tip If the set arrival times are a realization of the Hawkes process specified by the set parameters, the resulting compensator will be a Poisson process with unit intensity (assuming a large enough sample size). Returns: np.ndarray: 1d array of timestamps (the compensator)",
"func":1
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference.kernel_values",
"url":8,
"doc":"Returns the value of the memory kernel at given time values. The single exponential memory kernel:  g(t) = \\dfrac{\\eta}{\\theta} e^{(-t/\\theta)}  The P-sum exponential memory kernel:  g(t) = \\dfrac{\\eta}{P} \\sum_{k=1}^{P} \\dfrac{1}{\\theta_k} e^{(-t/\\theta_k)}  The approximate power-law memory kernel:  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} \\bigg]  And the approximate power-law kernel with smooth cutoff :  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \\bigg]  Args: times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel. Returns: np.ndarray: 1d array containing the values of the memory kernel value at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference.compute_logL",
"url":8,
"doc":"Computes the log-likelihood function given the set parameters, arrival times and process end time T. The log-likelihood function of a Hawkes process is given by:  logL(t_1, ., t_n) = -\\int_0^T \\lambda(t) du + \\sum_{t=1}^n log(\\lambda(t_i  Returns: float: The log-likelihood value",
"func":1
},
{
"ref":"HawkesPyLib.inference.ExpHawkesProcessInference.ic",
"url":8,
"doc":"Calculates one of the following information criteria given the set parameters and arrival times: - 'aic': Akaike information criteria - 'bic': Baysian information criteria - 'hq': Hannan-Quinn information criteria Args: type (str): Which information criterion to return, must be one of: 'aic', 'bic', 'hq' Returns: float: The information criterions value",
"func":1
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference",
"url":9,
"doc":"Inference of unvivariate Hawkes processes with P-sum exponentials memory kernel. The Hawkes process with P-sum exponential memory kernel ('sum-expo' kernel), is defined by the conditional intensity function: \\[\\lambda(t) = \\mu + \\dfrac{\\eta}{P} \\sum_{t_i < t} \\sum_{k=1}^{P} \\dfrac{1}{\\theta_k} e^{(-(t - t_i)/\\theta_k)}\\] Two maximum liklihood based inference methods are currently available: -  ExpHawkesProcessInference.estimate() maximizes the log-likelihood given a sample of arrival times. -  ExpHawkesProcessInference.estimate_grid() maximizes the log-likelihood on a grid of different starting values and returns the best model out of all fitted models (i.e. the model with the highest log-likelihood). The log-likelihood function of Hawkes processes can be very flat and a single optimization run may get stuck in a local supoptimal maximum. This class inherets from  HawkesPyLib.processes.UnivariateHawkesProcess and provides methods for further analysis of the fitted model: - SumExpHawkesProcessInference.compensator() computes the compensator and may be used as a starting point for a 'residual' analysis. - SumExpHawkesProcessInference.intensity() evaluates the estimated conditional intensity. - SumExpHawkesProcessInference.kernel_values() returns values of the estimated memory kernel. - SumExpHawkesProcessInference.ic() computes information criteria of the estimated model. Args: P (int): The number of exponentials that make up the P-sum exponential memory kernel. rng (optional): numpy random generator. For reproducible results use: rng=np.random.default_rng(seed) Attributes: mu (float): The estimated background intensity. eta (float): The estimated branching ratio. theta_vec (np.ndarray, optional): Estimated decay speeds of P-sum, exponential kernel. timestamps (np.ndarray): 1d array of arrival times that were used in the most recent fitting routine. T (float): End time of the Hawkes process. logL (Float): The log-likelihood value of the fitted process."
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference.estimate",
"url":9,
"doc":"Estimates the Hawkes process parameters using maximum likelihood estimation. Args: timestamps (np.ndarray): 1d array of arrival times. Must be sorted and only positive timestamps are allowed! T (float): The end time of the Hawkes process. return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0 max_attempts (int, optional): Number of times the maximum likelihood estimation repeats with new starting values if the optimization routine does not exit successfully. Defaults to 5. custom_param_vec0 (bool, optional): If custom initial values should be used. Defaults to False. If true you must supply an addtional variable 'param_vec0' a 1d numpy array containing the initial starting values: param_vec0 = [mu0 > 0, 0  0]. Returns: tuple: (optional) if  return_params =True returns tuple of fitted parameters: mu, eta, theta",
"func":1
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference.estimate_grid",
"url":9,
"doc":"Estimates the Hawkes process parameters using maximum likelihood estimation. Fits the model multiple times using different initial parameter values. Subsequently the fitted model with the largest logL value is returned. The grid applied only to the initial value of eta in the mle routine, mu0 is set in using the unconditonal mean of the process and theta is always chosed randomly. There are three options for the type of eta0 starting value grid: - ('random') starting values for eta0 are chosen randomly. - ('equidistant') starting values for eta0 are equidistant between 0 and 1 - ('custom') A custom eta0 grid of starting value is supplied to 'custom-grid' variable: You must supply an addtional variable 'custom-grid' a 1d numpy array containing a grid of initial starting values with constraint 0 < eta0 < 1. Args: timestamps (np.ndarray): 1d array of arrival times. Must be sorted and only positive timestamps are allowed! T (float): The end time of the Hawkes process. grid_type (str, optional): Type of grid for eta0 starting values, one of: 'random', 'equidistant' or 'custom'. Default to 'equidistant' grid_size (int, optional): The number of optimizations to run. Defaults to 20. return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0 Returns: tuple: (optional) if  return_params =True returns tuple of fitted parameters: mu, eta, theta",
"func":1
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference.get_params",
"url":9,
"doc":"Returns the fitted model parameters: Returns: tuple: Tuple of parameter values such as: -  mu (float),  eta (float),  theta_vec (np.ndarray)",
"func":1
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference.set_arrival_times",
"url":8,
"doc":"Set arrival times of the process. Args: timestamps (np.ndarray): 1d array of arrival times of the process. Arrival times must be positive and ordered. T (float): End time of the process. T must be larger or equal to the last arrival time.",
"func":1
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference.intensity",
"url":8,
"doc":"Evaluates the intensity function \\(\\lambda(t)\\) on a grid of equidistant timestamps over the closed interval [0, T] with step size  step_size .  ! note - Additionally to the equidistant time grid the intensity is evaluated at each simulated arrival time. - If the process end time T is not perfectly divisible by the step size the 'effective' step size deviates slightly from the given one. Args: step_size (float): Step size of the time grid. Returns: np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference.compensator",
"url":8,
"doc":"Computes the compensator of the process at the set timestamps given the set parameter values. The compensator of a point process is defined as the integral of the conditional intensity:  \\Lambda(t) = \\int_0^t \\lambda(u) du   ! tip If the set arrival times are a realization of the Hawkes process specified by the set parameters, the resulting compensator will be a Poisson process with unit intensity (assuming a large enough sample size). Returns: np.ndarray: 1d array of timestamps (the compensator)",
"func":1
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference.kernel_values",
"url":8,
"doc":"Returns the value of the memory kernel at given time values. The single exponential memory kernel:  g(t) = \\dfrac{\\eta}{\\theta} e^{(-t/\\theta)}  The P-sum exponential memory kernel:  g(t) = \\dfrac{\\eta}{P} \\sum_{k=1}^{P} \\dfrac{1}{\\theta_k} e^{(-t/\\theta_k)}  The approximate power-law memory kernel:  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} \\bigg]  And the approximate power-law kernel with smooth cutoff :  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \\bigg]  Args: times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel. Returns: np.ndarray: 1d array containing the values of the memory kernel value at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference.compute_logL",
"url":8,
"doc":"Computes the log-likelihood function given the set parameters, arrival times and process end time T. The log-likelihood function of a Hawkes process is given by:  logL(t_1, ., t_n) = -\\int_0^T \\lambda(t) du + \\sum_{t=1}^n log(\\lambda(t_i  Returns: float: The log-likelihood value",
"func":1
},
{
"ref":"HawkesPyLib.inference.SumExpHawkesProcessInference.ic",
"url":8,
"doc":"Calculates one of the following information criteria given the set parameters and arrival times: - 'aic': Akaike information criteria - 'bic': Baysian information criteria - 'hq': Hannan-Quinn information criteria Args: type (str): Which information criterion to return, must be one of: 'aic', 'bic', 'hq' Returns: float: The information criterions value",
"func":1
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference",
"url":9,
"doc":"Inference of unvivariate Hawkes processes with approximate power-law memory kernel. The Hawkes process with approximate power-law kernel ('powlaw'  kernel ), is defined by the conditional intenstiy fnuction: \\[\\lambda(t) = \\mu + \\sum_{t_i < t} \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} \\bigg],\\] where \\(\\mu\\) ( mu ) is the constant background intensity, \\(\\eta \\) ( eta ) is the branching ratio, \\(\\alpha\\) ( alpha ) is the power-law tail exponent and \\(\\tau_0\\) a scale parameter controlling the decay timescale. The Hawkes process with approximate power-law kernel with smooth cutoff ('powlaw-cutoff'  kernel ), defined by the conditional intenstiy fnuction: \\[\\lambda(t) = \\mu + \\sum_{t_i < t} \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \\bigg],\\] where \\(\\mu\\) ( mu ) is the constant background intensity, \\(\\eta \\) ( eta ) is the branching ratio, \\(\\alpha\\) ( alpha ) is the power-law tail exponent and \\(\\tau_0\\) a scale parameter controlling the decay timescale and the location of the smooth cutoff point (i.e. the time duration after each jump at which the intensity reaches a maximum). The true power-law is approximtated by a sum of \\(M\\) exponential function with power-law weights \\(a_k = \\tau_0 m^k\\). \\(M\\) is the number of exponentials used for the approximation and \\(m\\) is a scale parameter. The power-law approximation holds for times up to \\(m^{M-1}\\) after which the memory kernel decays exponentially. S and Z are scaling factors that ensure that the memory kernel integrates to \\(\\eta\\) and that the kernel value at time zero is zero (for the smooth cutoff version). S and Z are computed automatically. Two maximum liklihood based inference methods are currently available: -  ApproxPowerlawHawkesProcessInference.estimate() maximizes the log-likelihood given a sample of arrival times. -  ApproxPowerlawHawkesProcessInference.estimate_grid() maximizes the log-likelihood on a grid of different starting values and returns the best model out of all fitted models (i.e. the model with the highest log-likelihood). The log-likelihood function of Hawkes processes can be very flat and a single optimization run may get stuck in a local supoptimal maximum. This class inherets from  HawkesPyLib.processes.UnivariateHawkesProcess and provides methods for further analysis of the fitted model: - ApproxPowerlawHawkesProcessInference.compensator() computes the compensator and may be used as a starting point for a 'residual' analysis. - ApproxPowerlawHawkesProcessInference.intensity() evaluates the estimated conditional intensity. - ApproxPowerlawHawkesProcessInference.kernel_values() returns values of the estimated memory kernel. - ApproxPowerlawHawkesProcessInference.ic() computes information criteria of the estimated model. Args: kernel (str): Must be one of: 'powlaw', 'powlaw-cutoff'. Specifies the shape of the approximate power-law kernel m (float): Specifies the approximation of the true power-law. Must be positive M (int): The number of weighted exponential kernels that specifies the approximation of the true power-law. Must be positive. rng (optional): numpy numba generator. For reproducible results use: np.random.default_rng(seed) Attributes: mu (float): The estimated background intensity. eta (float): The estimated branching ratio. alpha (float): Estimated tail exponent of the power-law decay. tau0 (float): Estimated timescale parameter. timestamps (np.ndarray): 1d array of arrival times. T: (float): End time of the process. logL (Float): The log-likelihood value of the fitted process"
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.estimate",
"url":9,
"doc":"Estimates the Hawkes process parameters using maximum likelihood estimation. Args: timestamps (np.ndarray): 1d array of arrival times. Must be sorted and only positive timestamps are allowed! T (float): The end time of the Hawkes process. return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0 max_attempts (int, optional): Number of times the maximum likelihood estimation repeats with new starting values if the optimization routine does not exit successfully. Defaults to 5. custom_param_vec0 (bool, optional): If custom initial values should be used. Defaults to False. If true you must supply an addtional variable 'param_vec0' a 1d numpy array containing the initial starting values: param_vec0 = [mu0 > 0, 0  0]. Returns: tuple: (optional) if  return_params =True returns tuple of fitted parameters: mu, eta, theta",
"func":1
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.estimate_grid",
"url":9,
"doc":"Estimates the Hawkes process parameters using maximum likelihood estimation. Fits the model multiple times using different initial parameter values. Subsequently the fitted model with the largest logL value is returned. The grid applied only to the initial value of eta in the mle routine, mu0 is set in using the unconditonal mean of the process and theta is always chosed randomly. There are three options for the type of eta0 starting value grid: - ('random') starting values for eta0 are chosen randomly. - ('equidistant') starting values for eta0 are equidistant between 0 and 1 - ('custom') A custom eta0 grid of starting value is supplied to 'custom-grid' variable: You must supply an addtional variable 'custom-grid' a 1d numpy array containing a grid of initial starting values with constraint 0 < eta0 < 1. Args: timestamps (np.ndarray): 1d array of arrival times. Must be sorted and only positive timestamps are allowed! T (float): The end time of the Hawkes process. grid_type (str, optional): Type of grid for eta0 starting values, one of: 'random', 'equidistant' or 'custom'. Default to 'equidistant' grid_size (int, optional): The number of optimizations to run. Defaults to 20. return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0 Returns: tuple: (optional) if  return_params =True returns tuple of fitted parameters: mu, eta, theta",
"func":1
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.get_params",
"url":9,
"doc":"Returns the fitted model parameters: Returns: tuple: Tuple of parameter values such as: -  mu (float),  eta (float),  alpha (float),  tau0 (float),  m (float),  M (int)",
"func":1
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.set_arrival_times",
"url":8,
"doc":"Set arrival times of the process. Args: timestamps (np.ndarray): 1d array of arrival times of the process. Arrival times must be positive and ordered. T (float): End time of the process. T must be larger or equal to the last arrival time.",
"func":1
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.intensity",
"url":8,
"doc":"Evaluates the intensity function \\(\\lambda(t)\\) on a grid of equidistant timestamps over the closed interval [0, T] with step size  step_size .  ! note - Additionally to the equidistant time grid the intensity is evaluated at each simulated arrival time. - If the process end time T is not perfectly divisible by the step size the 'effective' step size deviates slightly from the given one. Args: step_size (float): Step size of the time grid. Returns: np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)",
"func":1
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.compensator",
"url":8,
"doc":"Computes the compensator of the process at the set timestamps given the set parameter values. The compensator of a point process is defined as the integral of the conditional intensity:  \\Lambda(t) = \\int_0^t \\lambda(u) du   ! tip If the set arrival times are a realization of the Hawkes process specified by the set parameters, the resulting compensator will be a Poisson process with unit intensity (assuming a large enough sample size). Returns: np.ndarray: 1d array of timestamps (the compensator)",
"func":1
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.kernel_values",
"url":8,
"doc":"Returns the value of the memory kernel at given time values. The single exponential memory kernel:  g(t) = \\dfrac{\\eta}{\\theta} e^{(-t/\\theta)}  The P-sum exponential memory kernel:  g(t) = \\dfrac{\\eta}{P} \\sum_{k=1}^{P} \\dfrac{1}{\\theta_k} e^{(-t/\\theta_k)}  The approximate power-law memory kernel:  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} \\bigg]  And the approximate power-law kernel with smooth cutoff :  g(t) = \\dfrac{\\eta}{Z} \\bigg[ \\sum_{k=0}^{M-1} a_k^{-(1+\\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \\bigg]  Args: times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel. Returns: np.ndarray: 1d array containing the values of the memory kernel value at the given times.",
"func":1
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.compute_logL",
"url":8,
"doc":"Computes the log-likelihood function given the set parameters, arrival times and process end time T. The log-likelihood function of a Hawkes process is given by:  logL(t_1, ., t_n) = -\\int_0^T \\lambda(t) du + \\sum_{t=1}^n log(\\lambda(t_i  Returns: float: The log-likelihood value",
"func":1
},
{
"ref":"HawkesPyLib.inference.ApproxPowerlawHawkesProcessInference.ic",
"url":8,
"doc":"Calculates one of the following information criteria given the set parameters and arrival times: - 'aic': Akaike information criteria - 'bic': Baysian information criteria - 'hq': Hannan-Quinn information criteria Args: type (str): Which information criterion to return, must be one of: 'aic', 'bic', 'hq' Returns: float: The information criterions value",
"func":1
}
]