import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from HawkesPyLib.processes import UnivariateHawkesProcess
from HawkesPyLib.core.logll import (uvhp_approx_powl_logL,
                                    uvhp_approx_powl_cut_logL,
                                    uvhp_expo_logL,
                                    uvhp_expo_logL_grad,
                                    uvhp_sum_expo_logL,
                                    uvhp_sum_expo_logL_grad)
from HawkesPyLib.util import OneOf, IntInExRange
rng = np.random.default_rng()
__all__ = ["ExpHawkesProcessInference", "SumExpHawkesProcessInference", "ApproxPowerlawHawkesProcessInference"]


def uvhp_expo_mle(timestamps: np.ndarray, T: float, param_vec0: np.ndarray) -> list:
    """ Provides maximum liklihood estimation for the single exponential kernel.
        Uses scipy's l_bfgs_b optimization routine to minimize the negative log-likelihood.
        The gradient of the log-likelihood function is passed explicitly to the minimization routine.

        Reference:

        - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines
         for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.

    Args:
        timestamps (np.ndarray): 1d array of arrival times of the process. Arrival times must be positive and ordered.
        T (float): End time of the process. T must be larger or equal to the last arrival time.
        param_vec0 (np.ndarray): 1d array [mu, eta, theta] of initial parameter values passed to the optimization routine.

    Returns:
        list: A list of values: list[0] optimal paramters, list[1] min function value, list[2] optim info
            (also see scipy.optimize.fmin_l_bfgs_b for more info).
    """
    bnds = [(1e-8, np.inf), (1e-8, 1.0), (1e-8, np.inf)]
    opt_result = fmin_l_bfgs_b(func=uvhp_expo_logL, x0=param_vec0, fprime=uvhp_expo_logL_grad,
                               args=(timestamps, T), approx_grad=False, bounds=bnds,
                               m=10, factr=100, pgtol=1e-05, iprint=-1, maxfun=15000,
                               maxiter=15000, disp=None, callback=None, maxls=20)
    return opt_result


def uvhp_sum_expo_mle(timestamps: np.ndarray, T: float, P: int, param_vec0: np.ndarray) -> list:
    """ Provides maximum liklihood estimation for the P-sum exponential kernel.
        Uses scipy's l_bfgs_b optimization routine to minimize the negative log-likelihood.
        The gradient of the log-likelihood function is passed explicitly to the minimization routine.

    Reference:

        - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines
         for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.

    Args:
        timestamps (np.ndarray): 1d array of arrival times of the process. Arrival times must be positive and ordered.
        T (float): End time of the process. T must be larger or equal to the last arrival time.
        P (int): Number of exponentials in the the P-sum exponential kernel.
        param_vec0 (np.ndarray): 1d array [mu, eta, theta_1, theta_2, ..., theta_P] of starting parameter values for the optimization routine.

    Returns:
        list: A list of values: list[0] optimal paramters, list[1] min function value, list[2] optim info
            (also see scipy.optimize.fmin_l_bfgs_b for more info)
    """
    bnds = [(1e-8, np.inf), (1e-8, 1)]
    for k in range(0, P):
        bnds.append((1e-8, np.inf))
    opt_result = fmin_l_bfgs_b(func=uvhp_sum_expo_logL, x0=param_vec0, fprime=uvhp_sum_expo_logL_grad,
                               args=(timestamps, T), approx_grad=False, bounds=bnds,
                               m=10, factr=100, pgtol=1e-05, iprint=- 1, maxfun=15000,
                               maxiter=15000, disp=None, callback=None, maxls=20)
    return opt_result


def uvhp_powlaw_mle(timestamps: np.ndarray, T: float, m: float, M: int, param_vec0: np.ndarray) -> list:
    """Provides maximum liklihood estimation for the P-sum exponential kernel.
        Uses scipy's l_bfgs_b optimization routine to minimize the negative log-likelihood.
        The gradient of the log-likelihood function is approximated by the l_bfgs_b routine.

        Reference:

        - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN
         for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.

    Args:
        timestamps (np.ndarray): 1d array of arrival times of the process. Arrival times must be positive and ordered.
        T (float): End time of the process. T must be larger or equal to the last arrival time.
        m (float): Scale parameter of the power-law weights, m > 0.
        M (int): Number of weighted exponential kernels used in the approximate power-law kernel.
        param_vec0 (np.ndarray): 1d array [mu, eta, alpha, tau0] of initial parameter values passed to the optimization routine.

    Returns:
        list: A list of values: list[0] optimal paramters, list[1] min function value, list[2] optim info
            (also see scipy.optimize.fmin_l_bfgs_b for more info)
    """
    bnds = [(1e-8, np.inf), (1e-8, 1.0), (1e-8, 10), (1e-8, np.inf)]
    opt_result = fmin_l_bfgs_b(func=uvhp_approx_powl_logL, x0=param_vec0, fprime=None,
                               args=(timestamps, T, m, M), approx_grad=True, bounds=bnds,
                               m=10, factr=100, epsilon=1e-07, pgtol=1e-05, iprint=- 1,
                               maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
    return opt_result


def uvhp_powlaw_cut_mle(timestamps: np.ndarray, T: float, m: float, M: int, param_vec0: np.ndarray) -> list:
    """Provides maximum liklihood estimation for the P-sum exponential kernel.
        Uses scipy's l_bfgs_b optimization routine to minimize the negative log-likelihood.
        The gradient of the log-likelihood function is approximated by the l_bfgs_b routine.

        Reference:

        - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines
         for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.

    Args:
        timestamps (np.ndarray): 1d array of arrival times of the process. Arrival times must be positive and ordered.
        T (float): End time of the process. T must be larger or equal to the last arrival time.
        m (float): Scale parameter of the power-law weights, m > 0.
        M (int): Number of weighted exponential kernels used in the approximate power-law kernel.
        param_vec0 (np.ndarray): 1d array [mu, eta, alpha, tau0] of initial parameter values passed to the optimization routine.

    Returns:
        list: A list of values: list[0] optimal paramters, list[1] min function value, list[2] optim info
            (also see scipy.optimize.fmin_l_bfgs_b for more info)
    """
    bnds = [(1e-8, np.inf), (1e-8, 1.0), (1e-8, 10), (1e-8, np.inf)]
    opt_result = fmin_l_bfgs_b(func=uvhp_approx_powl_cut_logL, x0=param_vec0, fprime=None,
                               args=(timestamps, T, m, M), approx_grad=True, bounds=bnds,
                               m=10, factr=100, epsilon=1e-07, pgtol=1e-05, iprint=- 1,
                               maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
    return opt_result


class ExpHawkesProcessInference(UnivariateHawkesProcess):
    r""" Inference of unvivariate Hawkes processes with single exponentials memory kernel.
        Hawkes process with single exponential memory kernel ('expo' kernel), is defined by the conditional intensity function:

        \[\lambda(t) = \mu + \dfrac{\eta}{\theta} \sum_{t_i < t} e^{(-(t - t_i)/\theta)}\]

        Two maximum liklihood based inference methods are currently available:

         - `ExpHawkesProcessInference.estimate()` maximizes the log-likelihood given a sample of arrival times.

         - `ExpHawkesProcessInference.estimate_grid()` maximizes the log-likelihood on a grid of different starting values
            and returns the best model out of all fitted models (i.e. the model with the highest log-likelihood).
            The log-likelihood function of Hawkes processes can be very flat and a single optimization run may get stuck in a local
            supoptimal maximum.

        This class inherets from `HawkesPyLib.processes.UnivariateHawkesProcess` and provides methods for further analysis of the fitted model:

        - ExpHawkesProcessInference.compensator() computes the compensator and may be used as a starting point for a 'residual' analysis.
        - ExpHawkesProcessInference.intensity() evaluates the estimated conditional intensity.
        - ExpHawkesProcessInference.kernel_values() returns values of the estimated memory kernel.
        - ExpHawkesProcessInference.ic() computes information criteria of the estimated model.
    """
    _grid_type = OneOf("random", "equidistant", "custom", "no-grid")
    _grid_size = IntInExRange("grid_size", lower_bound=0)

    def __init__(self, rng=rng) -> None:
        """
        Args:
            rng (optional): numpy random number generator. For reproducible results use: rng=np.random.default_rng(seed)

        Attributes:
            mu (float): The estimated background intensity.
            eta (float): The estimated branching ratio.
            theta (float): Estimated decay speed of single exponential kernel, theta > 0.
            timestamps (np.ndarray): 1d array of arrival times that were used in the most recent fitting routine.
            T: (float): End time of the estimated process.
            logL (Float): The log-likelihood value of the fitted process.

        """
        self._rng = rng
        self._params_set = False
        self._kernel = "expo"
        self._num_par = 3

    def estimate(self, timestamps: np.ndarray, T: float, return_params: bool = False, max_attempts: int = 5,
                 custom_param_vec0: bool = False, **kwargs) -> None:
        """ Estimates the Hawkes process parameters using maximum likelihood estimation.

        Args:
            timestamps (np.ndarray): 1d array of arrival times.
                                     Must be sorted and only positive timestamps are allowed!
            T (float): The end time of the Hawkes process.
            return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0
            max_attempts (int, optional): Number of times the maximum likelihood estimation repeats with new starting values
                                            if the optimization routine does not exit successfully. Defaults to 5.
            custom_param_vec0 (bool, optional): If custom initial values should be used. Defaults to False.
                                                If true you must supply an addtional variable 'param_vec0' a 1d numpy array
                                                containing the initial starting values: param_vec0 = [mu0 > 0, 0 < eta0 <1, theta > 0].
        Returns:
            tuple: (optional) if `return_params`=True returns tuple of fitted parameters: mu, eta, theta
        """
        self.timestamps = timestamps
        self.T = self._check_valid_T(T, timestamps[-1])
        self._timestamps_set = True

        succ_flag = None
        if custom_param_vec0 is False:
            attempt = 0
            while (succ_flag != 0) & (attempt <= max_attempts):
                attempt += 1
                # TODO: Improve the standard starting values: eta0 using the model-free branching ratio estimator, theta0 ????
                eta0 = rng.uniform(1e-3, 0.999)
                mu0 = len(timestamps) * (1 - eta0) / self.T
                theta0 = rng.uniform(1e-5, 5)
                param_vec0 = np.array([mu0, eta0, theta0], dtype=np.float64)

                opt_result = uvhp_expo_mle(self.timestamps, T, param_vec0)
                succ_flag = opt_result[2]["warnflag"]  # 0 if convergence, 1 if too many func evals or iters, 2 other reason see ["task"]
        else:
            opt_result = uvhp_expo_mle(self.timestamps, self.T, kwargs.pop("param_vec0"))
            succ_flag = opt_result[2]["warnflag"]  # 0 if convergence, 1 if too many func evals or iters, 2 other reason see ["task"]

        self._grid_type = "no-grid"
        if succ_flag == 0:

            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.theta = result_param_vec[0], result_param_vec[1], result_param_vec[2]
            self._params_set = True

            if return_params is True:
                return self.mu, self.eta, self.theta

        else:
            print(f"WARNING: Optimization not successful: {opt_result[2]['task']}")

    def estimate_grid(self, timestamps: np.ndarray, T: float, grid_type: str = 'equidistant', grid_size: int = 20, return_params: bool = False, **kwargs):
        """ Estimates the Hawkes process parameters using maximum likelihood estimation. Fits the model multiple times using different
            initial parameter values. Subsequently the fitted model with the largest logL value is returned.

        The grid applied only to the initial value of eta in the mle routine,
            mu0 is set in using the unconditonal mean of the process and theta is always chosed randomly.

        There are three options for the type of eta0 starting value grid:

        - ('random') starting values for eta0 are chosen randomly.

        - ('equidistant') starting values for eta0 are equidistant between 0 and 1

        - ('custom') A custom eta0 grid of starting value is supplied to 'custom-grid' variable:
                       You must supply an addtional variable 'custom-grid' a 1d numpy array
                        containing a grid of initial starting values with constraint 0 < eta0 < 1.

        Args:
            timestamps (np.ndarray): 1d array of arrival times.
                                     Must be sorted and only positive timestamps are allowed!
            T (float): The end time of the Hawkes process.
            grid_type (str, optional): Type of grid for eta0 starting values, one of: 'random', 'equidistant' or 'custom'. Default to 'equidistant'
            grid_size (int, optional): The number of optimizations to run. Defaults to 20.
            return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0

        Returns:
            tuple: (optional) if `return_params`=True returns tuple of fitted parameters: mu, eta, theta
        """
        self.timestamps = timestamps
        self.T = self._check_valid_T(T, timestamps[-1])
        self._timestamps_set = True

        self._grid_type = grid_type
        self._grid_size = grid_size

        if grid_type == "random":
            eta0_grid = rng.uniform(0.001, 0.999, grid_size)
        elif grid_type == "equidistant":
            eta0_grid = np.linspace(0.001, 0.999, grid_size)
        elif grid_type == "custom":
            eta0_grid = kwargs.get("custom_grid")

        # store the optimization results
        opt_res_ls = []
        logL_res_ls = []
        for eta0 in eta0_grid:
            # TODO: Improve the standard starting values: eta0 using the model-free branching ratio estimator, theta0 ????
            mu0 = len(timestamps) * (1 - eta0) / T
            theta0 = rng.uniform(1e-5, 5)
            param_vec0 = np.array([mu0, eta0, theta0], dtype=np.float64)
            opt_result = uvhp_expo_mle(self.timestamps, T, param_vec0)

            if opt_result[2]["warnflag"] == 0:  # if optimization exits successfully: append result
                opt_res_ls.append(opt_result)
                logL_res_ls.append(-opt_result[1])

        if len(opt_res_ls) > 0:
            # choose model with largest logL
            idx = logL_res_ls.index(max(logL_res_ls))
            opt_result = opt_res_ls[idx]

            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.theta = result_param_vec[0], result_param_vec[1], result_param_vec[2]
            self._params_set = True

            if return_params is True:
                return self.mu, self.eta, self.theta

        else:
            print(f"WARNING: Grid Optimization not successful, none of the {grid_size} attempts exited succesfully")

    def set_params(self, mu: float, eta: float, theta: float) -> None:
        """ Set parameters manually.

        Args:
            mu (float): The background intensity, mu > 0
            eta (float): The branching ratio, 0 < eta > 1
            theta (float): 1d array of P decay speeds defining the single exponential memory kernel, theta > 0
        """
        # self.mu, self.eta, self.theta = mu, eta, theta
        # self._params_set = True
        print(" ERROR: Parameters cannot be set manually, please use class: UnivariateHawkesProcess")
        return None

    def get_params(self) -> tuple:
        """Returns the fitted model parameters:

        Returns:
            tuple: Tuple of parameter values such as:

        - `mu` (float), `eta` (float), `theta` (float)
        """
        return self.mu, self.eta, self.theta


class SumExpHawkesProcessInference(UnivariateHawkesProcess):
    r""" Inference of unvivariate Hawkes processes with P-sum exponentials memory kernel.
        The Hawkes process with P-sum exponential memory kernel ('sum-expo' kernel), is defined by the conditional intensity function:
        \[\lambda(t) = \mu + \dfrac{\eta}{P} \sum_{t_i < t} \sum_{k=1}^{P} \dfrac{1}{\theta_k} e^{(-(t - t_i)/\theta_k)}\]

        Two maximum liklihood based inference methods are currently available:

         - `ExpHawkesProcessInference.estimate()` maximizes the log-likelihood given a sample of arrival times.

         - `ExpHawkesProcessInference.estimate_grid()` maximizes the log-likelihood on a grid of different starting values
            and returns the best model out of all fitted models (i.e. the model with the highest log-likelihood).
            The log-likelihood function of Hawkes processes can be very flat and a single optimization run may get stuck in a local
            supoptimal maximum.

        This class inherets from `HawkesPyLib.processes.UnivariateHawkesProcess`
        and provides methods for further analysis of the fitted model:

        - SumExpHawkesProcessInference.compensator() computes the compensator and may be used as a starting point for a 'residual' analysis.
        - SumExpHawkesProcessInference.intensity() evaluates the estimated conditional intensity.
        - SumExpHawkesProcessInference.kernel_values() returns values of the estimated memory kernel.
        - SumExpHawkesProcessInference.ic() computes information criteria of the estimated model.

    """
    _P = IntInExRange("P", lower_bound=0)
    _grid_type = OneOf("random", "equidistant", "custom", "no-grid")
    _grid_size = IntInExRange("grid_size", lower_bound=0)

    def __init__(self, P: int, rng=rng) -> None:
        """
        Args:
            P (int): The number of exponentials that make up the P-sum exponential memory kernel.
            rng (optional): numpy random generator. For reproducible results use: rng=np.random.default_rng(seed)

        Attributes:
            mu (float): The estimated background intensity.
            eta (float): The estimated branching ratio.
            theta_vec (np.ndarray, optional): Estimated decay speeds of P-sum, exponential kernel.
            timestamps (np.ndarray): 1d array of arrival times that were used in the most recent fitting routine.
            T (float): End time of the Hawkes process.
            logL (Float): The log-likelihood value of the fitted process.

        """
        self._P = P
        self._kernel = "sum-expo"
        self._rng = rng
        self._params_set = False
        self._num_par = P + 2

    def estimate(self, timestamps: np.ndarray, T: float, return_params: bool = False, max_attempts: int = 5, custom_param_vec0:  bool = False, **kwargs):
        """ Estimates the Hawkes process parameters using maximum likelihood estimation.

        Args:
            timestamps (np.ndarray): 1d array of arrival times.
                                     Must be sorted and only positive timestamps are allowed!
            T (float): The end time of the Hawkes process.
            return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0
            max_attempts (int, optional): Number of times the maximum likelihood estimation repeats with new starting values
                                            if the optimization routine does not exit successfully. Defaults to 5.
            custom_param_vec0 (bool, optional): If custom initial values should be used. Defaults to False.
                                                If true you must supply an addtional variable 'param_vec0' a 1d numpy array
                                                containing the initial starting values: param_vec0 = [mu0 > 0, 0 < eta0 <1, theta > 0].
        Returns:
            tuple: (optional) if `return_params`=True returns tuple of fitted parameters: mu, eta, theta
        """
        self.timestamps = timestamps
        self.T = self._check_valid_T(T, timestamps[-1])
        self._timestamps_set = True

        succ_flag = None
        if custom_param_vec0 is False:
            attempt = 0
            while (succ_flag != 0) & (attempt <= max_attempts):
                attempt += 1
                # TODO: Improve the standard starting values: eta0 using the model-free branching ratio estimator, theta0 ????
                eta0 = rng.uniform(1e-3, 0.999)
                mu0 = len(timestamps) * (1 - eta0) / T
                theta0_vec = rng.uniform(1e-5, 5, self._P)
                param_vec0 = np.append(np.append(mu0, eta0), theta0_vec)

                opt_result = uvhp_sum_expo_mle(self.timestamps, T, self._P, param_vec0)
                succ_flag = opt_result[2]["warnflag"]  # 0 if convergence, 1 if too many func evals or iters, 2 other reason see ["task"]
        else:
            opt_result = uvhp_sum_expo_mle(self.timestamps, T, self._P, kwargs.get("param_vec0"))
            succ_flag = opt_result[2]["warnflag"]

        self._grid_type = "no-grid"
        if succ_flag == 0:

            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.theta_vec = result_param_vec[0], result_param_vec[1], np.sort(result_param_vec[2::])
            self._params_set = True

            if return_params is True:
                return self.mu, self.eta, self.theta_vec

        else:
            print(f"WARNING: Optimization not successful: {opt_result[2]['task']}")

    def estimate_grid(self, timestamps: np.ndarray, T: float, grid_type: str = 'equidistant',
                      grid_size: int = 20, return_params: bool = False, rng=rng, **kwargs):
        """ Estimates the Hawkes process parameters using maximum likelihood estimation.
            Fits the model multiple times using different
            initial parameter values. Subsequently the fitted model with the largest logL value is returned.

        The grid applied only to the initial value of eta in the mle routine,
            mu0 is set in using the unconditonal mean of the process and theta is always chosed randomly.

        There are three options for the type of eta0 starting value grid:

        - ('random') starting values for eta0 are chosen randomly.

        - ('equidistant') starting values for eta0 are equidistant between 0 and 1

        - ('custom') A custom eta0 grid of starting value is supplied to 'custom-grid' variable:
                       You must supply an addtional variable 'custom-grid' a 1d numpy array
                        containing a grid of initial starting values with constraint 0 < eta0 < 1.

        Args:
            timestamps (np.ndarray): 1d array of arrival times.
                                     Must be sorted and only positive timestamps are allowed!
            T (float): The end time of the Hawkes process.
            grid_type (str, optional): Type of grid for eta0 starting values, one of: 'random', 'equidistant' or 'custom'. Default to 'equidistant'
            grid_size (int, optional): The number of optimizations to run. Defaults to 20.
            return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0

        Returns:
            tuple: (optional) if `return_params`=True returns tuple of fitted parameters: mu, eta, theta
        """
        self.timestamps = timestamps
        self.T = self._check_valid_T(T, timestamps[-1])
        self._timestamps_set = True

        self._grid_type = grid_type
        self._grid_size = grid_size

        if grid_type == "random":
            eta0_grid = rng.uniform(0.001, 0.999, grid_size)
        elif grid_type == "equidistant":
            eta0_grid = np.linspace(0.001, 0.999, grid_size)
        elif grid_type == "custom":
            eta0_grid = kwargs.get("custom_grid")

        # store the optimization results
        opt_res_ls = []
        logL_res_ls = []
        for eta0 in eta0_grid:
            # TODO: Improve the standard starting values: eta0 using the model-free branching ratio estimator, theta0 ????
            mu0 = len(timestamps) * (1 - eta0) / T
            theta0_vec = rng.uniform(1e-5, 5, self._P)
            param_vec0 = np.append(np.append(mu0, eta0), theta0_vec)
            opt_result = uvhp_sum_expo_mle(self.timestamps, T, self._P, param_vec0)

            if opt_result[2]["warnflag"] == 0:  # if optimization exits successfully: append result
                opt_res_ls.append(opt_result)
                logL_res_ls.append(-opt_result[1])

        if len(opt_res_ls) > 0:
            # choose model with largest logL
            idx = logL_res_ls.index(max(logL_res_ls))
            opt_result = opt_res_ls[idx]

            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.theta_vec = result_param_vec[0], result_param_vec[1], np.sort(result_param_vec[2::])
            self._params_set = True

            if return_params is True:
                return self.mu, self.eta, self.theta_vec

        else:
            print(f"WARNING: Grid Optimization not successful, none of the {grid_size} attempts exited succesfully")

    def get_params(self) -> tuple:
        """Returns the fitted model parameters:

        Returns:
            tuple: Tuple of parameter values such as:

        - `mu` (float), `eta` (float), `theta_vec` (np.ndarray)
        """

        return self.mu, self.eta, self.theta_vec

    def set_params(self, mu: float, eta: float, theta_vec: np.ndarray) -> None:
        """ Set parameters, timestamps and the end time T of the model manually.

        Args:
            mu (float): The background intensity, mu > 0
            eta (float): The branching ratio, 0 < eta > 1
            theta_vec (np.ndarray): 1d array of P decay speeds defining the P-sum exponential memory kernel, theta_k > 0
        """
        # self.mu, self.eta, self.theta_vec = mu, eta, theta_vec
        # self._num_par = 2 + len(theta_vec)
        # self._params_set = True
        print(" ERROR: Parameters cannot be set manually, please use class: UnivariateHawkesProcess")
        return None


class ApproxPowerlawHawkesProcessInference(UnivariateHawkesProcess):
    r""" Inference of unvivariate Hawkes processes with approximate power-law memory kernel.
        The Hawkes process with approximate power-law kernel ('powlaw' `kernel`), is defined by the conditional intenstiy fnuction:
        \[\lambda(t) = \mu + \sum_{t_i < t}  \dfrac{\eta}{Z} \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} \bigg],\]

        where \(\mu\) (`mu`) is the constant background intensity, \(\eta \) (`eta`)
        is the branching ratio, \(\alpha\) (`alpha`) is the power-law tail exponent and \(\tau_0\) a scale parameter controlling the decay timescale.

        The Hawkes process with approximate power-law kernel with smooth cutoff ('powlaw-cutoff' `kernel`), defined by the conditional intenstiy fnuction:
        \[\lambda(t) = \mu + \sum_{t_i < t} \dfrac{\eta}{Z} \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \bigg],\]

        where \(\mu\) (`mu`) is the constant background intensity, \(\eta \) (`eta`)
        is the branching ratio, \(\alpha\) (`alpha`) is the power-law tail exponent and \(\tau_0\) a scale parameter controlling the decay timescale and
        the location of the smooth cutoff point (i.e. the time duration after each jump at which the intensity reaches a maximum).

        The true power-law is approximtated by a sum of \(M\) exponential function with power-law weights
        \(a_k = \tau_0 m^k\). \(M\) is the number of exponentials used for the approximation and \(m\) is a scale
        parameter. The power-law approximation holds for times up to \(m^{M-1}\) after which the memory kernel
        decays exponentially. S and Z are scaling factors that ensure that the memory kernel integrates to \(\eta\)
        and that the kernel value at time zero is zero (for the smooth cutoff version). S and Z are computed automatically.

        Two maximum liklihood based inference methods are currently available:

         - `ApproxPowerlawHawkesProcessInference.estimate()` maximizes the log-likelihood given a sample of arrival times.

         - `ApproxPowerlawHawkesProcessInference.estimate_grid()` maximizes the log-likelihood on a grid of different starting values
            and returns the best model out of all fitted models (i.e. the model with the highest log-likelihood).
            The log-likelihood function of Hawkes processes can be very flat and a single optimization run may get stuck in a local
            supoptimal maximum.

        This class inherets from `HawkesPyLib.processes.UnivariateHawkesProcess` and provides methods for further analysis of the fitted model:

        - ApproxPowerlawHawkesProcessInference.compensator() computes the compensator and may be used as a starting point for a 'residual' analysis.
        - ApproxPowerlawHawkesProcessInference.intensity() evaluates the estimated conditional intensity.
        - ApproxPowerlawHawkesProcessInference.kernel_values() returns values of the estimated memory kernel.
        - ApproxPowerlawHawkesProcessInference.ic() computes information criteria of the estimated model.
    """
    _kernel = OneOf("powlaw", "powlaw-cutoff")
    _grid_type = OneOf("random", "equidistant", "custom", "no-grid")
    _grid_size = IntInExRange("grid_size", lower_bound=0)

    def __init__(self, kernel: str, m: float, M: int, rng=rng) -> None:
        """
        Args:
            kernel (str): Must be one of: 'powlaw', 'powlaw-cutoff'. Specifies the shape of the approximate power-law kernel
            m (float): Specifies the approximation of the true power-law. Must be positive
            M (int): The number of weighted exponential kernels that specifies the approximation of the true power-law. Must be positive.
            rng (optional): numpy numba generator. For reproducible results use: np.random.default_rng(seed)

        Attributes:
            mu (float): The estimated background intensity.
            eta (float): The estimated branching ratio.
            alpha (float): Estimated tail exponent of the power-law decay.
            tau0 (float): Estimated timescale parameter.
            timestamps (np.ndarray): 1d array of arrival times.
            T: (float): End time of the process.
            logL (Float): The log-likelihood value of the fitted process
        """
        self._m = m
        self._M = M
        self._kernel = kernel
        self._rng = rng
        self._params_set = False
        self._num_par = 4
        self._timestamps_set = False

    def estimate(self, timestamps: np.ndarray, T: float, return_params: bool = False, max_attempts: int = 5, custom_param_vec0: bool = False, **kwargs) -> None:
        """ Estimates the Hawkes process parameters using maximum likelihood estimation.

        Args:
            timestamps (np.ndarray): 1d array of arrival times.
                                     Must be sorted and only positive timestamps are allowed!
            T (float): The end time of the Hawkes process.
            return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0
            max_attempts (int, optional): Number of times the maximum likelihood estimation repeats with new starting values
                                            if the optimization routine does not exit successfully. Defaults to 5.
            custom_param_vec0 (bool, optional): If custom initial values should be used. Defaults to False.
                                                If true you must supply an addtional variable 'param_vec0' a 1d numpy array
                                                containing the initial starting values: param_vec0 = [mu0 > 0, 0 < eta0 <1, theta > 0].
        Returns:
            tuple: (optional) if `return_params`=True returns tuple of fitted parameters: mu, eta, theta
        """
        self.timestamps = timestamps
        self.T = self._check_valid_T(T, timestamps[-1])
        self._timestamps_set = True

        succ_flag = None

        if custom_param_vec0 is False:
            attempt = 0
            while (succ_flag != 0) & (attempt <= max_attempts):
                attempt += 1
                # TODO: Improve the standard starting values: eta0 using the model-free branching ratio estimator, alpha0/tau00 ????
                eta0 = rng.uniform(1e-3, 0.999)
                mu0 = len(timestamps) * (1 - eta0) / T
                alpha0 = rng.uniform(1e-3, 3)
                tau00 = rng.uniform(1e-7, 3)
                param_vec0 = np.array([mu0, eta0, alpha0, tau00], dtype=np.float64)

                if self._kernel == "powlaw":
                    opt_result = uvhp_powlaw_mle(timestamps, T, self._m, self._M, param_vec0)
                elif self._kernel == "powlaw-cutoff":
                    opt_result = uvhp_powlaw_cut_mle(timestamps, T, self._m, self._M, param_vec0)

                succ_flag = opt_result[2]["warnflag"]  # 0 if convergence, 1 if too many func evals or iters, 2 other reason see ["task"]
        else:
            if self._kernel == "powlaw":
                opt_result = uvhp_powlaw_mle(self.timestamps, self.T, self._m, self._M, kwargs.get("param_vec0"))
            elif self._kernel == "powlaw-cutoff":
                opt_result = uvhp_powlaw_cut_mle(self.timestamps, self.T, self._m, self._M, kwargs.get("param_vec0"))
            succ_flag = opt_result[2]["warnflag"]

        self._grid_type = "no-grid"

        if succ_flag == 0:

            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.alpha, self.tau0 = result_param_vec[0], result_param_vec[1], result_param_vec[2], result_param_vec[3]
            self._params_set = True

            if return_params is True:
                return self.mu, self.eta, self.alpha, self.tau0

        else:
            print(f"WARNING: Optimization not successful: {opt_result[2]['task']}")

    def estimate_grid(self, timestamps: np.ndarray, T: float, grid_type: str = 'equidistant',
                      grid_size: int = 20, return_params: bool = False, rng=rng, **kwargs) -> None:
        """ Estimates the Hawkes process parameters using maximum likelihood estimation.
            Fits the model multiple times using different
            initial parameter values. Subsequently the fitted model with the largest logL value is returned.

        The grid applied only to the initial value of eta in the mle routine,
            mu0 is set in using the unconditonal mean of the process and theta is always chosed randomly.

        There are three options for the type of eta0 starting value grid:

        - ('random') starting values for eta0 are chosen randomly.

        - ('equidistant') starting values for eta0 are equidistant between 0 and 1

        - ('custom') A custom eta0 grid of starting value is supplied to 'custom-grid' variable:
                       You must supply an addtional variable 'custom-grid' a 1d numpy array
                        containing a grid of initial starting values with constraint 0 < eta0 < 1.

        Args:
            timestamps (np.ndarray): 1d array of arrival times.
                                     Must be sorted and only positive timestamps are allowed!
            T (float): The end time of the Hawkes process.
            grid_type (str, optional): Type of grid for eta0 starting values, one of: 'random', 'equidistant' or 'custom'. Default to 'equidistant'
            grid_size (int, optional): The number of optimizations to run. Defaults to 20.
            return_params (bool, optional): If True returns a tuple of fitted parameters in the order: mu, eta, alpha, tau0

        Returns:
            tuple: (optional) if `return_params`=True returns tuple of fitted parameters: mu, eta, theta
        """
        self.timestamps = timestamps
        self.T = self._check_valid_T(T, timestamps[-1])
        self._timestamps_set = True

        self._grid_type = grid_type
        self._grid_size = grid_size

        if grid_type == "random":
            eta0_grid = rng.uniform(0.001, 0.999, grid_size)
        elif grid_type == "equidistant":
            eta0_grid = np.linspace(0.001, 0.999, grid_size)
        elif grid_type == "custom":
            eta0_grid = kwargs.get("custom_grid")

        # TODO: Allow parrallel optimization over the different starting values
        opt_res_ls = []
        logL_res_ls = []
        for eta0 in eta0_grid:
            mu0 = len(timestamps) * (1 - eta0) / T
            alpha0 = rng.uniform(1e-3, 3)
            tau00 = rng.uniform(1e-7, 3)
            param_vec0 = np.array([mu0, eta0, alpha0, tau00], dtype=np.float64)

            if self._kernel == "powlaw":
                opt_result = uvhp_powlaw_mle(timestamps, T, self._m, self._M, param_vec0)
            elif self._kernel == "powlaw-cutoff":
                opt_result = uvhp_powlaw_cut_mle(timestamps, T, self._m, self._M, param_vec0)

            if opt_result[2]["warnflag"] == 0:   # if optimization exits successfully: append result
                opt_res_ls.append(opt_result)
                logL_res_ls.append(-opt_result[1])

        if len(opt_res_ls) > 0:

            # choose model with largest logL
            idx = logL_res_ls.index(max(logL_res_ls))
            opt_result = opt_res_ls[idx]

            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.alpha, self.tau0 = result_param_vec[0], result_param_vec[1], result_param_vec[2], result_param_vec[3]
            self._params_set = True

            if return_params is True:
                return self.mu, self.eta, self.alpha, self.tau0
        else:
            print(f"WARNING: Grid Optimization not successful, none of the {grid_size} attempts exited succesfully")

    def set_params(self, mu: float, eta: float, alpha: float, tau0: float) -> None:
        """ Set parameters, timestamps and the end time T of the model manually.

        Args:
            mu (float): The background intensity, mu > 0
            eta (float): The branching ratio, 0 < eta > 1
            alpha (float): The power-law coefficient of the approximate power-law memory kernel, alpha > 0
            tau0 (float): Defines the decay speed of the memory kernel, tau0 > 0
        """
        # self.mu, self.eta, self.alpha, self.tau0 = mu, eta, alpha, tau0

        # self._params_set = True
        print(" ERROR: Parameters cannot be set manually, please use class: UnivariateHawkesProcess")

        return None

    def get_params(self) -> tuple:
        """Returns the fitted model parameters:

        Returns:
            tuple: Tuple of parameter values such as:

        - `mu` (float), `eta` (float), `alpha` (float), `tau0` (float), `m` (float), `M` (int)

        """
        return self.mu, self.eta, self.alpha, self.tau0, self._m, self._M
