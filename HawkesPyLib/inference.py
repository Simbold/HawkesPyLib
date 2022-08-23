import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from HawkesPyLib.num.logll import (uvhp_approx_powl_logL,
                                    uvhp_approx_powl_cut_logL,
                                    uvhp_expo_logL,
                                    uvhp_expo_logL_grad,
                                    uvhp_sum_expo_logL,
                                    uvhp_sum_expo_logL_grad)
from HawkesPyLib.num.comp import (uvhp_approx_powl_compensator,
                                    uvhp_approx_powl_cut_compensator,
                                    uvhp_expo_compensator,
                                    uvhp_sum_expo_compensator)
from HawkesPyLib.util import (OneOf,
                FloatInExRange,
                IntInExRange,
                PositiveOrderedFloatNdarray)


rng = np.random.default_rng()

def uvhp_expo_mle(timestamps: np.ndarray, T: float, param_vec0: np.ndarray):
    """ Minimizes the negative log-likelihood of a univariate Hawkes process with single normalised exponential kernel

    Args:
        timestamps (np.ndarray): 1 dimensional Array of timestamps. All timestamps must be positive and sorted in ascending order.
        T (float): End time of the process. Must be equal or larger than the largest/last timestamps in 'timestamps'
        param_vec0 (np.ndarray): Array of initial parameter values for the optimization routine

    Returns:
        list: A list of values: list[0] optimal paramters, list[1] min function value, list[2] optim info 
            (also see scipy.optimize.fmin_l_bfgs_b for more info) 
    """
    bnds = [(1e-8, np.inf), (1e-8, 1.0), (1e-8, np.inf)]
    opt_result = fmin_l_bfgs_b(func=uvhp_expo_logL, x0=param_vec0, fprime=uvhp_expo_logL_grad,
                        args=(timestamps, T), approx_grad=False, bounds=bnds,
                        m=10, factr=100, pgtol=1e-05, iprint=-1, maxfun=15000,
                        maxiter=15000, disp=None, callback=None, maxls=20) 
    return opt_result


def uvhp_sum_expo_mle(timestamps: np.ndarray, T: float, P: int, param_vec0: np.ndarray):
    """Minimizes the negative log-likelihood of a univariate Hawkes process with sum of P exponential kernels"

    Args:
        timestamps (np.ndarray): 1 dimensional Array of timestamps. All timestamps must be positive and sorted in ascending order.
        T (float): End time of the process. Must be equal or larger than the largest/last timestamps in 'timestamps'
        P (int): Number of normalised exponential kernels that specify the Hawkes kernel
        param_vec0 (np.ndarray): Array of initial parameter values for the optimization routine

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


def uvhp_powlaw_mle(timestamps: np.ndarray, T: float, m: float, M: int, param_vec0: np.ndarray):
    """Minimizes the negative log-likelihood of a univariate Hawkes process with approximate power-law kernel without cutoff

    Args:
        timestamps (np.ndarray): 1 dimensional Array of timestamps. All timestamps must be positive and sorted in ascending order.
        T (float): End time of the process. Must be equal or larger than the largest/last timestamps in 'timestamps'
        m (float): Fixed paramter of the Hawkes kernel. Specifies the approximation of the true power-law. Must be positive.
        M (int): Fixed paramter of the Hawkes kernel. Specifies the approximation of the true power-law. Must be positive
        param_vec0 (np.ndarray): Array of initial parameter values for the optimization routine

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


def uvhp_powlaw_cut_mle(timestamps: np.ndarray, T: float, m: float, M: int, param_vec0: np.ndarray):
    """ Minimizes the negative log-likelihood of a univariate Hawkes process with approximate power-law kernel with cutoff

    Args:
        timestamps (np.ndarray): 1 dimensional Array of timestamps. All timestamps must be positive and sorted in ascending order.
        T (float): End time of the process. Must be equal or larger than the largest/last timestamps in 'timestamps'
        m (float): Fixed paramter of the Hawkes kernel. Specifies the approximation of the true power-law. Must be positive.
        M (int): Fixed paramter of the Hawkes kernel. Specifies the approximation of the true power-law. Must be positive
        param_vec0 (np.ndarray): Array of initial parameter values for the optimization routine

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


def compute_ic(logL_value: float, num_par: int) -> tuple:
    """ Computes AIC, BIC and HQ information criteria 

    Args:
        logL_value (float): The log-likelihood value of an estimated model
        num_par (int): The number of free paramters in the estimated model

    Returns:
        tuple: aic, bic, hq
    """
    aic = -2 * logL_value + 2 * num_par
    bic = -2 * logL_value + num_par * np.log(num_par)
    hq = -2 * logL_value + 2 * num_par * np.log(np.log(num_par))
    return aic, bic, hq
    

def qq_plot(x: np.ndarray, F_inv) -> np.ndarray:
    """ Plots a QQ-plot

    Args:
        x (np.ndarray): Array of values for which to plot the QQ-plot
        F_inv function: Quantile function of the theoretical distribution. Calls like: F_inv(x: np.ndarray)

    Returns:
        _type_: Plots the QQ-plot
    """
    n = len(x)
    x_sorted = np.flip(np.sort(x))
    F_th = F_inv((n-np.arange(1, n+1)+1)/(n+1))  # np.linspace(n/(n+1), 1/(n+1), num= n)
    plt.plot(F_th, x_sorted, 'ob')
    plt.ylabel("empirical quantile")
    plt.xlabel("theoretical quantile")

    plt.plot(x_sorted, x_sorted)
    plt.show()
    return F_th


def exp1_inv(x: np.ndarray) -> float:
    """ Quantile function of the unit exponential distribution

    Args:
        x (np.ndarray): array of in ascending order sorted values 

    Returns:
        np.ndarray: Quantile function evaluated at each value of x
    """
    return -np.log(1-x)


# TODO: add option to have log scale qq-plot
def qq_plot_unit_exponential(inter_arrival_times: np.ndarray):
    """ Convenience function to plot a QQ-plot for inter-arrival / duration times
        If a sample of temporal point process is a realization of a unit Poisson process.
        It's inter-arrival or also duration times are exponentially distributed.  

    Args:
        inter_arrival_times (np.ndarray): Array of inter-arrival times 

    Returns:
        Plot: Plots a QQ-plot
    """
    qq_plot(inter_arrival_times, exp1_inv)
    return None

class HawkesProcessEstimation():
    """
    Abstract Hawkes process estimation class defining method common to all estimation classes.
    """

class ApproxPowlawHawkesProcessEstimation():
    """Class for inference of unvivariate Hawkes processes with approximate powerlaw memory kernel. The conditional intensity
    function for the kernel without cutoff is defined as:
    .. math::
        \lambda^*(t) = \mu + \sum_{t_i < t}  \dfrac{\\eta}{Z} \\bigg[ \sum_{k=0}^{M-1} \\alpha_k^{-(1+\\alpha)} \exp(-(t - t_i)/\\alpha_k) \\bigg],
    where :math:`\mu` ("mu") is a constant background intensity, :math:`\\eta` ("eta")
    is the branching ratio and :math:\\alpha_k = \\tau_0 m^k the k'th powerlaw weight. S and Z are scaling factors and computed automatically. 
    M and m define the accuracy of the power-law approximation.
    The intensity for the kernel with cutoff is given by:
        .. math::
        \lambda^*(t) = \mu + \sum_{t_i < t}  \dfrac{\\eta}{Z} \\bigg[ \sum_{k=0}^{M-1} \\alpha_k^{-(1+\\alpha)} \exp(-(t - t_i)/\\alpha_k) - S \exp(-(t - t_i)/\\talpha_{-1}) \\bigg],
   
   The estimation is done by numerically maximizing the corresponding log-likelihood function.

   Attributes:
        success_flag (bool): True if process successfully estimated and the following attributes set.
        mu (float): The estimated background intensity :math:`\mu`.
        eta (float): The estimated branching ratio :math:`\\eta`.
        alpha (float): The estimated power-law coeficient, influencing the decay speed.
        tau0 (float): The estimated kernel paramter influencing decay speed and the location of the cutoff
        logL (Float): The log-likelihood value of the estimated process
    """
    _m = FloatInExRange("m", lower_bound=0)
    _M = IntInExRange("M", lower_bound=0)
    _kernel = OneOf("powlaw", "powlaw-cutoff")
    _grid_type = OneOf("random", "equidistant", "custom", "no-grid")
    _timestamps = PositiveOrderedFloatNdarray("timestamps")
    _grid_size = IntInExRange("grid_size", lower_bound=0)

    def __init__(self, kernel: str, m: float, M: int, rng=rng) -> None:
        """ Initlizes the ApproxPowlawHawkesProcessEstimation class

        Args:
            kernel (str): Must be one of: 'powlaw', 'powlaw-cutoff'. Specifies the shape of the approximate power-law kernel
            m (float): Specifies the approximation of the true power-law. Must be positive 
            M (int): The number of weighted exponential kernels that specifies the approximation of the true power-law. Must be positive.
            rng (optional): numpy numba generator. For reproducible results use: np.random.default_rng(seed)
        """
        self._m = m
        self._M = M
        self._kernel = kernel
        self._rng = rng
        self.success_flag = False
        self._num_par = 4

    def _check_valid_T(self, T: float, min_value) -> float:
        """Checks if input 'T' is valid

        Args:
            T (float): End time of the process
            min_value (float): The minimum value for the end time of the process. i.e. the largest/last timestamps in 'timestamps'

        Raises:
            ValueError: If T is invalid

        Returns:
            float: process end time 'T' 
        """
        if (T < min_value):
            raise ValueError(f"The process end time 'T' must be a number larger or equal to the last value in the sorted array 'timestamps'")
        else:
            return float(T)
    
    def aic(self) -> float:
        """Returns the Akaike information criteria of the estimated model

        Raises:
            Exception: If process has not been succesfully estimated yet. I.e. if 'success_flag' is False

        Returns:
            float: The AIC value
        """
        if self.success_flag == True:
            return -2 * self.logL + 2 * self._num_par
        else:
            raise Exception("ERROR: Cannot compute AIC, model has not been succesfully estimated!")

    def bic(self) -> float:
        """Returns the Baysian information criteria of the estimated model

        Raises:
            Exception: If process has not been succesfully estimated yet. I.e. if 'success_flag' is False

        Returns:
            float: The BIC value
        """
        if self.success_flag == True:
            return -2 * self.logL + self._num_par * np.log(self._num_par)
        else:
            raise Exception("ERROR: Cannot compute BIC, model has not been succesfully estimated!")

    def hq(self) -> float:
        """Returns the Hannan-Quinn  information criteria of the estimated model

        Raises:
            Exception: If process has not been succesfully estimated yet. I.e. if 'success_flag' is False

        Returns:
            float: The HQ value
        """
        if self.success_flag == True:
            return -2 * self.logL + 2 * self._num_par * np.log(np.log(self._num_par))
        else:
            raise Exception("ERROR: Cannot compute HQ, model has not been succesfully estimated!")
        
    def estimate(self, timestamps: np.ndarray, T: float, max_attempts: int=5, custom_param_vec0: bool=False, **kwargs) -> None: 
        """ Method for estimating the Hawkes process parameters using maximum likelihood estimation.

        Args:
            timestamps (np.ndarray): 1d numpy array containing the timestamp observations. 
                                     Must be sorted and only positive timestamps are allowed
            T (float): The end time of the Hawkes process.
            max_attempts (int, optional): If the optimization routine does not exit successfully. Number of times the optimization repeats with new starting values. Defaults to 5.
            custom_param_vec0 (bool, optional): If custom initial values should be used. Defaults to False. 
                                                If true you must supply an addtional variable 'custom_grid' containing a npumpy array of correct dimensions and bound.
        """
        self._timestamps = timestamps
        self._T = self._check_valid_T(T, timestamps[-1])

        succ_flag = None

        if custom_param_vec0 == False:
            attempt = 0
            while (succ_flag != 0) & (attempt <= max_attempts):
                attempt += 1
                # TODO: Improve the standard starting values: eta0 using the model-free branching ratio estimator, alpha0/tau00 ????
                eta0 = rng.uniform(1e-3, 0.999)
                mu0 = len(timestamps) * (1 - eta0) / self._T
                alpha0 = rng.uniform(1e-3, 3)
                tau00 = rng.uniform(1e-7, 3)
                param_vec0 = np.array([mu0, eta0, alpha0, tau00], dtype=np.float64)
                
                if self._kernel == "powlaw":
                    opt_result = uvhp_powlaw_mle(self._timestamps, self._T, self._m, self._M, param_vec0)
                elif self._kernel == "powlaw-cutoff":
                    opt_result = uvhp_powlaw_cut_mle(self._timestamps, self._T, self._m, self._M, param_vec0)

                succ_flag = opt_result[2]["warnflag"] # 0 if convergence, 1 if too many func evals or iters, 2 other reason see ["task"]
        else:
            if self._kernel == "powlaw":
                opt_result = uvhp_powlaw_mle(self._timestamps, self._T, self._m, self._M, kwargs.get("param_vec0"))
            elif self._kernel == "powlaw-cutoff":
                opt_result = uvhp_powlaw_cut_mle(self._timestamps, self._T, self._m, self._M, kwargs.get("param_vec0"))
            succ_flag = opt_result[2]["warnflag"]

        self._opt_result = opt_result
        self._grid_type = "no-grid"    
        if succ_flag == 0:
            self.success_flag = True
            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.alpha, self.tau0 = result_param_vec[0], result_param_vec[1], result_param_vec[2], result_param_vec[3]

        else:
            print(f"WARNING: Optimization not successful: {opt_result[2]['task']}")
        
        
    def estimate_grid(self, timestamps: np.ndarray, T: float, grid_type: str, grid_size: int=20, **kwargs):
        """ Method for performing multiple optimizations using different initial values
            The estimated model is chosen to be the one with the largest log-likelihood value.

        Args:
            timestamps (np.ndarray): 1d numpy array containing the timestamp observations. 
                                     Must be sorted and only positive timestamps are allowed
            T (float): The end time of the Hawkes process.
            grid_type (str): One of: ('random') starting values for eta0 are chosen randomly.
                                     ('equidistant') starting values for eta0 are equidistant between 0 and 1
                                     ('custom') A custom eta0 grid of starting value is supplied to 'custom-grid' variable
            grid_size (int, optional): The number of optimizations from which the best model is chosen. Defaults to 20.
        """
        self._timestamps = timestamps
        T = self._check_valid_T(T, timestamps[-1])
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
                opt_result = uvhp_powlaw_mle(self._timestamps, T, self._m, self._M, param_vec0)
            elif self._kernel == "powlaw-cutoff":
                opt_result = uvhp_powlaw_cut_mle(self._timestamps, T, self._m, self._M, param_vec0)

            if opt_result[2]["warnflag"] == 0: # if optimization exits successfully: append result
                opt_res_ls.append(opt_result)
                logL_res_ls.append(-opt_result[1])

        self._opt_result = opt_result   
        if len(opt_res_ls) > 0:
            # choose model with largest logL
            idx = logL_res_ls.index(max(logL_res_ls))
            opt_result = opt_res_ls[idx]   
        
            self.success_flag = True
            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.alpha, self.tau0 = result_param_vec[0], result_param_vec[1], result_param_vec[2], result_param_vec[3]

        else:
            print(f"WARNING: Grid Optimization not successful, none of the {grid_size} attempts exited succesfully")  

    def compensator(self) -> np.ndarray:
        """ Computes the compensator corresponding to the estimated process. 
            The compensator of a point process is defined as:

        Raises:
            Exception: If model has not yet been successfully estimated yet

        Returns:
            np.ndarray: numpy array of timestamps (the compensator)
        """
        if self.success_flag == False:
            raise Exception("ERROR: Cannot compute compensator, model has not been succesfully estimated!")

        if self._kernel == "powlaw":
            compensator = uvhp_approx_powl_compensator(self._timestamps, self.mu, self.eta, self.alpha, self.tau0, self._m, self._M)
        elif self._kernel == "powlaw-cutoff":
            compensator = uvhp_approx_powl_cut_compensator(self._timestamps, self.mu, self.eta, self.alpha, self.tau0, self._m, self._M)

        return compensator
    
    # TODO: Method for evaluating estimated intensity with option of plotting the intensity
    # TODO: Method for evaluating log-likelihood given timestamps and paramters??



class SumExpHawkesProcessEstimation():
    """
    Class for inference of unvivariate Hawkes processes with sum of P exponentials memory kernel. The conditional intensity
    function for the kernel without cutoff is defined as:
    TODO: Intensity formulas 
   
    The estiation is done by numerically maximizing the corresponding log-likelihood function.

    Attributes:
        success_flag (bool): True if process successfully estimated and the following attributes set.
        mu (float): The estimated background intensity :math:`\mu`.
        eta (float): The estimated branching ratio :math:`\\eta`.
        theta_vec (np.ndarray): Array of P estimated decay speeds 
        logL (Float): The log-likelihood value of the estimated process


    """

    _P = IntInExRange("P", lower_bound=0)
    _timestamps = PositiveOrderedFloatNdarray("timestamps")
    _grid_type = OneOf("random", "equidistant", "custom", "no-grid")
    _grid_size = IntInExRange("grid_size", lower_bound=0)


    def __init__(self, P: int, rng=rng) -> None:
        """Initilize the SumExpHawkesProcessEstimation class.

        Args:
            P (int): The number of exponential kernels that make up the memory kernel.
            rng (optional): numpy numba generator. For reproducible results use: np.random.default_rng(seed)
        """
        self._P = P
        self._kernel = "sum-expo"
        self._rng = rng
        self.success_flag = False
        self._ic = False

    def _check_valid_T(self, T: float, min_value) -> float:
        """Checks if input 'T' is valid

        Args:
            T (float): End time of the process
            min_value (float): The minimum value for the end time of the process. i.e. the largest/last timestamps in 'timestamps'

        Raises:
            ValueError: If T is invalid

        Returns:
            float: process end time 'T' 
        """
        if (T < min_value):
            raise ValueError(f"The process end time 'T' must be a number larger or equal to the last value in the sorted array 'timestamps'")
        else:
            return float(T)
    
    def aic(self) -> float:
        """Returns the Akaike information criteria of the estimated model

        Raises:
            Exception: If process has not been succesfully estimated yet. I.e. if 'success_flag' is False

        Returns:
            float: The AIC value
        """
        if self.success_flag == True:
            return -2 * self.logL + 2 * self._num_par
        else:
            raise Exception("ERROR: Cannot compute AIC, model has not been succesfully estimated!")

    def bic(self) -> float:
        """Returns the Baysian information criteria of the estimated model

        Raises:
            Exception: If process has not been succesfully estimated yet. I.e. if 'success_flag' is False

        Returns:
            float: The BIC value
        """
        if self.success_flag == True:
            return -2 * self.logL + self._num_par * np.log(self._num_par)
        else:
            raise Exception("ERROR: Cannot compute BIC, model has not been succesfully estimated!")

    def hq(self) -> float:
        """Returns the Hannan-Quinn  information criteria of the estimated model

        Raises:
            Exception: If process has not been succesfully estimated yet. I.e. if 'success_flag' is False

        Returns:
            float: The HQ value
        """
        if self.success_flag == True:
            return -2 * self.logL + 2 * self._num_par * np.log(np.log(self._num_par))
        else:
            raise Exception("ERROR: Cannot compute HQ, model has not been succesfully estimated!")

        
    def estimate(self, timestamps: np.ndarray, T: float, max_attempts: int=5, custom_param_vec0: bool=False, **kwargs): 
        """ Method for estimating the Hawkes process parameters using maximum likelihood estimation.

        Args:
            timestamps (np.ndarray): 1d numpy array containing the timestamp observations. 
                                     Must be sorted and only positive timestamps are allowed
            T (float): The end time of the Hawkes process.
            max_attempts (int, optional): If the optimization routine does not exit successfully. Number of times the optimization repeats with new starting values. Defaults to 5.
            custom_param_vec0 (bool, optional): If custom initial values should be used. Defaults to False. 
                                                If true you must supply an addtional variable 'custom_grid' containing a npumpy array of correct dimensions and bound.
        """
        self._timestamps = timestamps
        T = self._check_valid_T(T, timestamps[-1])

        succ_flag = None
        if custom_param_vec0 == False:
            attempt = 0
            while (succ_flag != 0) & (attempt <= max_attempts):
                attempt += 1
                # TODO: Improve the standard starting values: eta0 using the model-free branching ratio estimator, theta0 ????
                eta0 = rng.uniform(1e-3, 0.999)
                mu0 = len(timestamps) * (1 - eta0) / T
                theta0_vec = rng.uniform(1e-5, 5, self._P) 
                param_vec0 = np.append(np.append(mu0, eta0), theta0_vec)
            
                opt_result = uvhp_sum_expo_mle(self._timestamps, T, self._P, param_vec0)
                succ_flag = opt_result[2]["warnflag"] # 0 if convergence, 1 if too many func evals or iters, 2 other reason see ["task"]
        else:
            opt_result = uvhp_sum_expo_mle(self._timestamps, T, self._P, kwargs.get("param_vec0"))
            succ_flag = opt_result[2]["warnflag"] 
                
        self._opt_result = opt_result
        self._grid_type = "no-grid"    
        if succ_flag == 0:
            self.success_flag = True
            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.theta_vec = result_param_vec[0], result_param_vec[1], np.sort(result_param_vec[2::])

        else:
            print(f"WARNING: Optimization not successful: {opt_result[2]['task']}")

        
    def estimate_grid(self, timestamps: np.ndarray, T: float, grid_type: str, grid_size: int=20, rng=rng, **kwargs):
        """ Method for performing multiple optimizations using different initial values
            The estimated model is chosen to be the one with the largest log-likelihood value.

        Args:
            timestamps (np.ndarray): 1d numpy array containing the timestamp observations. 
                                     Must be sorted and only positive timestamps are allowed
            T (float): The end time of the Hawkes process.
            grid_type (str): One of: ('random') starting values for eta0 are chosen randomly.
                                     ('equidistant') starting values for eta0 are equidistant between 0 and 1
                                     ('custom') A custom eta0 grid of starting value is supplied to 'custom-grid' variable
            grid_size (int, optional): The number of optimizations from which the best model is chosen. Defaults to 20.
        """        
        self._timestamps = timestamps
        T = self._check_valid_T(T, timestamps[-1])
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
            opt_result = uvhp_sum_expo_mle(self._timestamps, T, self._P, param_vec0)
            
            if opt_result[2]["warnflag"] == 0: # if optimization exits successfully: append result
                opt_res_ls.append(opt_result)
                logL_res_ls.append(-opt_result[1])

        self._opt_result = opt_result   
        if len(opt_res_ls) > 0:
            # choose model with largest logL
            idx = logL_res_ls.index(max(logL_res_ls))
            opt_result = opt_res_ls[idx]   
        
            self.success_flag = True
            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.theta_vec = result_param_vec[0], result_param_vec[1], np.sort(result_param_vec[2::])

        else:
            print(f"WARNING: Grid Optimization not successful, none of the {grid_size} attempts exited succesfully")  


    def compensator(self):
        """ Computes the compensator corresponding to the estimated process. 
            The compensator of a point process is defined as:

        Raises:
            Exception: If model has not yet been successfully estimated yet

        Returns:
            np.ndarray: numpy array of timestamps (the compensator)
        """
        if self.success_flag == False:
            raise Exception("ERROR: Cannot compute compensator, model has not been succesfully estimated!")

        compensator = uvhp_sum_expo_compensator(self._timestamps, self.mu, self.eta, self.theta_vec)

        return compensator


    
class ExpHawkesProcessEstimation():
    """
    Class for inference of unvivariate Hawkes processes with normalized single exponentials memory kernel. The conditional intensity
    function for the kernel without cutoff is defined as:
    TODO: Intensity formulas 

    The estiation is done by numerically maximizing the corresponding log-likelihood function.


    Attributes:
        success_flag (bool): True if process successfully estimated and the following attributes set.
        mu (float): The estimated background intensity :math:`\mu`.
        eta (float): The estimated branching ratio :math:`\\eta`.
        theta_vec (np.ndarray): Array of P estimated decay speeds 
        logL (Float): The log-likelihood value of the estimated process

    """

    _timestamps = PositiveOrderedFloatNdarray("timestamps")
    _grid_type = OneOf("random", "equidistant", "custom", "no-grid")
    _grid_size = IntInExRange("grid_size", lower_bound=0)


    def __init__(self, rng=rng) -> None:
        """Initilize ExpHawkesProcessEstimation class

        Args:
            rng (optional): numpy numba generator. For reproducible results use: np.random.default_rng(seed)
        """
        self._rng = rng
        self.success_flag = False

    def _check_valid_T(self, T: float, min_value) -> float:
        """Checks if input 'T' is valid

        Args:
            T (float): End time of the process
            min_value (float): The minimum value for the end time of the process. i.e. the largest/last timestamps in 'timestamps'

        Raises:
            ValueError: If T is invalid

        Returns:
            float: process end time 'T' 
        """
        if (T < min_value):
            raise ValueError(f"The process end time 'T' must be a number larger or equal to the last value in the sorted array 'timestamps'")
        else:
            return float(T)
    
    def aic(self) -> float:
        """Returns the Akaike information criteria of the estimated model

        Raises:
            Exception: If process has not been succesfully estimated yet. I.e. if 'success_flag' is False

        Returns:
            float: The AIC value
        """
        if self.success_flag == True:
            return -2 * self.logL + 2 * self._num_par
        else:
            raise Exception("ERROR: Cannot compute AIC, model has not been succesfully estimated!")

    def bic(self) -> float:
        """Returns the Baysian information criteria of the estimated model

        Raises:
            Exception: If process has not been succesfully estimated yet. I.e. if 'success_flag' is False

        Returns:
            float: The BIC value
        """
        if self.success_flag == True:
            return -2 * self.logL + self._num_par * np.log(self._num_par)
        else:
            raise Exception("ERROR: Cannot compute BIC, model has not been succesfully estimated!")

    def hq(self) -> float:
        """Returns the Hannan-Quinn  information criteria of the estimated model

        Raises:
            Exception: If process has not been succesfully estimated yet. I.e. if 'success_flag' is False

        Returns:
            float: The HQ value
        """
        if self.success_flag == True:
            return -2 * self.logL + 2 * self._num_par * np.log(np.log(self._num_par))
        else:
            raise Exception("ERROR: Cannot compute HQ, model has not been succesfully estimated!")
        
    def estimate(self, timestamps: np.ndarray, T: float, max_attempts: int=5,  custom_param_vec0: bool=False, **kwargs) -> None: 
        """ Method for estimating the Hawkes process parameters using maximum likelihood estimation.

        Args:
            timestamps (np.ndarray): 1d numpy array containing the timestamp observations. 
                                     Must be sorted and only positive timestamps are allowed
            T (float): The end time of the Hawkes process.
            max_attempts (int, optional): If the optimization routine does not exit successfully. Number of times the optimization repeats with new starting values. Defaults to 5.
            custom_param_vec0 (bool, optional): If custom initial values should be used. Defaults to False. 
                                                If true you must supply an addtional variable 'custom_grid' containing a npumpy array of correct dimensions and bound.
        """
        self._timestamps = timestamps
        T = self._check_valid_T(T, timestamps[-1])

        succ_flag = None
        if custom_param_vec0 == False:
            attempt = 0
            while (succ_flag != 0) & (attempt <= max_attempts):
                attempt += 1
                # TODO: Improve the standard starting values: eta0 using the model-free branching ratio estimator, theta0 ????
                eta0 = rng.uniform(1e-3, 0.999)
                mu0 = len(timestamps) * (1 - eta0) / T
                theta0 = rng.uniform(1e-5, 5) 
                param_vec0 = np.array([mu0, eta0, theta0], dtype=np.float64)
            
                opt_result = uvhp_expo_mle(self._timestamps, T, param_vec0)
                succ_flag = opt_result[2]["warnflag"] # 0 if convergence, 1 if too many func evals or iters, 2 other reason see ["task"]
        else:
            opt_result = uvhp_expo_mle(self._timestamps, T, kwargs.get("param_vec0"))
            succ_flag = opt_result[2]["warnflag"] # 0 if convergence, 1 if too many func evals or iters, 2 other reason see ["task"]
                
        self._opt_result = opt_result
        self._grid_type = "no-grid"    
        if succ_flag == 0:
            self.success_flag = True
            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.theta = result_param_vec[0], result_param_vec[1], result_param_vec[2]

        else:
            print(f"WARNING: Optimization not successful: {opt_result[2]['task']}")
        
        
    def estimate_grid(self, timestamps: np.ndarray, T: float, grid_type: str, grid_size: int=20, rng=rng, **kwargs):
        """ Method for performing multiple optimizations using different initial values
            The estimated model is chosen to be the one with the largest log-likelihood value.

        Args:
            timestamps (np.ndarray): 1d numpy array containing the timestamp observations. 
                                     Must be sorted and only positive timestamps are allowed
            T (float): The end time of the Hawkes process.
            grid_type (str): One of: ('random') starting values for eta0 are chosen randomly.
                                     ('equidistant') starting values for eta0 are equidistant between 0 and 1
                                     ('custom') A custom eta0 grid of starting value is supplied to 'custom-grid' variable
            grid_size (int, optional): The number of optimizations from which the best model is chosen. Defaults to 20.
        """
        self._timestamps = timestamps
        T = self._check_valid_T(T, timestamps[-1])
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
            opt_result = uvhp_expo_mle(self._timestamps, T, param_vec0)
            
            if opt_result[2]["warnflag"] == 0: # if optimization exits successfully: append result
                opt_res_ls.append(opt_result)
                logL_res_ls.append(-opt_result[1])

        self._opt_result = opt_result   
        if len(opt_res_ls) > 0:
            # choose model with largest logL
            idx = logL_res_ls.index(max(logL_res_ls))
            opt_result = opt_res_ls[idx]   
        
            self.success_flag = True
            result_param_vec = opt_result[0]
            self.logL = -opt_result[1]
            self.mu, self.eta, self.theta = result_param_vec[0], result_param_vec[1], result_param_vec[2]

        else:
            print(f"WARNING: Grid Optimization not successful, none of the {grid_size} attempts exited succesfully")  
        

    def compensator(self):
        """ Computes the compensator corresponding to the estimated process. 
            The compensator of a point process is defined as:

        Raises:
            Exception: If model has not yet been successfully estimated yet

        Returns:
            np.ndarray: numpy array of timestamps (the compensator)
        """
        if self.success_flag == False:
            raise Exception("Cannot compute compensator, model has not been succesfully estimated!")

        compensator = uvhp_expo_compensator(self._timestamps, self.mu, self.eta, self.theta)

        return compensator


