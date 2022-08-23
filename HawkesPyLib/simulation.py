import numpy as np
from HawkesPyLib.num.simu import (uvhp_approx_powl_cutoff_simulator,
                      uvhp_expo_simulator,
                      uvhp_approx_powl_simulator,
                      uvhp_sum_expo_simulator)
from HawkesPyLib.util import (OneOf,
                FloatInExRange,
                IntInExRange,
                PositiveFloatNdarray)


class ExpHawkesProcessSimulation():
    """ Class for simulation of Hawkes processes with single exponential kernel.
        The conditional intensity function is defined as:
    .. math::
        \lambda^*(t) = \mu + \sum_{t_i < t} \dfrac{\\eta}{\\theta} \exp(-(t - t_i)/\\theta),
    where :math:`\mu` ("mu") is a constant background intensity, :math:`\\eta` ("eta")
    is the branching ratio and :math:"\\theta" is the exponential decay parameter. 
    The simultion is done using Ogata's thinning algorithm.

     Attributes:
        mu (float): The background intensity :math:`\mu`.
        eta (float): The branching ratio :math:`\\eta`.
        theta (float): Decay speed of exponential kernel :math: '\\theta'. 
        T (int): Time until the Hawkes process was simulated
        timestamps (np.ndarray): numpy array of simulated timestamps.
        n_jumps (int): The number of simulated timestamps.
    """
                                                                                
    mu = FloatInExRange("mu", lower_bound=0)
    eta = FloatInExRange("eta", lower_bound=0, upper_bound=1)
    theta = FloatInExRange("theta", lower_bound=0)
    T = FloatInExRange("T", lower_bound=0)

    def __init__(self, mu: float, eta: float, theta: float) -> None:
        self.mu = mu
        self.eta = eta
        self.theta = theta
    
    # TODO: Add simulation options: max_n, multiple paths, branching sampling istead of Ogata's thinning algorithm
    def simulate(self, T: float, seed: int=None) -> np.ndarray:
        """ Method for generating a realisation of the specified Hawkes process.

        Args:
            T (float): Maximum time until which the Hawkes process will be simulated.
            seed (int, optional): To seed the numba random number generator. Defaults to None.

        Returns:
            np.ndarray: numpy array of simulated timestamps
        """
        self.T = T
        self.timestamps = uvhp_expo_simulator(self.T, self.mu, self.eta, self.theta, seed=seed)
        self.n_jumps = len(self.timestamps)
        return self.timestamps
    
    # TODO: Add methods: evaluate intensity on grid, simulate conditional on a set of timestamps


class SumExpHawkesProcessSimulation():
    """Class for simulation of Hawkes processes with P sum of exponential kernel.
    The conditional intensity function is defined as:
    .. math::
        \lambda^*(t) = \mu + \sum_{t_i < t} \sum_{k=1}^{P} \dfrac{\\eta}{\\theta_k} \exp(-(t - t_i)/\\theta_k),
    where :math:`\mu` ("mu") is a constant background intensity, :math:`\\eta` ("eta")
    is the branching ratio and :math:"\\theta_k" is the k'th exponential decay parameter. The simultion is done using 
    Ogata's thinning algorithm.


    Attributes:
        mu (float): The background intensity :math:`\mu`.
        eta (float): The branching ratio :math:`\\eta`.
        theta _vec (np.ndarray): Decay speed of the P exponential kernels :math: '\\theta_k'. 
        T (int): Time until the Hawkes process was simulated
        timestamps (np.ndarray): numpy array of simulated timestamps.
        n_jumps (int): The number of simulated timestamps.
    """
    mu = FloatInExRange("mu", lower_bound=0)
    eta = FloatInExRange("eta", lower_bound=0, upper_bound=1)
    theta_vec = PositiveFloatNdarray("theta_vec")
    T = FloatInExRange("T", lower_bound=0)

    def __init__(self, mu: float, eta: float, theta_vec: np.ndarray) -> None:
        self.mu = mu
        self.eta = eta
        self.theta_vec = theta_vec
    
    # TODO: Add simulation options: max_n, multiple paths, branching sampling istead of Ogata's thinning algorithm
    def simulate(self, T: float, seed: int=None) -> np.ndarray:
        """ Method for generating a realisation of the specified Hawkes process.

        Args:
            T (float): Maximum time until which the Hawkes process will be simulated.
            seed (int, optional): To seed the numba random number generator. Defaults to None.

        Returns:
            np.ndarray: numpy array of simulated timestamps
        """
        self.T = T
        self.timestamps = uvhp_sum_expo_simulator(self.T, self.mu, self.eta, self.theta_vec, seed=seed)
        self.n_jumps = len(self.timestamps)
        return self.timestamps
    
    # TODO: Add methods: evaluate intensity on grid, simulate conditional on a set of timestamps


class ApproxPowerlawHawkesProcessSimulation():
    """
    Class for simulation of Hawkes processes with approximate power-law kernel.
    The conditional intensity
    function for the kernel without cutoff is defined as:
    .. math::
        \lambda^*(t) = \mu + \sum_{t_i < t}  \dfrac{\\eta}{Z} \\bigg[ \sum_{k=0}^{M-1} \\alpha_k^{-(1+\\alpha)} \exp(-(t - t_i)/\\alpha_k) \\bigg],
    where :math:`\mu` ("mu") is a constant background intensity, :math:`\\eta` ("eta")
    is the branching ratio and :math:\\alpha_k = \\tau_0 m^k the k'th powerlaw weight. S and Z are scaling factors and computed automatically. 
    M and m define the accuracy of the power-law approximation.
    The intensity for the kernel with cutoff is given by:
        .. math::
        \lambda^*(t) = \mu + \sum_{t_i < t}  \dfrac{\\eta}{Z} \\bigg[ \sum_{k=0}^{M-1} \\alpha_k^{-(1+\\alpha)} \exp(-(t - t_i)/\\alpha_k) - S \exp(-(t - t_i)/\\talpha_{-1}) \\bigg],
   
    The simultion is done using Ogata's thinning algorithm.

    Attributes:
        mu (float): The background intensity :math:`\mu`.
        eta (float): The branching ratio :math:`\\eta`.
        alpha (float): Power-law coefficient :math: '\\alpha'.
        tau0 (float): Specifies shape of the memory kernel :math: '\\tau_0'.  
        m (float): Specify the approximation of the true power-law.
        M (int): Number of weighted exponential function that approximate the true power-law.
        T (int): Time until the Hawkes process was simulated
        timestamps (np.ndarray): numpy array of simulated timestamps.
        n_jumps (int): The number of simulated timestamps.
    """
    mu = FloatInExRange("mu", lower_bound=0)
    eta = FloatInExRange("eta", lower_bound=0, upper_bound=1)
    alpha = FloatInExRange("alpha", lower_bound=0)
    tau0 = FloatInExRange("tau0", lower_bound=0)
    m = FloatInExRange("m", lower_bound=0)
    M = IntInExRange("M", lower_bound=0)
    T = FloatInExRange("T", lower_bound=0)
    kernel = OneOf("powlaw", "powlaw-cutoff")

    def __init__(self, kernel: str, mu: float, eta: float, alpha: float, tau0: float, m: float, M: int) -> None:
        """ Initilizes the ApproxPowerlawHawkesProcessSimulation class

        Args:
            kernel (str): Can be one of: 'powlaw' or 'powlaw-cutoff'. Specifies the shape of the memory kernel.
            mu (float): The background intensity :math:`\mu`.
            eta (float): The branching ratio :math:`\\eta`.
            alpha (float): Power-law coefficient :math: '\\alpha'.
            tau0 (float): Specifies shape of the memory kernel :math: '\\tau_0'.  
            m (float): Specify the approximation of the true power-law.
            M (int): Number of weighted exponential function that approximate the true power-law.
            """
        self.mu = mu
        self.eta = eta
        self.alpha = alpha
        self.tau0 = tau0
        self.m = m
        self.M = M
        self.kernel = kernel
    
    # TODO: Add simulation options: max_n, multiple paths, branching sampling instead of Ogata's thinning algorithm
    def simulate(self, T: float, seed: int=None):
        """ Method for generating a realisation of the specified Hawkes process.

        Args:
            T (float): Maximum time until which the Hawkes process will be simulated.
            seed (int, optional): To seed the numba random number generator. Defaults to None.

        Returns:
            np.ndarray: numpy array of simulated timestamps
        """
        self.T = T
        if self.kernel == "powlaw-cutoff":
            self.timestamps = uvhp_approx_powl_cutoff_simulator(self.T, self.mu, self.eta, self.alpha, self.tau0, self.m, self.M, seed=seed)

        else:
            self.timestamps = uvhp_approx_powl_simulator(self.T, self.mu, self.eta, self.alpha, self.tau0, self.m, self.M, seed=seed)         
        self.n_jumps = len(self.timestamps)
        return self.timestamps
    
    # TODO: Add methods: evaluate intensity on grid, simulate conditional on a set of timestamps
