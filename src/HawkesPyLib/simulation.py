import numpy as np
from HawkesPyLib.core.simulation import (uvhp_approx_powl_cutoff_simulator,
                                         uvhp_expo_simulator,
                                         uvhp_approx_powl_simulator,
                                         uvhp_sum_expo_simulator,
                                         homogenous_poisson_simulator)
from HawkesPyLib.core.intensity import (uvhp_approx_powl_cutoff_intensity,
                                        uvhp_approx_powl_intensity,
                                        uvhp_expo_intensity,
                                        uvhp_sum_expo_intensity,
                                        generate_eval_grid)
from HawkesPyLib.util import (OneOf,
                              FloatInExRange,
                              IntInExRange,
                              PositiveFloatNdarray)
from HawkesPyLib.core.kernel import (uvhp_approx_powl_cutoff_kernel,
                                     uvhp_approx_powl_kernel,
                                     uvhp_expo_kernel,
                                     uvhp_sum_expo_kernel)

__all__ = ["ExpHawkesProcessSimulation", "SumExpHawkesProcessSimulation",
           "ApproxPowerlawHawkesProcessSimulation", "PoissonProcessSimulation"]


class PoissonProcessSimulation():
    r""" Class for simulation a homogenous Poisson process with rate parameter `mu`.
         The Poisson process is simulated over the half open intervall (0, T].
         In contrast to Hawkes processes, the intensity function \( \lambda(t) \),
         is constant and does not depend on \( t \): \( \lambda(t) = \mu, \; \forall t \)

         The simulation is performed by generating exponentially distributed inter-arrival times.
    """
    mu = FloatInExRange("mu", lower_bound=0)
    T = FloatInExRange("T", lower_bound=0)

    def __init__(self, mu: float) -> None:
        """

        Args:
            mu (float): Constant intensity rate parameter, mu > 0.
        """
        self.mu = mu
        self._simulated = False

    def simulate(self, T: float, seed: int = 0) -> np.ndarray:
        """ Generates a realization of the specified homogenous Poisson process.

        Args:
            T (float): Maximum time until which the Poisson process will be simulated.
            seed (int, optional): Seed the random number generator.

        Returns:
            np.ndarray: 1d array of simulated arrival times.
        """
        self.T = T
        self.timestamps = homogenous_poisson_simulator(self.T, self.mu, seed=seed)
        self.n_jumps = len(self.timestamps)
        self._simulated = True
        return self.timestamps


class ExpHawkesProcessSimulation():
    r""" Class for simulation of univariate Hawkes processes with single exponential memory kernel.
        The conditional intensity function is defined as:
        $$ \lambda(t) = \mu + \dfrac{\eta}{\theta} \sum_{t_i < t} e^{(-(t - t_i)/\theta)} $$

        where \(\mu\) (`mu`) is the constant background intensity, \(\eta\) (`eta`)
        is the branching ratio and \(\theta\) (`theta`) is the exponential decay parameter.

        The implemented simulation algorithm is based on Ogata's modified thinning algorithm.
    """
    mu = FloatInExRange("mu", lower_bound=0)
    eta = FloatInExRange("eta", lower_bound=0, upper_bound=1)
    theta = FloatInExRange("theta", lower_bound=0)
    T = FloatInExRange("T", lower_bound=0)

    def __init__(self, mu: float, eta: float, theta: float) -> None:
        r""" To initlize the class provide the following parameters that specify the single exponential Hawkes process.
        Args:
            mu (float): The background intensity, \(\mu > 0\).
            eta (float): The branching ratio, \(0 < \eta > 1\).
            theta (float): Exponential decay parameter, \(\theta > 0\).

        Attributes:
            mu (float): The background intensity, \(\mu > 0\).
            eta (float): The branching ratio, \(0 < \eta > 1\).
            theta (float): Exponential decay parameter, \(\theta > 0\).
            T (float): Maximum time until the Hawkes process was simulated.
            timestamps (np.ndarray): 1d array of simulated arrival times.
            n_jumps (int): Number of simulated arrival times.
        """
        self.mu = mu
        self.eta = eta
        self.theta = theta
        self._simulated = False

    # TODO: Add simulation options: max_n, multiple paths, branching sampler and conditional simulation
    def simulate(self, T: float, seed: int = None) -> np.ndarray:
        """ Generates a realization of the specified Hawkes process.

        Args:
            T (float): Maximum time until which the Hawkes process will be simulated.
            seed (int, optional): Seed the random number generator.

        Returns:
            np.ndarray: 1d array of simulated arrival times.
        """
        self.T = T
        self.timestamps = uvhp_expo_simulator(self.T, self.mu, self.eta, self.theta, seed=seed)
        self.n_jumps = len(self.timestamps)
        self._simulated = True
        return self.timestamps

    def intensity(self, step_size: float) -> np.ndarray:
        r""" Evaluates the intensity function \(\lambda(t)\) on a grid of equidistant timestamps over the
            closed interval [0, T] with step size `step_size`.

        !!! note
            - Additionally to the equidistant time grid the intensity is evaluated
            at each simulated arrival time.
            - If the process end time T is not perfectly divisible by
            the step size the 'effective' step size deviates slightly from the given one.

        Args:
            step_size (float): Step size of the time grid.

        Returns:
            np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)
        """
        if self._simulated is True:
            grid = generate_eval_grid(step_size, self.T)
            intensity = uvhp_expo_intensity(self.timestamps, grid, self.mu, self.eta, self.theta)
            return intensity

        else:
            raise Exception("ERROR: The Intensity of the simulated process can only be computed after the process is simulated")

    def kernel_values(self, times: np.ndarray) -> np.ndarray:
        r""" Returns the value of the memory kernel at given time values.
            The memory kernel of the single exponential memory kernel is defined as:
            $$ g(t) = \dfrac{\eta}{\theta}  e^{(-t/\theta)} $$

        Args:
            times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel.

        Returns:
            np.ndarray: 1d array containing the values of the memory kernel value at the given time(s).
        """
        kernel_values = uvhp_expo_kernel(times, self.eta, self.theta)
        return kernel_values


class SumExpHawkesProcessSimulation():
    r""" Class for simulation of univariate Hawkes processes with P-sum exponential memory kernel.
        The conditional intensity function is defined as:
        $$ \lambda(t) = \mu + \dfrac{\eta}{P} \sum_{t_i < t} \sum_{k=1}^{P} \dfrac{1}{\theta_k} e^{(-(t - t_i)/\theta_k)} $$

        where \(\mu\) (`mu`) is the constant background intensity, \(\eta \) (`eta`)
        is the branching ratio and \(\theta_k\) is the k'th exponential decay parameter in (`theta_vec`).

        The implemented simulation algorithm is based on Ogata's modified thinning algorithm.
    """
    mu = FloatInExRange("mu", lower_bound=0)
    eta = FloatInExRange("eta", lower_bound=0, upper_bound=1)
    theta_vec = PositiveFloatNdarray("theta_vec")
    T = FloatInExRange("T", lower_bound=0)

    def __init__(self, mu: float, eta: float, theta_vec: np.ndarray) -> None:
        r""" To initlize the class provide the following parameters that specify the P-sum exponential Hawkes process.
        Args:
            mu (float): The background intensity, \(\mu > 0\).
            eta (float): The branching ratio, \(0 < \eta > 1\).
            theta_vec (np.ndarray): 1d array of exponential decay parameters, \(\theta_k > 0\).

        Attributes:
            mu (float): The background intensity, \(\mu > 0\).
            eta (float): The branching ratio, \(0 < \eta > 1\).
            theta_vec (np.ndarray): 1d array of exponential decay parameters, \(\theta_k > 0\).
            T (float): Maximum time until the Hawkes process was simulated.
            timestamps (np.ndarray): 1d array of simulated arrival times.
            n_jumps (int): Number of simulated arrival times.
        """
        self.mu = mu
        self.eta = eta
        self.theta_vec = theta_vec
        self._simulated = False

    # TODO: Add simulation options: max_n, multiple paths, branching sampler and conditional simulation
    def simulate(self, T: float, seed: int = None) -> np.ndarray:
        """ Generates a realization of the specified Hawkes process.

        Args:
            T (float): Maximum time until which the Hawkes process will be simulated.
            seed (int, optional): Seed the random number generator.

        Returns:
            np.ndarray: 1d array of simulated arrival times.
        """
        self.T = T
        self.timestamps = uvhp_sum_expo_simulator(self.T, self.mu, self.eta, self.theta_vec, seed=seed)
        self.n_jumps = len(self.timestamps)
        self._simulated = True
        return self.timestamps

    def intensity(self, step_size: float) -> np.ndarray:
        r""" Evaluates the intensity function \(\lambda(t)\) on a grid of equidistant timestamps over the
            closed interval [0, T] with step size `step_size`.

        !!! note
            - Additionally to the equidistant time grid the intensity is evaluated
            at each simulated arrival time.
            - If the process end time T is not perfectly divisible by
            the step size the 'effective' step size deviates slightly from the given one.

        Args:
            step_size (float): Step size of the time grid.

        Returns:
            np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)
        """
        if self._simulated is True:
            grid = generate_eval_grid(step_size, self.T)
            intensity = uvhp_sum_expo_intensity(self.timestamps, grid, self.mu, self.eta, self.theta_vec)
            return intensity

        else:
            raise Exception("ERROR: The Intensity of the simulated process can only be computed after the process is simulated")

    def kernel_values(self, times: np.ndarray) -> np.ndarray:
        r""" Returns the value of the memory kernel at given time values. The memory kernel of the P-sum exponential
            memory kernel is defined as:
            $$ g(t) = \dfrac{\eta}{P} \sum_{k=1}^{P} \dfrac{1}{\theta_k} e^{(-t/\theta_k)} $$

        Args:
            times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel.

        Returns:
            np.ndarray: 1d array containing the values of the memory kernel value at the given time(s).
        """
        kernel_values = uvhp_sum_expo_kernel(times, self.eta, self.theta_vec)
        return kernel_values


class ApproxPowerlawHawkesProcessSimulation():
    r""" Class for simulation of univariate Hawkes processes with approximate power-law memory kernel.
        The conditional intensity function for the approximate power-law kernel is defined as:

        \[\lambda(t) = \mu + \sum_{t_i < t}  \dfrac{\eta}{Z} \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} \bigg],\]

        and the conditional intensity function for the approximate power-law kernel with smooth cutoff is defined as:

        \[\lambda(t) = \mu + \sum_{t_i < t} \dfrac{\eta}{Z} \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} -
        S e^{(-(t - t_i)/a_{-1})} \bigg],\]

        where \(\mu\) (`mu`) is the constant background intensity, \(\eta \) (`eta`)
        is the branching ratio, \(\alpha\) (`alpha`) is the tail exponent and \(\tau_0\) a scale parameter that
        also controls the decay timescale as well as the location of the smooth cutoff
        (i.e. the time at which the memory kernel reaches its maximum).

        The true power-law is approximtated by a sum of \(M\) exponential function with power-law weights
        \(a_k = \tau_0 m^k\). \(M\) is the number of exponentials used for the approximation and \(m\) is a scale
        parameter. The power-law approximation holds for times up to \(m^{M-1}\) after which the memory kernel
        decays exponentially. S and Z are scaling factors that ensure that the memory kernel integrates to \(\eta\)
        and that the kernel value at time zero is zero (for the smooth cutoff version). S and Z are computed automatically.

        The implemented simulation algorithm is based on Ogata's modified thinning algorithm.
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
        r""" Initilize class by setting the processes parameters.

        Args:
            kernel (str): Can be one of: 'powlaw' or 'powlaw-cutoff'.
            mu (float): The background intensity, \(\mu > 0\).
            eta (float): The branching ratio, \(0 < \eta > 1\).
            alpha (float): Tail exponent of the power-law decay, \(\alpha > 0\).
            tau0 (float): Timescale parameter, \(\tau_0 > 0\).
            m (float): Scale parameter of the power-law weights, \(m > 0 \).
            M (int): Number of weighted exponential functions that approximate the power-law.

        Attributes:
            mu (float): The background intensity, \(\mu > 0\).
            eta (float): The branching ratio, \(0 < \eta > 1\).
            alpha (float): Tail exponent of the power-law decay, \(\alpha > 0\).
            tau0 (float): Timescale parameter, \(\tau_0 > 0\).
            m (float): Scale parameter of the power-law weights, \(m > 0 \).
            M (int): Number of weighted exponential functions that approximate the power-law.
            T (float): Maximum time until the Hawkes process was simulated.
            timestamps (np.ndarray): 1d array of simulated arrival times.
            n_jumps (int): Number of simulated arrival times.
            """
        self.mu = mu
        self.eta = eta
        self.alpha = alpha
        self.tau0 = tau0
        self.m = m
        self.M = M
        self.kernel = kernel
        self._simulated = False

    # TODO: Add simulation options: max_n, multiple paths, branching sampler and conditional simulation
    def simulate(self, T: float, seed: int = None):
        """ Generates a realization of the specified Hawkes process.

        Args:
            T (float): Maximum time until which the Hawkes process will be simulated.
            seed (int, optional): Seed the random number generator.

        Returns:
            np.ndarray: 1d array of simulated arrival times.
        """
        self.T = T
        if self.kernel == "powlaw-cutoff":
            self.timestamps = uvhp_approx_powl_cutoff_simulator(self.T, self.mu, self.eta, self.alpha, self.tau0, self.m, self.M, seed=seed)

        else:
            self.timestamps = uvhp_approx_powl_simulator(self.T, self.mu, self.eta, self.alpha, self.tau0, self.m, self.M, seed=seed)

        self.n_jumps = len(self.timestamps)
        self._simulated = True

        return self.timestamps

    def intensity(self, step_size: float) -> np.ndarray:
        r""" Evaluates the intensity function \(\lambda(t)\) on a grid of equidistant timestamps over the
            closed interval [0, T] with step size `step_size`.

        !!! note
            - Additionally to the equidistant time grid the intensity is evaluated
            at each simulated arrival time.
            - If the process end time T is not perfectly divisible by
            the step size the 'effective' step size deviates slightly from the given one.

        Args:
            step_size (float): Step size of the time grid.

        Returns:
            np.ndarray: 2d array of timestamps (column 0) and corresponding intensity values (column 1)
        """
        if self._simulated is True:
            grid = generate_eval_grid(step_size, self.T)

            if self.kernel == "powlaw-cutoff":
                intensity = uvhp_approx_powl_cutoff_intensity(self.timestamps, grid, self.mu, self.eta,
                                                              self.alpha, self.tau0, self.m, self.M)
            elif self.kernel == "powlaw":
                intensity = uvhp_approx_powl_intensity(self.timestamps, grid, self.mu, self.eta,
                                                       self.alpha, self.tau0, self.m, self.M)

            return intensity

        else:
            raise Exception("ERROR: The Intensity of the simulated process can only be computed after the process is simulated")

    def kernel_values(self, times: np.ndarray) -> np.ndarray:
        r""" Returns the value of the memory kernel at given time values. The memory kernel of the approximate
        power-law memory kernel is defined as:
        $$ g(t) = \dfrac{\eta}{Z} \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} \bigg] $$

        and the memory kernel of the approximate power-law kernel with smooth cutoff is defined as:
        $$ g(t) = \dfrac{\eta}{Z} \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \bigg] $$

        Args:
            times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel.

        Returns:
            np.ndarray: 1d array containing the values of the memory kernel value at the given time(s).
        """
        if self.kernel == "powlaw-cutoff":
            return uvhp_approx_powl_cutoff_kernel(times, self.eta, self.alpha, self.tau0, self.m, self.M)

        elif self.kernel == "powlaw":
            return uvhp_approx_powl_kernel(times, self.eta, self.alpha, self.tau0, self.m, self.M)
