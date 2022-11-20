
import numpy as np
from HawkesPyLib.core.intensity import (uvhp_approx_powl_cutoff_intensity,
                                        uvhp_approx_powl_intensity,
                                        uvhp_expo_intensity,
                                        uvhp_sum_expo_intensity,
                                        generate_eval_grid)

from HawkesPyLib.core.kernel import (uvhp_approx_powl_cutoff_kernel,
                                     uvhp_approx_powl_kernel,
                                     uvhp_expo_kernel,
                                     uvhp_sum_expo_kernel)

from HawkesPyLib.core.compensator import (uvhp_approx_powl_compensator,
                                          uvhp_approx_powl_cut_compensator,
                                          uvhp_expo_compensator,
                                          uvhp_sum_expo_compensator)

from HawkesPyLib.core.logll import (uvhp_approx_powl_logL,
                                    uvhp_approx_powl_cut_logL,
                                    uvhp_expo_logL,
                                    uvhp_sum_expo_logL)

from HawkesPyLib.util import (OneOf,
                              FloatInExRange,
                              IntInExRange,
                              PositiveOrderedFloatNdarray,
                              PositiveFloatNdarray)

__all__ = ["UnivariateHawkesProcess"]


class UnivariateHawkesProcess:
    r""" Implements multiple univariate Hawkes process models and
        allows for the calculation of basic process characteristics.

        The following Hawkes process models are implemented:

        - Hawkes process with single exponential memory kernel ('expo' `kernel`), defined by the conditional intensity function:
            \[\lambda(t) = \mu + \dfrac{\eta}{\theta} \sum_{t_i < t} e^{(-(t - t_i)/\theta)}\]

            where \(\mu\) (`mu`) is the constant background intensity, \(\eta \) (`eta`)
            is the branching ratio and \(\theta\) (`theta`) is the exponential decay parameter.

        - Hawkes process with P-sum exponential memory kernel ('sum-expo' `kernel`), defined by the conditional intensity function:
            \[\lambda(t) = \mu + \dfrac{\eta}{P} \sum_{t_i < t} \sum_{k=1}^{P} \dfrac{1}{\theta_k} e^{(-(t - t_i)/\theta_k)}\]

            where \(\mu\) (`mu`) is the constant background intensity, \(\eta \) (`eta`)
            is the branching ratio and \(\theta_k\) is the k'th exponential decay parameter in (`theta_vec`).

        - Hawkes process with approximate power-law kernel ('powlaw' `kernel`), defined by the conditional intenstiy function:
            \[\lambda(t) = \mu + \sum_{t_i < t}  \dfrac{\eta}{Z} \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} \bigg],\]

            where \(\mu\) (`mu`) is the constant background intensity, \(\eta \) (`eta`)
            is the branching ratio, \(\alpha\) (`alpha`) is the power-law tail exponent and \(\tau_0\)
            a scale parameter controlling the decay timescale.

        - Hawkes process with approximate power-law kernel with smooth cutoff ('powlaw-cutoff' `kernel`),
          defined by the conditional intenstiy function:
            \[\lambda(t) = \mu + \sum_{t_i < t} \dfrac{\eta}{Z}
            \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \bigg],\]

            where \(\mu\) (`mu`) is the constant background intensity, \(\eta \) (`eta`)
            is the branching ratio, \(\alpha\) (`alpha`) is the power-law tail exponent and \(\tau_0\)
            a scale parameter controlling the decay timescale and
            the location of the smooth cutoff point (i.e. the time duration after each jump at which the intensity reaches a maximum).

        The true power-law is approximtated by a sum of \(M\) exponential function with power-law weights
        \(a_k = \tau_0 m^k\). \(M\) is the number of exponentials used for the approximation and \(m\) is a scale
        parameter. The power-law approximation holds for times up to \(m^{M-1}\) after which the memory kernel
        decays exponentially. S and Z are scaling factors that ensure that the memory kernel integrates to \(\eta\)
        and that the kernel value at time zero is zero (for the smooth cutoff version). S and Z are computed automatically.

        The following main methods are available:

        - `UnivariateHawkesProcess.intensity` evaluates the conditional intensity function \(\lambda(t)\).

        - `UnivariateHawkesProcess.compensator` computes the compensator process \(\Lambda(t) = \int_0^t \lambda(u) du\).

        - `UnivariateHawkesProcess.kernel_values` returns values for the memory kernel.

        - `UnivariateHawkesProcess.compute_logL` computes the log-likelihood function.

        The class works by first initlizing with the desired kernel type and secondly by setting the corresponding parameters
        and arrival times using the `set_params()` and `set_arrival_times()` methods.

     """

    # instance variables for the Approximate power law models
    _m = FloatInExRange("m", lower_bound=0)
    _M = IntInExRange("M", lower_bound=0)
    alpha = FloatInExRange("alpha", lower_bound=0)
    tau0 = FloatInExRange("tau0", lower_bound=0)
    # for sum of exponential and single exponential
    theta_vec = PositiveFloatNdarray("theta_vec")
    theta = FloatInExRange("alpha", lower_bound=0)

    _kernel = OneOf("expo", "sum-expo", "powlaw", "powlaw-cutoff")
    timestamps = PositiveOrderedFloatNdarray("timestamps")
    mu = FloatInExRange("mu", lower_bound=0)
    eta = FloatInExRange("eta", lower_bound=0, upper_bound=1)

    def __init__(self, kernel: str) -> None:
        r"""
        Args:
            kernel (str): Type of Hawkes process memory kernel, one of 'expo', 'sum-expo', 'powlaw', 'powlaw-cutoff'

        Attributes:
            mu (float): The background intensity, \(\mu > 0\).
            eta (float): The branching ratio, \(0 < eta > 1\).
            theta (float): Decay speed of single exponential kernel, \(\theta > 0\). Only set if `kernel` is set to 'expo'.
            theta_vec (np.ndarray, optional): Decay speeds of the P-sum, exponential kernel, \(\theta_k > 0\).
                                                Only set if `kernel` is set to 'sum-expo'.
            alpha (float): Tail exponent of the power-law decay, \(\alpha > 0\). Only set if `kernel` is set to 'powlaw' or 'powlaw-cutoff'.
            tau0 (float): Timescale parameter, \(\tau_0 > 0\). Only set if `kernel` is set to 'powlaw' or 'powlaw-cutoff'.
            T (float): End time of the Hawkes process.
            timestamps (np.ndarray): 1d array of arrival times.
        """
        self._kernel = kernel
        self._params_set = False
        self._timestamps_set = False

    def set_params(self, mu: float, eta: float, **kwargs) -> None:
        r"""Set Hawkes process model parameters.
            Arguments `mu` and `eta` are always requiered.
            The other arguements depend on the chosen `kernel` and are supplied to **kwargs.

        Args:
            mu (float): The background intensity, \(\mu > 0\).
            eta (float): The branching ratio, \(0 < \eta > 1\).
            theta (float): Decay speed of single exponential kernel, \(\theta\) > 0. Must be set if `kernel` is set to 'expo'.
            theta_vec (np.ndarray, optional): Decay speed of P-sum, exponential kernel, theta_k > 0.
                                              Must be set if `kernel` is set to 'sum-expo'.
            alpha (float): Tail index of the power-law decay, \(\alpha > 0\). Must be set if `kernel` is set to 'powlaw' or 'powlaw-cutoff'.
            tau0 (float): Timescale parameter, \(\tau_0 > 0\). Must be set if `kernel` is set to 'powlaw' or 'powlaw-cutoff'.
            m (float, optional): Scale parameter of the power-law weights, m > 0.
                                 Must be set if `kernel` is set to 'powlaw' or 'powlaw-cutoff'.
            M (int, optional): Number of weighted exponential kernels used in the approximate power-law kernel.
             Must be set if `kernel` is set to 'powlaw' or 'powlaw-cutoff'.
        """
        self.mu = mu
        self.eta = eta

        if self._kernel == "expo":

            try:
                self.theta = kwargs.pop("theta")
                self._num_par = 3

            except KeyError:
                raise TypeError("For kernel 'expo' input for parameter 'theta' is requiered")

        elif self._kernel == "sum-expo":

            try:
                self.theta_vec = kwargs.pop("theta_vec")
                self._num_par = 2 + len(self.theta_vec)

            except KeyError:
                raise TypeError("For kernel 'sum-expo' input for parameter 'theta_vec' is requiered")

        elif self._kernel in ["powlaw", "powlaw-cutoff"]:

            try:
                self.alpha = kwargs.pop("alpha")
                self.tau0 = kwargs.pop("tau0")
                self.m = kwargs.pop("m")
                self.M = kwargs.pop("M")
                self._num_par = 4

            except KeyError:
                raise TypeError("For kernels 'powlaw' and 'powlaw-cutoff' input for parameters 'alpha', 'tau0', 'm' and 'M' is requiered")

        self._params_set = True

    def set_arrival_times(self, timestamps: np.ndarray, T: float) -> None:
        """Set arrival times of the process.

        Args:
            timestamps (np.ndarray): 1d array of arrival times of the process. Arrival times must be positive and ordered.
            T (float): End time of the process. T must be larger or equal to the last arrival time.
        """
        self.timestamps = timestamps
        self.T = self._check_valid_T(T, timestamps[-1])
        self._timestamps_set = True

    def get_params(self) -> tuple:
        """Returns the model parameters:

        Returns:
            tuple: Tuple of parameter values such as:

        If `kernel` set to 'expo':

        - `mu` (float), `eta` (float), `theta` (float)

        If `kernel` set to 'sum-expo':

        - `mu` (float), `eta` (float), `theta_vec` (np.ndarray)

        If `kernel` set to 'powlaw' or 'powlaw-cutoff':

        - `mu` (float), `eta` (float), `alpha` (float), `tau0` (float), `m` (float), `M` (int)
        """

        if self._kernel == "expo":
            return self.mu, self.eta, self.theta

        elif self._kernel == "sum-expo":
            return self.mu, self.eta, self.theta_vec

        else:  # _kernel can only be set to one off the four
            return self.mu, self.eta, self.alpha, self.tau0, self.m, self.M

    def _check_valid_T(self, T: float, min_value) -> float:
        """Checks if input 'T' is valid

        Args:
            T (float): End time of the process
            min_value (float): The minimum value for the end time of the process. i.e. the largest/last timestamps in 'timestamps'

        Raises:
            ValueError: If T is invalid

        Returns:
            float: process end time T as a float
        """
        if (T < min_value):
            raise ValueError("The process end time 'T' must be larger or equal to the last value in the sorted array 'timestamps'")
        else:
            return float(T)

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
        if (self._params_set is True) & (self._timestamps_set is True):
            grid = generate_eval_grid(step_size, self.T)

            if self._kernel == "powlaw-cutoff":
                intensity = uvhp_approx_powl_cutoff_intensity(self.timestamps, grid, self.mu, self.eta,
                                                              self.alpha, self.tau0, self._m, self._M)

            elif self._kernel == "expo":
                intensity = uvhp_expo_intensity(self.timestamps, grid, self.mu, self.eta, self.theta)

            elif self._kernel == "powlaw":
                intensity = uvhp_approx_powl_intensity(self.timestamps, grid, self.mu, self.eta,
                                                       self.alpha, self.tau0, self._m, self._M)

            elif self._kernel == "sum-expo":
                intensity = uvhp_sum_expo_intensity(self.timestamps, grid, self.mu, self.eta, self.theta_vec)

            return intensity

        else:
            raise Exception("ERROR: Cannot compute intensity, parameters or arrival times are not set!")

    def compensator(self):
        r""" Computes the compensator of the process at the set timestamps given the set parameter values.
            The compensator of a point process is defined as the integral of the conditional intensity:
            $$ \Lambda(t) = \int_0^t \lambda(u) du $$

        !!! tip
            If the set arrival times are a realization of the Hawkes process specified by the set parameters,
            the resulting compensator will be a Poisson process with unit intensity (assuming a large enough sample size).

        Returns:
            np.ndarray: 1d array of timestamps (the compensator)
        """

        if (self._params_set is True) & (self._timestamps_set is True):

            if self._kernel == "powlaw-cutoff":
                compensator = uvhp_approx_powl_cut_compensator(self.timestamps, self.mu, self.eta, self.alpha, self.tau0, self._m, self._M)

            elif self._kernel == "expo":
                compensator = uvhp_expo_compensator(self.timestamps, self.mu, self.eta, self.theta)

            elif self._kernel == "powlaw":
                compensator = uvhp_approx_powl_compensator(self.timestamps, self.mu, self.eta, self.alpha, self.tau0, self._m, self._M)

            elif self._kernel == "sum-expo":
                compensator = uvhp_sum_expo_compensator(self.timestamps, self.mu, self.eta, self.theta_vec)

            return compensator

        else:
            raise Exception("ERROR: Cannot compute intensity, parameters or arrival times are not set!")

    def kernel_values(self, times: np.ndarray) -> np.ndarray:
        r""" Returns the value of the memory kernel at given time values.

        The single exponential memory kernel:

        $$ g(t) = \dfrac{\eta}{\theta}  e^{(-t/\theta)} $$

        The P-sum exponential memory kernel:

        $$ g(t) = \dfrac{\eta}{P} \sum_{k=1}^{P} \dfrac{1}{\theta_k} e^{(-t/\theta_k)} $$

        The approximate power-law memory kernel:

        $$ g(t) = \dfrac{\eta}{Z} \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} \bigg] $$

        And the approximate power-law kernel with smooth cutoff :

        $$ g(t) = \dfrac{\eta}{Z} \bigg[ \sum_{k=0}^{M-1} a_k^{-(1+\alpha)} e^{(-(t - t_i)/a_k)} - S e^{(-(t - t_i)/a_{-1})} \bigg] $$

        Args:
            times (np.ndarray): 1d array of time values for which to compute the value of the memory kernel.

        Returns:
            np.ndarray: 1d array containing the values of the memory kernel value at the given time(s).
        """

        if self._params_set is True:

            if self._kernel == "powlaw-cutoff":
                kernel_values = uvhp_approx_powl_cutoff_kernel(times, self.eta, self.alpha, self.tau0, self._m, self._M)

            elif self._kernel == "expo":
                kernel_values = uvhp_expo_kernel(times, self.eta, self.theta)

            elif self._kernel == "powlaw":
                kernel_values = uvhp_approx_powl_kernel(times, self.eta, self.alpha, self.tau0, self._m, self._M)

            elif self._kernel == "sum-expo":
                kernel_values = uvhp_sum_expo_kernel(times, self.eta, self.theta_vec)

            return kernel_values

        else:
            raise Exception("ERROR: Cannot compute kernel values, model has, parameters are not set!")

    def compute_logL(self):
        r""" Computes the log-likelihood function given the process parameters, arrival times and process end time T.

         The log-likelihood function of a Hawkes process is given by:
            $$ logL(t_1,..., t_n) = -\int_0^T \lambda(t) du + \sum_{t=1}^n log(\lambda(t_i))$$

        Returns:
            float: The log-likelihood value
        """
        if (self._params_set is True) & (self._timestamps_set is True):

            if self._kernel == "powlaw-cutoff":
                param_vec = np.array([self.mu, self.eta, self.alpha, self.tau0])
                nlogL = uvhp_approx_powl_cut_logL(param_vec, self.timestamps, self.T, self._m, self._M)

            elif self._kernel == "expo":
                param_vec = np.array([self.mu, self.eta, self.theta])
                nlogL = uvhp_expo_logL(param_vec, self.timestamps, self.T)

            elif self._kernel == "powlaw":
                param_vec = np.array([self.mu, self.eta, self.alpha, self.tau0])
                nlogL = uvhp_approx_powl_logL(param_vec, self.timestamps, self.T, self._m, self._M)

            elif self._kernel == "sum-expo":
                param_vec = np.append(np.append(self.mu, self.eta), self.theta_vec)
                nlogL = uvhp_sum_expo_logL(param_vec, self.timestamps, self.T)

            self.logL = -nlogL

            return -nlogL

        else:
            raise Exception("ERROR: Cannot compute log-likelihood value, parameters or arrival times are not set!")

    def ic(self, type: str) -> float:
        """Calculates one of the following information criteria given the set parameters and arrival times:

        - 'aic': Akaike information criteria

        - 'bic': Baysian information criteria

        - 'hq': Hannan-Quinn  information criteria

        Args:
            type (str): Which information criterion to return, must be one of: 'aic', 'bic', 'hq'

        Returns:
            float: The information criterions value
        """
        if (self._params_set is True) & (self._timestamps_set is True):

            if not hasattr(self, 'logL'):
                self.compute_logL()

            if type == "aic":
                return -2 * self.logL + 2 * self._num_par
            elif type == "bic":
                return -2 * self.logL + self._num_par * np.log(self._num_par)
            elif type == "hq":
                return -2 * self.logL + 2 * self._num_par * np.log(np.log(self._num_par))
            else:
                raise Exception('ERROR: type must be one of: "aic", "bic", "hq"')

        else:
            raise Exception("ERROR: Cannot compute IC, parameters or arrival times are not set!")
