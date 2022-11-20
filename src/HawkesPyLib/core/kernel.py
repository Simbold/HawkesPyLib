import numpy as np


def uvhp_expo_kernel(t: np.ndarray, eta: float, theta: float) -> np.ndarray:
    """ Computes values of the single exponential Hawkes process memory kernel.

    Args:
        t (np.ndarray): Single time value or 1d array containing all the times
                        at which the kernel value will be computed. Must be positive.
        eta (float): The branching ratio, 0 < eta > 1
        theta (float): 1d array of theta decay parameters, theta > 0

    Returns:
        np.ndarray: Single value or 1d array containing the kernel values at the given times.

    """
    kernel_value = (eta / theta) * np.exp(-t / theta)
    return kernel_value


def uvhp_sum_expo_kernel(t: np.ndarray, eta: float, theta_vec: float) -> np.ndarray:
    """ Computes values of the P-sum exponential Hawkes process memory kernel.

    Args:
        t (np.ndarray): Single time value or 1d array containing all the times
                        at which the kernel value will be computed. Must be positive.
        eta (float): The branching ratio, 0 < eta > 1
        theta_vec (float): 1d array of theta decay parameters, theta_k > 0

    Returns:
        np.ndarray: Single value or 1d array containing the kernel values at the given times.

    """
    P = theta_vec.shape[0]
    summ = 0
    for k in range(0, P):
        summ += 1 / theta_vec[k] * np.exp(-t / theta_vec[k])

    kernel_value = eta / np.float64(P) * summ
    return kernel_value


def uvhp_approx_powl_cutoff_kernel(t: np.ndarray, eta: float, alpha: float, tau: float, m: float, M: int) -> np.ndarray:
    """ Computes values of the Approximate power-law memory kernel with smooth cutoff component.

    Args:
        t (np.ndarray): Single time value or 1d array containing all the times
                        at which the kernel value will be computed. Must be positive.
        eta (float): Branching ratio of the Hawkes process, 0 > eta < 1
        alpha (float): Power-law coefficient, alpha > 0
        tau (float): Approximate location of cutoff, tau > 0
        m (float): Approximate power-law parameter, m > 0
        M (int): Number of weighted exponential kernels that approximate the power-law

    Returns:
        np.ndarray: Single value or 1d array containing the kernel values at the given times.
    """

    s = tau**(-1 - alpha) * (1 - np.power(m, -(1 + alpha) * M)) / (1 - np.power(m, -(1 + alpha)))
    z1 = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    z = z1 - s * tau * np.power(m, -1)

    ak = tau * m**np.arange(0, M, 1)
    an1 = tau / m
    summ = 0
    for k in range(0, M):
        summ += ak[k]**(-1 - alpha) * np.exp(-t / ak[k])

    kernel_value = eta * (summ - s * np.exp(-t / an1)) / z
    return kernel_value


def uvhp_approx_powl_kernel(t: np.ndarray, eta: float, alpha: float, tau: float, m: float, M: int) -> np.ndarray:
    """ Computes values of the Approximate power-law memory kernel.

    Args:
        t (np.ndarray): Single time value or 1d array containing all the times
                        at which the kernel value will be computed. Must be positive.
        eta (float): Branching ratio of the Hawkes process, 0 > eta < 1
        alpha (float): Power-law coefficient, alpha > 0
        tau (float): Approximate location of cutoff, tau > 0
        m (float): Approximate power-law parameter, m > 0
        M (int): Number of weighted exponential kernels that approximate the power-law

    Returns:
        np.ndarray: Single value or 1d array containing the kernel values at the given times.
    """
    z = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    ak = tau * m**np.arange(0, M, 1)
    summ = 0
    for k in range(0, M):
        summ += ak[k]**(-1 - alpha) * np.exp(-t / ak[k])

    kernel_value = eta * summ / z
    return kernel_value
