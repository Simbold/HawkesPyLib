import numpy as np
from numba import jit, float64
from numba.types import int32


@jit(float64[:](float64[:], float64, float64, float64), nopython=True, cache=False, nogil=True)
def uvhp_expo_compensator(sample_vec: np.ndarray, mu: float, eta: float, theta: float) -> np.ndarray:
    """Computes the compensator for a Hawkes procss with single exponential kernel.

    Args:
        sample_vec (np.ndarray): numpy array of timestamps. Must be non-negative and sorted in ascending order.
        mu (float): Background intensity of the Hawkes process. mu > 0
        eta (float): Branching ratio of the Hawkes process. 0 < eta < 1
        theta (float): Decay speed of the exponential memory kernel. theta > 0

    Returns:
        np.ndarray: numpy array of timestamps
    """
    n = len(sample_vec)
    it = np.zeros(n)

    it[0] = mu * sample_vec[0]
    td1 = sample_vec[1] - sample_vec[0]
    rk = 1
    it[1] = it[0] + mu * td1 + eta * (1 - np.exp(-td1 / theta)) * rk

    for i in range(2, n):
        tdi = sample_vec[i] - sample_vec[i-1]
        rk = np.exp(-(sample_vec[i-1] - sample_vec[i-2]) / theta) * rk + 1
        it[i] = it[i-1] + mu * tdi + eta * rk * (1 - np.exp(-tdi / theta))

    return it


@jit(float64[:](float64[:], float64, float64, float64, float64, float64, int32),
     nopython=True, cache=False, nogil=True)
def uvhp_approx_powl_compensator(sample_vec: np.ndarray, mu: float, eta: float, alpha: float,
                                 tau: float, m: float, M: int) -> np.ndarray:
    """Computes the compensator for a Hawkes procss with approximate power-law kernel.

    Args:
        sample_vec (np.ndarray): numpy array of timestamps. Must be non-negative and sorted in ascending order.
        mu (float): Background intensity of the Hawkes process. mu > 0
        eta (float): Branching ratio of the Hawkes process. 0 < eta < 1
        alpha (float): Power-law coefficient. alpha > 0
        tau (float): Decay speed of the kernel. tau > 0
        m (float): Memory kernel parameter, m > 0
        M (int): Memory kernel parameter, M > 0

    Returns:
        np.ndarray: numpy array of timestamps
    """
    n = len(sample_vec)
    it = np.zeros(n)
    z = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    etaoz = eta / z

    ak = tau * m**np.arange(0, M, 1)
    akp1 = ak**(-alpha)

    it[0] = mu * sample_vec[0]
    td1 = sample_vec[1] - sample_vec[0]
    rk = np.ones(M)
    summ = 0.
    for k in range(0, M):
        summ += akp1[k] * (1 - np.exp(-td1 / ak[k]))
    it[1] = it[0] + mu * td1 + etaoz * summ

    for i in range(2, n):
        tdi = sample_vec[i] - sample_vec[i-1]
        summ = 0.
        for k in range(0, M):
            rk[k] = np.exp(-(sample_vec[i-1] - sample_vec[i-2]) / ak[k]) * rk[k] + 1
            summ += akp1[k] * rk[k] * (1 - np.exp(-tdi / ak[k]))

        it[i] = it[i-1] + mu * tdi + etaoz * summ

    return it


@jit(float64[:](float64[:], float64, float64, float64, float64, float64, int32), nopython=True, cache=False, nogil=True)
def uvhp_approx_powl_cut_compensator(sample_vec, mu, eta, alpha, tau, m, M) -> np.ndarray:
    """Computes the compensator for a Hawkes procss with approximate power-law kernel with cutoff.

    Args:
        sample_vec (np.ndarray): numpy array of timestamps. Must be non-negative and sorted in ascending order.
        mu (float): background intensity of the Hawkes process
        eta (float): Branching ratio of the Hawkes process. 0 < eta < 1
        alpha (float): Power-law coefficient. alpha > 0
        tau (float): Decay speed of the kernel. tau > 0
        m (float): Memory kernel parameter, m > 0
        M (int): Memory kernel parameter, M > 0

    Returns:
        np.ndarray: numpy array of timestamps
    """
    n = len(sample_vec)
    it = np.zeros(n)
    s = tau**(-1 - alpha) * (1 - np.power(m, -(1 + alpha) * M)) / (1 - np.power(m, -(1 + alpha)))
    z1 = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    z = z1 - s * tau * np.power(m, -1)
    etaoz = eta / z

    ak = tau * m**np.arange(0, M, 1)
    an1 = tau * m**(-1)
    akp1 = ak**(-alpha)

    it[0] = mu * sample_vec[0]
    tdi = sample_vec[1] - sample_vec[0]
    rk = np.ones(M)
    rk1 = 1
    summ = 0.
    for k in range(0, M):
        summ += akp1[k] * (1 - np.exp(-tdi / ak[k]))
    it[1] = it[0] + mu * tdi + etaoz * (summ - s * an1 * (1 - np.exp(-tdi / an1)))

    for i in range(2, n):
        tdi = sample_vec[i] - sample_vec[i-1]
        tdi2 = sample_vec[i-1] - sample_vec[i-2]
        summ = 0.
        for k in range(0, M):
            rk[k] = np.exp(-tdi2 / ak[k]) * rk[k] + 1
            summ += akp1[k] * rk[k] * (1 - np.exp(-tdi / ak[k]))

        rk1 = np.exp(-tdi2 / an1) * rk1 + 1
        it[i] = it[i-1] + mu * tdi + etaoz * (summ - s * an1 * rk1 * (1 - np.exp(-tdi / an1)))

    return it


@jit(float64[:](float64[:], float64, float64, float64[:]), nopython=True, cache=False, nogil=True)
def uvhp_sum_expo_compensator(sample_vec: np.ndarray, mu: float, eta: float, theta_vec: np.ndarray) -> np.ndarray:
    """Computes the compensator for a Hawkes procss with P-sum exponential kernel.

    Args:
        sample_vec (np.ndarray): numpy array of timestamps. Must be non-negative and sorted in ascending order.
        mu (float): background intensity of the Hawkes process. mu > 0
        eta (float): Branching ratio of the Hawkes process. 0 < eta < 1
        theta_vec (np.ndarray): Decay speed of the P exponential memory kernels. theta_vec > 0 all values

    Returns:
        np.ndarray: numpy array of timestamps
    """
    M = len(theta_vec)
    n = len(sample_vec)
    it = np.zeros(n)
    etaom = np.float64(eta / M)

    it[0] = mu * sample_vec[0]
    tdi = sample_vec[1] - sample_vec[0]
    rk = np.ones(M)
    summ = 0.
    for k in range(0, M):
        summ += 1 - np.exp(-tdi / theta_vec[k])

    it[1] = it[0] + mu * tdi + etaom * summ

    for i in range(2, n):
        tdi = sample_vec[i] - sample_vec[i-1]
        summ = 0.
        for k in range(0, M):
            rk[k] = np.exp(-(sample_vec[i-1] - sample_vec[i-2]) / theta_vec[k]) * rk[k] + 1
            summ += rk[k] * (1 - np.exp(-tdi / theta_vec[k]))

        it[i] = it[i-1] + mu * tdi + etaom * summ

    return it
