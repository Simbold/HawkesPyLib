import numpy as np
from numba import jit, float64
from numba.types import int32


@jit(float64[:](float64, float64), nopython=True, cache=False, nogil=True)
def generate_eval_grid(step_size: float, T: float) -> np.ndarray:
    """Generates an equidistant grid in the closed interval [0, T]
    with step size given by step_size.

    Args:
        step_size (float): Step size of the equidistant grid.
        T (float): End point of the equidistant grid (the last value).

    Returns:
        np.ndarray: 1d array equidistant grid between from 0 to t with step size.
    """
    num = round(T / step_size) + 1
    grid = np.linspace(0., T, num)
    return grid


@jit(float64[:, :](float64[:], float64[:], float64, float64, float64),
     nopython=True, cache=False, nogil=True)
def uvhp_expo_intensity(sample_vec: np.ndarray, grid: np.ndarray, mu: float,
                        eta: float, theta: float) -> np.ndarray:
    """ Evaluation of the intensity function of a univariate Hawkes process with single exponential kernel

    Args:
        sample_vec (np.ndarray): Jump times of the Hawkes process.
                                 Must be non-negative and in ascending order!
        grid (np.ndarray): Times at which the intensity function is evaluated at.
                           Must be non-negative and in ascending order
        mu (float): Background intensity of the Hawkes process, mu > 0
        eta (float): Branching ratio of the Hawkes process, 0 > eta < 1
        theta (float): Decay speed of the single exponential kernel, theta > 0

    Returns:
        np.ndarray: A 2d numpy array of timestamps (column 0) and corresponding intensity values (column 1)
    """
    eval_times = np.unique(np.hstack((grid, sample_vec)))  # sorted unique evaluation times
    intensity = np.zeros((len(eval_times), 2), dtype=np.float64)
    intensity[:, 0] = eval_times

    prec = np.float64(eta / theta)
    rk = np.float64(0)
    n = len(sample_vec)

    # first compute intensity up to first jump
    next_jump = sample_vec[0]
    idx = np.where(intensity[:, 0] == next_jump)[0][0]
    intensity[0:idx, 1] = mu
    intensity[idx, 1] = np.float64(mu + prec)

    j = 1
    last_jump = next_jump
    next_jump = sample_vec[j]

    for i in range((idx+1), intensity.shape[0]):
        t = intensity[i, 0]
        exn = np.exp(-(t-last_jump)/theta)

        if t == next_jump:  # case of event arrival
            rk = exn * (rk + 1)
            intensity[i, 1] = mu + prec * (rk + 1)
            j = min(j + 1, n-1)
            last_jump = next_jump
            next_jump = sample_vec[j]

        else:  # case of no event arrival
            intensity[i, 1] = mu + prec * exn * (rk + 1)

    return intensity


@jit(float64[:, :](float64[:], float64[:], float64, float64, float64[:]),
     nopython=True, cache=False, nogil=True)
def uvhp_sum_expo_intensity(sample_vec: np.ndarray, grid: np.ndarray, mu: float,
                            eta: float, theta_vec: np.ndarray) -> np.ndarray:
    """ Evaluation of the intensity function of a univariate Hawkes process
        with P-sum exponential kernel

    Args:
        sample_vec (np.ndarray): Jump times of the Hawkes process.
                                 Must be non-negative and in ascending order!
        grid (np.ndarray): Times at which the intensity function is evaluated at.
                           Must be non-negative and in ascending order
        mu (float): Background intensity of the Hawkes process, mu > 0
        eta (float): Branching ratio of the Hawkes process, 0 > eta < 1
        theta_vec (np.ndarray): Array of decay speeds of the P-sum exponential kernel, theta_k > 0

    Returns:
        np.ndarray: A 2d numpy array of timestamps (column 0) and corresponding intensity values (column 1)
    """
    eval_times = np.unique(np.hstack((grid, sample_vec)))  # sorted unique evaluation times
    intensity = np.zeros((len(eval_times), 2), dtype=np.float64)
    intensity[:, 0] = eval_times

    P = len(theta_vec)
    etaom = np.float64(eta / P)
    prec = etaom * np.sum(1/theta_vec)  # jump size
    rk = np.zeros(P)
    n = len(sample_vec)

    # first compute intensity up to first jump
    next_jump = sample_vec[0]
    idx = np.where(intensity[:, 0] == next_jump)[0][0]
    intensity[0:idx, 1] = mu
    intensity[idx, 1] = np.float64(mu + prec)

    j = 1
    last_jump = next_jump
    next_jump = sample_vec[j]

    for i in range((idx+1), intensity.shape[0]):
        t = intensity[i, 0]
        exps = np.exp(-(t-last_jump)/theta_vec)

        if t == next_jump:  # case of event arrival

            summ = np.float64(0)
            for k in range(0, P):
                rk[k] = exps[k] * (rk[k] + 1)
                summ += 1 / theta_vec[k] * (rk[k] + 1)

            intensity[i, 1] = mu + etaom * summ
            j = min(j + 1, n-1)
            last_jump = next_jump
            next_jump = sample_vec[j]

        else:  # case of no event arrival

            summ = np.float64(0)
            for k in range(0, P):
                summ += 1 / theta_vec[k] * exps[k] * (rk[k] + 1)

            intensity[i, 1] = mu + etaom * summ

    return intensity


@jit(float64[:, :](float64[:], float64[:], float64, float64, float64, float64, float64, int32),
     nopython=True, cache=False, nogil=True)
def uvhp_approx_powl_cutoff_intensity(sample_vec: np.ndarray, grid: np.ndarray, mu: float, eta: float,
                                      alpha: float, tau: float, m: float, M: int) -> np.ndarray:
    """ Evaluation of the intensity function of a univariate Hawkes process
        with approximate power-law kernel with smooth cutoff component

    Args:
        sample_vec (np.ndarray): Jump times of the Hawkes process. Must be non-negative and in ascending order!
        grid (np.ndarray): Times at which the intensity function is evaluated at.
                           Must be non-negative and in ascending order
        mu (float): Background intensity of the Hawkes process, mu > 0
        eta (float): Branching ratio of the Hawkes process, 0 > eta < 1
        alpha (float): Power-law coefficient, alpha > 0
        tau (float): Approximate location of cutoff, tau > 0
        m (float): Approximate power-law parameter, m > 0
        M (int): Number of weighted exponential kernels that approximate the power-law

    Returns:
        np.ndarray: A 2d numpy array of timestamps (column 0) and corresponding intensity values (column 1)
    """
    eval_times = np.unique(np.hstack((grid, sample_vec)))  # sorted unique evaluation times
    intensity = np.zeros((len(eval_times), 2), dtype=np.float64)
    intensity[:, 0] = eval_times

    # calculate fixed values
    sc = tau**(-1 - alpha) * (1 - m**(-(1 + alpha) * M)) / (1 - m**(-(1 + alpha)))
    z1 = tau**(-alpha) * (1 - m**(-alpha * M)) / (1 - m**(-alpha))
    z = z1 - sc * tau * m**(-1)

    etaoz = eta / z
    an1 = tau * m**(-1)
    ak = tau * m**np.arange(0, M, 1)
    akp1 = ak**(-1-alpha)
    prec = etaoz * (np.sum(akp1) - sc)
    rk = np.zeros(M)
    rn1 = np.float64(0)
    n = len(sample_vec)

    # first compute intensity up to first jump
    next_jump = sample_vec[0]
    idx = np.where(intensity[:, 0] == next_jump)[0][0]
    intensity[0:idx, 1] = mu
    intensity[idx, 1] = np.float64(mu + prec)

    j = 1
    last_jump = next_jump
    next_jump = sample_vec[j]

    for i in range((idx+1), intensity.shape[0]):
        t = intensity[i, 0]
        exps = np.exp(-(t-last_jump)/ak)
        expsn1 = np.exp(-(t-last_jump)/an1)

        if t == next_jump:  # case of event arrival

            rn1 = expsn1 * (rn1 + 1)
            summ = np.float64(0)
            for k in range(0, M):
                rk[k] = exps[k] * (rk[k] + 1)
                summ += akp1[k] * (rk[k] + 1)

            intensity[i, 1] = mu + etaoz * (summ - sc * (rn1 + 1))
            j = min(j + 1, n-1)
            last_jump = next_jump
            next_jump = sample_vec[j]

        else:  # case of no event arrival

            summ = np.float64(0)
            for k in range(0, M):
                summ += akp1[k] * exps[k] * (rk[k] + 1)

            intensity[i, 1] = mu + etaoz * (summ - sc * expsn1 * (rn1 + 1))

    return intensity


@jit(float64[:, :](float64[:], float64[:], float64, float64, float64, float64, float64, int32),
     nopython=True, cache=False, nogil=True)
def uvhp_approx_powl_intensity(sample_vec: np.ndarray, grid: np.ndarray, mu: float, eta: float,
                               alpha: float, tau: float, m: float, M: int) -> np.ndarray:
    """ Evaluation of the intensity function of a univariate Hawkes process with approximate power-law kernel.

    Args:
        sample_vec (np.ndarray): Jump times of the Hawkes process. Must be non-negative and in ascending order!
        grid (np.ndarray): Times at which the intensity function is evaluated at.
                           Must be non-negative and in ascending order.
        mu (float): Background intensity of the Hawkes process, mu > 0
        eta (float): Branching ratio of the Hawkes process, 0 > eta < 1
        alpha (float): Power-law coefficient, alpha > 0
        tau (float): Approximate location of cutoff, tau > 0
        m (float): Approximate power-law parameter, m > 0
        M (int): Number of weighted exponential kernels that approximate the power-law

    Returns:
        np.ndarray: A 2d numpy array of timestamps (column 0) and corresponding intensity values (column 1)
    """
    eval_times = np.unique(np.hstack((grid, sample_vec)))         # sorted unique evaluation times
    intensity = np.zeros((len(eval_times), 2), dtype=np.float64)
    intensity[:, 0] = eval_times

    # calculate fixed values
    z = tau**(-alpha) * (1 - m**(-alpha * M)) / (1 - m**(-alpha))
    etaoz = eta / z
    ak = tau * m**np.arange(0, M, 1)
    akp1 = ak**(-1-alpha)
    prec = etaoz * np.sum(akp1)
    rk = np.zeros(M)
    n = len(sample_vec)

    # first compute intensity up to first jump
    next_jump = sample_vec[0]
    idx = np.where(intensity[:, 0] == next_jump)[0][0]
    intensity[0:idx, 1] = mu
    intensity[idx, 1] = np.float64(mu + prec)

    j = 1
    last_jump = next_jump
    next_jump = sample_vec[j]

    for i in range((idx+1), intensity.shape[0]):
        t = intensity[i, 0]
        exps = np.exp(-(t-last_jump)/ak)

        if t == next_jump:  # case of event arrival

            summ = np.float64(0)
            for k in range(0, M):
                rk[k] = exps[k] * (rk[k] + 1)
                summ += akp1[k] * (rk[k] + 1)

            intensity[i, 1] = mu + etaoz * summ
            j = min(j + 1, n-1)
            last_jump = next_jump
            next_jump = sample_vec[j]

        else:  # case of no event arrival

            summ = np.float64(0)
            for k in range(0, M):
                summ += akp1[k] * exps[k] * (rk[k] + 1)

            intensity[i, 1] = mu + etaoz * summ

    return intensity
