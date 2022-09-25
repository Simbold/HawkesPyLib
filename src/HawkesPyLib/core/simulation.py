import numpy as np
from numba import jit, float64
from numba.types import int32


@jit(float64[:](float64, float64, int32), nopython=False, cache=False, nogil=True)
def homogenous_poisson_simulator(T: float, mu: float, seed: int = 0) -> np.ndarray:
    """Simulates a homogenous Poisson process with constant intensity mu.

    Args:
        T (float): Time until which the Poisson process will be simulated. T > 0
        mu (float): Constant intensity of the homogenous Poisson process. mu > 0
        seed (int, optional): Seed for numba's random number generator. Defaults to 0 (no seed).

    Returns:
        np.ndarray: Array of simulated timestamps
    """
    if seed != 0:
        np.random.seed(seed)

    # generate first event
    u = 1 - np.random.rand(1)[0]
    s = -np.log(u) / mu
    if s > T:
        return np.empty(0)

    arr_size = np.int64(mu * T * 2)  # Allocate two times the expected number of arrival times (memory is cheap?)
    t_arr = np.empty(arr_size, np.float64)
    t_arr[0] = s
    n = 0
    while s < T:
        n += 1
        u = 1 - np.random.rand(1)[0]
        w = -np.log(u) / mu
        s = s + w
        t_arr[n] = s

    if t_arr[n] > T:
        return t_arr[0:n]
    else:
        return t_arr[0:(n+1)]


@jit(float64[:](float64, float64, float64, float64, float64, float64, int32, int32),
     nopython=False, cache=False, nogil=True, fastmath=True)
def uvhp_approx_powl_cutoff_simulator(T: float, mu: float, eta: float, alpha: float,
                                      tau: float, m: float, M: int, seed: int = 0) -> np.ndarray:
    """Simulates a Hawkes process with approximate power-law memory kernel.
        Implements Ogata's modified thinning algorithm as described in algorithm 2 in (Ogata 1981) .

        References:
    - Ogata, Y. (1981). On lewis simulation method for point processes.
      IEEE transactions on information theory, 27(1):2331.

    Args:
        T (float): Time until which the Hawkes process will be simulated. T > 0
        mu (float): Background intensity of the Hawkes process. mu > 0
        eta (float): Branching ratio of the Hawkes process. 0 < eta < 1
        alpha (float): Power-law coefficient of the memory kernel. alpha > 0
        tau (float): Memory kernel parameter. tau > 0
        m (float):  Memory kernel variable. m > 0
        M (int):  Memory kernel variable. M > 0
        seed (int, optional): Seed for numba's random number generator. Defaults to 0 (no seed).

    Returns:
        np.ndarray: Array of simulated timestamps
    """
    if seed != 0:
        np.random.seed(seed)

    # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return np.empty(0)  # TODO: Maybe specify error code (T to small for given mu)

    # calculate fixed values
    sc = tau**(-1 - alpha) * (1 - m**(-(1 + alpha) * M)) / (1 - m**(-(1 + alpha)))
    z1 = tau**(-alpha) * (1 - m**(-alpha * M)) / (1 - m**(-alpha))
    z = z1 - sc * tau * m**(-1)

    etaoz = eta / z
    an1 = tau * m**(-1)
    ak = tau * m**np.arange(0, M, 1)
    akp1 = ak**(-1-alpha)
    prec = etaoz * np.sum(akp1)

    avg_n = mu / (1-eta) * T
    t_arr = np.empty(np.int64(avg_n*5), np.float64)  # allocate sufficient space for the result array
    i = 0
    tn1 = np.float64(s)  # time of the last accepted event
    t_arr[i] = tn1
    rk = np.zeros(M)  # initilize recursions
    rn1 = 0.
    ul = np.float64(mu + prec)  # initial intensity upper limit
    summ = np.float64(0)
    exps = np.float64(0)
    while s < T:
        # calculate prospective new event time s
        u = 1 - np.random.rand(2)
        s += -np.log(u[0]) / ul

        # evaluate intensity at new event time s
        exps = np.exp(-(s-tn1)/ak)
        expsn1 = np.exp(-(s-tn1)/an1)
        summ = np.float64(0)
        for k in range(0, M):
            summ += akp1[k] * exps[k] * (rk[k] + 1)

        lamb = mu + etaoz * (summ - sc * expsn1 * (rn1 + 1))
        if u[1] <= (lamb / ul):  # test if new event accepted or rejected
            i += 1
            t_arr[i] = s

            # update values for next iteration
            rn1 = expsn1 * (rn1 + 1)
            summ = np.float64(0)
            for k in range(0, M):
                rk[k] = exps[k] * (rk[k] + 1)
                summ += akp1[k] * (rk[k] + 1)

            ul = mu + etaoz * summ  # setting s=0 such that ul >> lamb for all t
            tn1 = s

        else:  # new event rejected: modify upper limit using modified intensity
            ul = mu + etaoz * summ

    # check if last accepted event exceeds T
    if t_arr[i] > T:
        return t_arr[0:i]
    return t_arr[0:(i+1)]


@jit(float64[:](float64, float64, float64, float64, int32), nopython=True, cache=False, nogil=True, fastmath=True)
def uvhp_expo_simulator(T: float, mu: float, eta: float, theta: float, seed: int = 0) -> np.ndarray:
    """Simulates a Hawkes process with single exponential memory kernel.
        Implements Ogata's modified thinning algorithm as described in algorithm 2 in (Ogata 1981).

        References:

    - Ogata, Y. (1981). On lewis simulation method for point processes.
      IEEE transactions on information theory, 27(1):2331.

    Args:
        T (float): Time until which the Hawkes process will be simulated. T > 0
        mu (float): Background intensity of the Hawkes process. mu > 0
        eta (float): Branching ratio of the Hawkes process. 0 < eta < 1
        theta (float): Decay speed of kernel. theta > 0
        seed (_type_, optional): Seed numbda's random number generator. Defaults to 0 (no seed).

    Returns:
        np.ndarray: Array of simulated timestamps
    """
    if seed != 0:
        np.random.seed(seed)

    # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return np.empty(0)  # TODO: Maybe specify error code (T to small for given mu)

    prec = np.float64(eta / theta)
    avg_n = mu / (1-eta) * T
    t_arr = np.empty(np.int64(avg_n*5), np.float64)  # Allocate a sufficiently large array
    i = 0
    tn1 = np.float64(s)  # time of the last accepted event
    t_arr[i] = tn1
    rk = np.float64(0)  # initilize recursion
    ul = np.float64(mu + prec)  # initial intensity upper limit
    while s < T:
        # calculate prospective new event time s
        u = 1 - np.random.rand(2)
        s += -np.log(u[0]) / ul

        # evaluate intensity at new event time s
        exn = np.exp(-(s-tn1)/theta)
        lamb = mu + prec * exn * (rk + 1)
        if u[1] <= (lamb / ul):  # acception / rejection test
            i += 1
            t_arr[i] = s

            #  update values for next iteration
            rk = exn * (rk + 1)
            ul = mu + prec * (rk + 1)
            tn1 = s

        else:  # modify upper limit
            ul = lamb

    # check if last accepted event exceeds T
    if t_arr[i] > T:
        return t_arr[0:i]
    return t_arr[0:(i+1)]


@jit(float64[:](float64, float64, float64, float64, float64, float64, int32, int32),
     nopython=True, cache=False, nogil=True, fastmath=True)
def uvhp_approx_powl_simulator(T: float, mu: float, eta: float, alpha: float,
                               tau: float, m: float, M: int, seed: int = 0) -> np.ndarray:
    """Simulates a Hawkes process with approximate power-law memory kernel.
        Implements Ogata's modified thinning algorithmas described in algorithm 2 in (Ogata 1981).

        References:

    - Ogata, Y. (1981). On lewis simulation method for point processes.
      IEEE transactions on information theory, 27(1):2331.

    Args:
        T (float): Time until which the Hawkes process will be simulated. T > 0
        mu (float): Background intensity of the Hawkes process. mu > 0
        eta (float): Branching ratio of the Hawkes process. 0 < eta < 1
        alpha (float): Power-law coefficient of the memory kernel. alpha > 0
        tau (float): Memory kernel parameter. tau > 0
        m (float):  Memory kernel variable. m > 0
        M (int):  Memory kernel variable. M > 0
        seed (int, optional): Seed for numba's random number generator. Defaults to 0 (no seed).

    Returns:
        np.ndarray: Array of simulated timestamps
    """
    if seed != 0:
        np.random.seed(seed)

    # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return np.empty(0)  # TODO: Maybe specify error code (T to small for given mu)

    z = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    etaoz = eta / z
    ak = tau * m**np.arange(0, M, 1)
    akp1 = ak**(-1-alpha)
    prec = etaoz * np.sum(akp1)  # jump size

    avg_n = mu / (1-eta) * T
    t_arr = np.empty(np.int64(avg_n*5), np.float64)  # Allocate sufficiently large result array
    i = 0
    tn1 = np.float64(s)  # time of the last accepted event
    t_arr[i] = tn1
    rk = np.zeros(M)  # initilize recursion
    ul = np.float64(mu + prec)  # initial intensity upper limit
    summ = np.float64(0)
    exps = np.float64(0)
    while s < T:
        # calculate prospective new event time s
        u = 1 - np.random.rand(2)
        s += -np.log(u[0]) / ul
        # evaluate intensity at new event time s
        exps = np.exp(-(s-tn1)/ak)
        summ = np.float64(0)
        for k in range(0, M):
            summ += akp1[k] * exps[k] * (rk[k] + 1)

        lamb = mu + etaoz * summ
        if u[1] <= (lamb / ul):  # rejection acceptance test
            i += 1
            t_arr[i] = s

            # update values for next iteration
            summ = np.float64(0)
            for k in range(0, M):
                rk[k] = exps[k] * (rk[k] + 1)
                summ += akp1[k] * (rk[k] + 1)

            ul = mu + etaoz * summ
            tn1 = s

        else:  # modify upper limit
            ul = lamb

    # check if last accepted event exceeds T
    if t_arr[i] > T:
        return t_arr[0:i]
    return t_arr[0:(i+1)]


@jit(float64[:](float64, float64, float64, float64[:], int32), nopython=True, cache=False, nogil=True, fastmath=True)
def uvhp_sum_expo_simulator(T: float, mu: float, eta: float, theta_vec: np.ndarray, seed: int = 0) -> np.ndarray:
    """Simulates a Hawkes process with P-sum expoential memory kernel.
        Implements Ogata's modified thinning algorithmas described in algorithm 2 in (Ogata 1981).

        References:

    - Ogata, Y. (1981). On lewis simulation method for point processes.
      IEEE transactions on information theory, 27(1):2331.

    Args:
        T (float): Time until which the Hawkes process will be simulated. T > 0
        mu (float): Background intensity of the Hawkes process. mu > 0
        eta (float): Branching ratio of the Hawkes process. 0 < eta < 1
        theta_vec (np.ndarray): Array of P exponential decay speeds. theta_vec > 0
        seed (int, optional): Seed for numba's random number generator. Defaults to 0 (no seed).

    Returns:
        np.ndarray: Array of simulated timestamps
    """
    if seed != 0:
        np.random.seed(seed)

    # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return np.empty(0)  # TODO: Maybe specify error code (T to small for given mu)

    P = len(theta_vec)
    etaom = eta / P
    prec = etaom * np.sum(1/theta_vec)  # jump size

    avg_n = mu / (1-eta) * T
    t_arr = np.empty(np.int64(avg_n*5), np.float64)  # Allocate sufficiently large array for simulated timestamps
    i = 0

    tn1 = np.float64(s)  # time of the last accepted event
    t_arr[i] = tn1

    rk = np.zeros(P)  # initilize recursion
    ul = np.float64(mu + prec)  # initial intensity upper limit
    exps = np.float64(0)
    while s < T:
        # calculate prospective new event time s
        u = 1 - np.random.rand(2)
        s += -np.log(u[0]) / ul

        # evaluate intensity at new event time s
        exps = np.exp(-(s-tn1)/theta_vec)
        summ = np.float64(0)
        for k in range(0, P):
            summ += 1 / theta_vec[k] * exps[k] * (rk[k] + 1)
        lamb = mu + etaom * summ
        if u[1] <= (lamb / ul):  # new event accepted
            i += 1
            t_arr[i] = s

            # update values for next iteration
            summ = np.float64(0)
            for k in range(0, P):
                rk[k] = exps[k] * (rk[k] + 1)
                summ += 1 / theta_vec[k] * (rk[k] + 1)
            ul = mu + etaom * summ
            tn1 = s

        else:  # modify upper limit
            ul = lamb

    # check if last accepted event exceeds T
    if t_arr[i] > T:
        return t_arr[0:i]
    return t_arr[0:(i+1)]
