import numpy as np
from numba import jit, float64
from numba.types import int32


@jit(float64(float64[:], float64[:], float64), nopython=True, cache=False, nogil=True)
def uvhp_expo_logL(param_vec: np.ndarray, sample_vec: np.ndarray, tn: float) -> float:
    """ Log-likelihood function for a Hawkes Process with single exponential kernel

    Args:
        param_vec (np.ndarray): array of parameter: [mu, eta, theta]
        sample_vec (np.ndarray): array of timestamps.
                                 Must be sorted in ascending order. All timestamps must be positive.
        tn (float): End time of the Hawkes process.
                    Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1]

    Returns:
        float: The negative log-likelihood value
    """
    n = len(sample_vec)
    mu = param_vec[0]

    if n == 0:  # case of zero observation
        return mu * tn

    eta = param_vec[1]
    theta = param_vec[2]

    # initilize helper values
    h1 = np.exp(-(tn - sample_vec[0]) / theta) - 1
    h2 = np.log(mu)
    r1 = 0

    for i in range(1, n):
        r1 = np.exp(-(sample_vec[i] - sample_vec[i-1]) / theta) * (1 + r1)
        h1 += np.exp(-(tn - sample_vec[i]) / theta) - 1
        h2 += np.log(mu + eta * r1 / theta)

    logL = -mu * tn + eta * h1 + h2

    return -logL


@jit(float64[:](float64[:], float64[:], float64), nopython=True, cache=False, nogil=True)
def uvhp_expo_logL_grad(param_vec: np.ndarray, sample_vec: np.ndarray, tn: float) -> np.ndarray:
    """ Gradient of the log-likelihood function for a Hawkes Process with single exponential kernel

    Args:
        param_vec (np.ndarray): array of parameter: [mu, eta, theta]
        sample_vec (np.ndarray): array of timestamps.
                                 Must be sorted in ascending order. All timestamps must be positive.
        tn (float): End time of the Hawkes process.
                    Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1]

    Returns:
        np.ndarray: The negative value of each gradient: [mu gradient, eta gradient, theta gradient]
    """
    n = len(sample_vec)
    mu = param_vec[0]
    if n == 0:  # case of zero observations
        return np.array([tn, 0., 0.])

    eta = param_vec[1]
    theta = param_vec[2]

    # intilize the two recursives
    r1 = 0
    r2 = 0

    # intilize gradient values
    tdn = tn - sample_vec[0]
    extdn = np.exp(-tdn / theta)
    theta_sq = theta**2

    mu_grad = 1/mu - tn
    eta_grad = extdn - 1
    theta_grad = eta * tdn * extdn / theta_sq

    for i in range(1, n):
        # calculate helper values
        tdi = sample_vec[i] - sample_vec[i-1]
        tdn = tn - sample_vec[i]
        extdn = np.exp(-tdn / theta)
        extdi = np.exp(-tdi / theta)

        # calculate recursives
        r2 = extdi * (tdi * (1 + r1) + r2)
        r1 = extdi * (1 + r1)

        den = mu + eta * r1 / theta

        # the gradient calculation
        mu_grad += 1 / den
        eta_grad += (extdn - 1) + r1 / (theta * den)
        theta_grad += (r2 / theta - r1) / den - (tdn * extdn)

    theta_grad = theta_grad * eta / theta_sq

    return np.array([-mu_grad, -eta_grad, -theta_grad])


@jit(float64(float64[:], float64[:], float64), nopython=True, cache=False, nogil=True)
def uvhp_sum_expo_logL(param_vec: np.ndarray, sample_vec: np.ndarray, tn: float) -> float:
    """ Log-likelihood function for a Hawkes Process with P-sum exponential kernel

    Args:
        param_vec (np.ndarray): array of parameter: [mu, eta, theta1, theta2, ..., thetaP]
        sample_vec (np.ndarray): array of timestamps.
                                 Must be sorted in ascending order. All timestamps must be positive.
        tn (float): End time of the Hawkes process.
                    Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1]

    Returns:
        float: The negative log-likelihood value
    """
    n = len(sample_vec)
    mu = param_vec[0]

    if n == 0:  # case of zero observations
        return mu * tn

    eta = param_vec[1]
    theta_vec = param_vec[2::]
    M = len(theta_vec)

    # initilize helper values and recursives
    h1 = np.float64(0)
    tdn = tn - sample_vec[0]
    etaom = eta / M
    for k in range(0, M):
        h1 += (1 - np.exp(-tdn / theta_vec[k]))
    h2 = np.log(mu)
    rk = np.zeros(M)

    for i in range(1, n):
        tdi = sample_vec[i] - sample_vec[i-1]
        tdn = tn - sample_vec[i]
        summ = 0.
        for k in range(0, M):
            rk[k] = np.exp(-tdi / theta_vec[k]) * (1 + rk[k])
            h1 += (1 - np.exp(-tdn / theta_vec[k]))
            summ += rk[k] / theta_vec[k]

        h2 += np.log(mu + etaom * summ)

    logL = -mu * tn - etaom * h1 + h2

    return -logL


@jit(float64[:](float64[:], float64[:], float64), nopython=True, cache=False, nogil=True)
def uvhp_sum_expo_logL_grad(param_vec: np.ndarray, sample_vec: np.ndarray, tn: float) -> np.ndarray:
    """ Gradient of the log-likelihood function for a Hawkes Process with P-sum exponential kernel

    Args:
        param_vec (np.ndarray): array of parameter: [mu, eta, theta1, theta2, ..., thetaP]
        sample_vec (np.ndarray): array of timestamps. Must be sorted in ascending order.
                                 All timestamps must be positive.
        tn (float): End time of the Hawkes process.
                    Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1]

    Returns:
        np.ndarray: The negative value of each gradient:
                    [mu gradient, eta gradient, theta1 gradient, ..., thetaP gradient]
    """
    n = len(sample_vec)
    mu = param_vec[0]
    eta = param_vec[1]
    theta_vec = param_vec[2::]
    M = len(theta_vec)

    if n == 0:  # case of zero observations
        return np.append(np.array([tn, 0.]), np.zeros(M, dtype=np.float64))

    etaom = eta / M
    # intilize the two recursives
    rk1 = np.zeros(M)
    rk2 = np.zeros(M)

    h1_eta = 0.
    h2_eta = 0.

    h1_theta = np.zeros(M)
    h2_theta = np.zeros(M)

    theta_sq = theta_vec**2

    # intilize gradient values
    tdn = tn - sample_vec[0]
    extdn = np.exp(-tdn / theta_vec)
    for k in range(0, M):
        h1_eta += 1 - extdn[k]
        h1_theta[k] = tdn / theta_sq[k] * extdn[k]

    mu_grad = 1/mu - tn

    for i in range(1, n):
        # calculate helper values
        tdi = sample_vec[i] - sample_vec[i-1]
        tdn = tn - sample_vec[i]
        extdn = np.exp(-tdn / theta_vec)
        extdi = np.exp(-tdi / theta_vec)

        den_summ = np.float64(0)
        h2_eta_summ = np.float64(0)
        for k in range(0, M):
            # calculate recursives and compute current common denominator
            rk2[k] = extdi[k] * (tdi * (1 + rk1[k]) + rk2[k])
            rk1[k] = extdi[k] * (1 + rk1[k])
            den_summ += rk1[k] / theta_vec[k]
            h2_eta_summ += rk1[k] / theta_vec[k]
            h1_eta += 1 - extdn[k]

        # the gradient calculation
        den = mu + etaom * den_summ
        mu_grad += 1 / den
        h2_eta += h2_eta_summ / den
        for k in range(0, M):
            h1_theta[k] += tdn / theta_sq[k] * extdn[k]
            h2_theta[k] += ((rk2[k] / theta_vec[k] - rk1[k]) / theta_sq[k]) / den

    eta_grad = (h2_eta - h1_eta) / M
    theta_grad = (h2_theta - h1_theta) * etaom

    return np.append(np.append(-mu_grad, -eta_grad), -theta_grad)


@jit(float64(float64[:], float64[:], float64, float64, int32), nopython=True, cache=False, nogil=True)
def uvhp_approx_powl_cut_logL(param_vec: np.ndarray, sample_vec: np.ndarray, tn: float, m: float, M: int) -> float:
    """Log-likelihood function for a Hawkes Process with approximate power-law kernel with cutoff component


    Args:
        param_vec (np.ndarray): array of parameter: [mu, eta, theta1, theta2, ..., thetaP]
        sample_vec (np.ndarray): array of timestamps.
                                 Must be sorted in ascending order. All timestamps must be positive.
        tn (float): End time of the Hawkes process.
                    Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1]
        m (float): Fixed kernel variable
        M (int): Fixed kernel variable

    Returns:
        float: negative log-likelihood value
    """
    n = len(sample_vec)
    mu = param_vec[0]

    if n == 0:  # case of zero observations
        return mu * tn

    eta = param_vec[1]
    alpha = param_vec[2]
    tau = param_vec[3]

    # calculate fixed values
    s = tau**(-1 - alpha) * (1 - np.power(m, -(1 + alpha) * M)) / (1 - np.power(m, -(1 + alpha)))
    z1 = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    z = z1 - s * tau * np.power(m, -1)

    etaoz = eta / z
    ak = tau * m**np.arange(0, M, 1)
    an1 = tau * m**(-1)

    # initialize recursives
    rk = np.zeros(M)
    rn1 = 0.

    powna = np.zeros(M)
    pown1a = np.zeros(M)

    # initialize h1 and h2 using i=1
    tdn = tn - sample_vec[0]
    summ1 = 0.
    for k in range(0, M):
        # pre-compute powers:
        powna[k] = np.power(ak[k], -alpha)
        pown1a[k] = np.power(ak[k], -1 - alpha)
        # summ1 for i=1
        summ1 += powna[k] * (1 - np.exp(-tdn / ak[k]))

    h1 = summ1 - s * an1 * (1 - np.exp(-tdn / an1))
    h2 = np.log(mu)

    # loop over remaining observation starting at i=1
    for i in range(1, n):
        tdi = sample_vec[i] - sample_vec[i-1]
        tdn = tn - sample_vec[i]

        # update recursive rn1
        rn1 = np.exp(-tdi / an1) * (1 + rn1)
        summ1 = 0.
        summ2 = 0.
        for k in range(0, M):
            # update recursive rk[k]
            rk[k] = np.exp(-tdi / ak[k]) * (1 + rk[k])
            # compute first and second M sums
            summ1 += powna[k] * (1 - np.exp(-tdn / ak[k]))
            summ2 += pown1a[k] * rk[k]

        h1 += (summ1 - s * an1 * (1 - np.exp(-tdn / an1)))
        h2 += np.log(mu + etaoz * (summ2 - s * rn1))

    logL = -mu * tn - etaoz * h1 + h2

    return -logL


@jit(float64(float64[:], float64[:], float64, float64, int32), nopython=True, cache=False, nogil=True)
def uvhp_approx_powl_logL(param_vec: np.ndarray, sample_vec: np.ndarray, tn: float, m: float, M: int) -> float:
    """Log-likelihood function for a Hawkes Process with approximate power-law kernel without cutoff component

    Args:
        param_vec (np.ndarray): array of parameter: [mu, eta, theta1, theta2, ..., thetaP]
        sample_vec (np.ndarray): array of timestamps. Must be sorted in ascending order.
                                 All timestamps must be positive.
        tn (float): End time of the Hawkes process.
                    Must be larger or equal than the last timestamp i.e. tn >= sample_vec[-1]
        m (float): Fixed kernel variable
        M (int): Fixed kernel variable

    Returns:
        float: negative log-likelihood value
    """
    n = len(sample_vec)
    mu = param_vec[0]

    if n == 0:  # case of zero observations
        return mu * tn

    eta = param_vec[1]
    alpha = param_vec[2]
    tau = param_vec[3]
    Mf = np.float64(M)
    z = np.power(tau, -alpha) * (1 - np.power(m, -alpha * Mf)) / (1 - np.power(m, -alpha))
    ak = tau * m**np.arange(0, M, 1)
    etaoz = eta / z

    rk = np.zeros(M)
    powna = np.zeros(M)
    pown1a = np.zeros(M)
    h1 = 0.
    td1 = tn - sample_vec[0]
    for k in range(0, M):
        powna[k] = ak[k]**(-alpha)
        pown1a[k] = ak[k]**(-1 - alpha)
        h1 += powna[k] * (1 - np.exp(-td1 / ak[k]))

    h2 = np.log(mu)
    for i in range(1, n):
        tdn = tn - sample_vec[i]
        tdi = sample_vec[i] - sample_vec[i-1]

        summ2 = 0.
        for k in range(0, M):
            rk[k] = np.exp(-tdi / ak[k]) * (rk[k] + 1)
            h1 += powna[k] * (1 - np.exp(-tdn / ak[k]))
            summ2 += pown1a[k] * rk[k]

        h2 += np.log(mu + etaoz * summ2)

    return mu*tn + etaoz * h1 - h2
