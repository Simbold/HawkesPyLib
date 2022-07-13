# -*- coding: utf-8 -*-

import numpy as np #!/home/azureuser/miniconda/envs/thesis_compute/bin/python
import numba

@numba.jit(nopython=True)
def uvhp_expo_logL(param_vec, sample_vec, tn):
    """
    log-likelihood of a univariate Hawkes process with single exponential kernel
    :param param_vec: 1-d ndarray of paramters mu, eta and theta
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :param tn: the end time of the process
    :return: the negative log-likelihood value
    """
    n = len(sample_vec)
    mu = param_vec[0]  # the constant exogenous intensity
    eta = param_vec[1]  # the branching ratio
    theta = param_vec[2]  # intensity parameter of the delay density

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


@numba.jit(nopython=True)
def uvhp_expo_logL_grad(param_vec, sample_vec, tn):
    """
    Gradient of the log-likelihood of a univariate Hawkes process with single exponential kernel
    :param param_vec: 1-d ndarray of paramters mu, eta and theta respectively
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :param tn: the end time of the process
    :return: the negative gradient of the logL as a 1-d ndarray. Gradients w.r.t. mu, alpha, theta respectively
    """
    n = len(sample_vec)
    mu = param_vec[0]
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
        r2 = extdi * (tdi * (1 + r1) + r2) # r2 be calculated first since it uses r1 from the previous step
        r1 = extdi * (1 + r1)

        den = mu + eta * r1 / theta
        
        # the gradient calculation
        mu_grad += 1 / den
        eta_grad += (extdn - 1) + r1 / (theta * den)
        theta_grad += (r2 / theta - r1) / den - (tdn * extdn) 
    
    theta_grad = theta_grad * eta / theta_sq
    
    return np.array([-mu_grad, -eta_grad, -theta_grad])


@numba.jit(nopython=True)
def uvhp_sum_expo_logL(param_vec, sample_vec, tn, M):
    """
    log-likelihood of a univariate Hawkes process with sum of M expoential kernel
    :param param_vec: 1-d ndarray of paramters mu, eta and theta_vec
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :param tn: the end time of the process
    :param M: number of exponentials, must be equivalent to len(theta_vec)
    :return: the negative log-likelihood value
    """
    n = len(sample_vec)
    mu = param_vec[0]  # the constant exogenous intensity
    eta = param_vec[1]  
    theta_vec = param_vec[2::]  # intensity parameter of the delay density

    # initilize helper values
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


@numba.jit(nopython=True)
def uvhp_sum_expo_logL_grad(param_vec, sample_vec, tn, M):
    """
    Gradient of the log-likelihood of a univariate Hawkes process with sum of M exponential kernels
    :param param_vec: 1-d ndarray of paramters mu, eta and theta respectively
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :param tn: the end time of the process
    :param M: number of exponentials, must be equivalent to len(theta_vec)
    :return: the negative gradient of the logL as a 1-d ndarray. Gradients w.r.t. mu, alpha, theta_vec respectively
    """
    n = len(sample_vec)
    mu = param_vec[0]  # the constant exogenous intensity
    eta = param_vec[1]  
    theta_vec = param_vec[2::]  # intensity parameter of the delay density
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


@numba.jit(nopython=True)
def uvhp_approx_powl2_logL(param_vec, sample_vec, tn, m, M):
    """
    log-likelihood of a univariate Hawkes process with approximate power-law kernel with short-lag cutoff (PWc) kernel
    :param param_vec: 1-d ndarray of paramters mu, eta, alpha and tau
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :param tn: the end time of the process
    :return: the negative log-likelihood value
    """
    n = len(sample_vec)
    
    # parameters to be estimated
    mu = param_vec[0]
    eta = param_vec[1]
    alpha = param_vec[2] # power law exponent alpha
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



@numba.jit(nopython=True)
def uvhp_approx_powl1_logL(param_vec, sample_vec, tn, m, M):
    """
    log-likelihood of a univariate Hawkes process with approximate power-law kernel without short-lag cutoff (PWc) kernel
    :param param_vec: 1-d ndarray of paramters mu, eta, alpha and tau
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :param tn: the end time of the process
    :return: the negative log-likelihood value
    """
    n = len(sample_vec)
    # parameters to be estimated
    mu = param_vec[0]
    eta = param_vec[1]
    alpha = param_vec[2] # power law exponent alpha
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
        h1 += powna[k] * (1 -  np.exp(-td1 / ak[k]))

    h2 = np.log(mu)
    for i in range(1, n):
        tdn = tn - sample_vec[i]
        tdi = sample_vec[i] - sample_vec[i-1]

        summ2 = 0.
        for k in range(0, M):
            rk[k] = np.exp(-tdi / ak[k]) * (rk[k] + 1)
            h1 += powna[k] * (1 -  np.exp(-tdn / ak[k]))
            summ2 += pown1a[k] * rk[k]

        h2 += np.log(mu + etaoz * summ2)

    return mu*tn + etaoz * h1 - h2



@numba.jit(nopython=True)
def eta_approx_mc(timestamps):
    """
    Approximate branching ratio estimator using Monte Carlo samples
    :param timestamps: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :return: approximate branchig ratio 
    """
    n = len(timestamps)
    tn = timestamps[-1]-timestamps[0]
    w = tn/3
    aw = w * n / tn # the average number of events per window of length w
    
    sigma2 = 0
    sam = 100
    c = np.random.uniform(timestamps[0] + w,timestamps[-1], sam)
    for i in range(0, sam):
        cp = c[i]
        nw = len(timestamps[(timestamps >= cp - w) & (timestamps <= cp)])
        sigma2 += np.power(nw - aw, 2)
        #print(nw)
    
    sigma2 = sigma2 / (sam-1)
    eta_apr = 1 - np.sqrt(aw / sigma2)
    return max(min(1, eta_apr), 0)


@numba.jit(nopython=True)
def uvhp_expo_int_density(sample_vec, mu, eta, theta):
    """
    Computes residual process (or compensator) i.e. for the Hawkes process with single exponential kernel
    i.e. integrates the estimated intensity from 0 -> sample_vec[i] for all i
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :params mu, eta, theta: paramters specifiying the single expo kernel
    :return: numpy 1d-array containing the residual process
    """
    n = len(sample_vec)
    it = np.zeros(n)

    it[0] = mu * sample_vec[0]
    td1 = sample_vec[1] - sample_vec[0]
    rk =  1
    it[1] = it[0] + mu * td1 + eta * (1 - np.exp(-td1 / theta)) * rk

    for i in range(2, n):
        tdi = sample_vec[i] - sample_vec[i-1]
        rk = np.exp(-(sample_vec[i-1] - sample_vec[i-2]) / theta) * rk + 1
        it[i] = it[i-1] + mu * tdi + eta * rk * (1 - np.exp(-tdi / theta)) 

    return it


@numba.jit(nopython=True)
def uvhp_approx_powl1_int_density(sample_vec, mu, eta, alpha, tau, m , M):
    """
    Computes residual process (or compensator) i.e. for the Hawkes process with approximate power-law kernel without cutoff
    i.e. integrates the estimated intensity from 0 -> sample_vec[i] for all i
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :params mu, eta, alpha, tau, m, M: paramters specifiying the approximate power-law kernel without cutoff
    :return: numpy 1d-array containing the residual process
    """
    n = len(sample_vec)
    it = np.zeros(n)
    z = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    etaoz = eta / z

    ak = tau * m**np.arange(0, M, 1)
    akp1 = ak**(-alpha)

    it[0] = mu * sample_vec[0]
    td1 = sample_vec[1] - sample_vec[0]
    rk =  np.ones(M)
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


@numba.jit(nopython=True)
def uvhp_approx_powl2_int_density(sample_vec, mu, eta, alpha, tau, m , M):
    """
    Computes residual process (or compensator) i.e. for the Hawkes process with approximate power-law kernel with cutoff component
    i.e. integrates the estimated intensity from 0 -> sample_vec[i] for all i
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :params mu, eta, alpha, tau, m, M: paramters specifiying the approximate power-law kernel without cutoff
    :return: numpy 1d-array containing the residual process
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
    rk =  np.ones(M)
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


@numba.jit(nopython=True)
def uvhp_sum_expo_int_density(sample_vec, mu, eta, theta_vec, M):
    """
    Computes residual process (or compensator) i.e. for the Hawkes process with sum of exponential kernel
    i.e. integrates the estimated intensity from 0 -> sample_vec[i] for all i
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :params mu, eta, theta_vec: paramters specifiying the sum of expo kernel
    :return: numpy 1d-array containing the residual process
    """    
    n = len(sample_vec)
    it = np.zeros(n)
    etaom = np.float64(eta / M)

    it[0] = mu * sample_vec[0]
    tdi = sample_vec[1] - sample_vec[0]
    rk =  np.ones(M)
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






def h_approx_powl1(t, alpha, tau, m, M):
    # offspring density approximate power-law with cutoff
    s = tau**(-1 - alpha) * (1 - np.power(m, -(1 + alpha) * M)) / (1 - np.power(m, -(1 + alpha)))
    z1 = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    z = z1 - s * tau * np.power(m, -1)

    ak = tau * m**np.arange(0, M, 1)
    an1 = tau / m
    summ = 0
    for k in range(0, M):
        summ += ak[k]**(-1 - alpha) * np.exp(-t / ak[k])
    return (summ - s * np.exp(-t / an1)) / z


def h_approx_powl2(t, alpha, tau, m, M):
    # offspring density approximate power-law without cutoff
    z = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    ak = tau * m**np.arange(0, M, 1)
    summ = 0
    for k in range(0, M):
        summ += ak[k]**(-1 - alpha) * np.exp(-t / ak[k])
    return summ / z


def h_exp(t, theta):
    # offspring density  single expo kernel
    return np.exp(-t/theta) / theta




