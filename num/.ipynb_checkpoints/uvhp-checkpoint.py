#!/home/azureuser/miniconda/envs/thesis_compute/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numba
from scipy.optimize import fmin_l_bfgs_b
from scipy import stats
import statsmodels.api as sm
rng = np.random.default_rng()

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


def maximize_logL(sample_vec, kernel, tn=None, rng=rng, **kwargs):
    """
    Estimation of a univariate Hawkes process via maximization of the log-likelihood
    The intensity of the process is parameterized as follows: 
        \lambda(t) = mu + eta * sum_{t_i < t} g(t - t_i)
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :param kernel: "expo", "sum-expo", "powlaw-cutoff", "powlaw"
    :param tn: the end time of the process
    :param rng: random number generator
    :return: a dictionary containing the results and additional values
    """
    # if no end/max time given use the last observation as tn
    if isinstance(tn, type(None)):
        tn = sample_vec[-1]
    
    max_opt = 15
    n_opt = 0
    succ_flag = None
    eta0 = eta_approx_mc(sample_vec)    
    while (succ_flag != 0) & (n_opt <=max_opt):
        n_opt += 1
        mu0 = len(sample_vec) * (1 - eta0) / tn
        if kernel == "expo":
            # starting parameters 
            theta0 = rng.random(1)[0] * 0.1
            param_vec0 = np.array((mu0, eta0, theta0), dtype=np.float64)
            # parameter bounds
            bnds = [(1e-5, np.inf), (1e-5, 1.0), (1e-5, np.inf)]
            # minimize negative logL using scipy L-BFGS-B minimization routine
            opt_res = fmin_l_bfgs_b(func=uvhp_expo_logL, x0=param_vec0, fprime=uvhp_expo_logL_grad,
                                    args=(sample_vec, tn), approx_grad=False, bounds=bnds,
                                    m=10, factr=100, pgtol=1e-05, iprint=- 1, maxfun=15000,
                                    maxiter=15000, disp=None, callback=None, maxls=20)

        if kernel == "sum-expo":
            P = kwargs.get("P")
            # starting parameters 
            theta0_vec = rng.random(P) * 0.1
            param_vec0 = np.append(np.append(mu0, eta0), theta0_vec)
            
            # parameter bounds
            bnds = [(1e-5, np.inf), (1e-5, 1)]
            for k in range(0, P):
                bnds.append((1e-5, np.inf))

            opt_res = fmin_l_bfgs_b(func=uvhp_sum_expo_logL, x0=param_vec0, fprime=uvhp_sum_expo_logL_grad,
                                    args=(sample_vec, tn, P), approx_grad=False, bounds=bnds,
                                    m=10, factr=100, pgtol=1e-05, iprint=- 1, maxfun=15000,
                                    maxiter=15000, disp=None, callback=None, maxls=20)

        elif kernel == "powlaw-cutoff":
            # starting paramters
            alpha0 = rng.random(1)[0]
            tau0 = rng.random(1)[0] * 0.01
            param_vec0 = np.array([mu0, eta0, alpha0, tau0], dtype=np.float64)
            # parameter bounds
            bnds = [(1e-8, np.inf), (1e-8, 1.0), (1e-8, 10), (1e-8, np.inf)]
            m = kwargs.get("m")
            M = kwargs.get("M") 
            # L-BFGS-B 
            opt_res = fmin_l_bfgs_b(func=uvhp_approx_powl2_logL, x0=param_vec0, fprime=None,
                                    args=(sample_vec, tn, m, M), approx_grad=True, bounds=bnds,
                                    m=10, factr=100, epsilon=1e-07, pgtol=1e-05, iprint=- 1,
                                    maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)

        elif kernel == "powlaw":
            # starting paramters
            alpha0 = rng.random(1)[0]
            tau0 = rng.random(1)[0] * 0.01
            param_vec0 = np.array([mu0, eta0, alpha0, tau0], dtype=np.float64)
            # parameter bounds
            bnds = [(1e-8, np.inf), (1e-8, 1.0), (1e-8, 10), (1e-8, np.inf)]
            m = kwargs.get("m")
            M = kwargs.get("M") 
            # L-BFGS-B
            opt_res = fmin_l_bfgs_b(func=uvhp_approx_powl1_logL, x0=param_vec0, fprime=None,
                                    args=(sample_vec, tn, m, M), approx_grad=True, bounds=bnds,
                                    m=10, factr=100, epsilon=1e-07, pgtol=1e-05, iprint=- 1,
                                    maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
        
        succ_flag = opt_res[2]["warnflag"] # 0 if convergence, 1 if too many func evals or iters, 2 other reason see ["task"]
        if succ_flag != 0:
            up = max(min(1, eta0+0.3), 0)
            lo = max(min(1, eta0-0.3), 0)
            eta0 = rng.uniform(lo, up, 1)[0] # if optimization not successful choose new random eta0 and try again

    res_param_vec = opt_res[0]
    logL = -opt_res[1]
    if succ_flag != 0:
        print(f"WARNING: Optimization not successful: {opt_res[2]['task']}")
        return {"succ_flag": False}

    # calculate some information criteria
    num_par = len(res_param_vec)
    aic = -2 * logL + 2 * num_par
    bic = -2 * logL + num_par * np.log(num_par)
    hq = -2 * logL + 2 * num_par * np.log(np.log(num_par))

    par_dic = {"mu": res_param_vec[0],
               "eta": res_param_vec[1]
               }
    if kernel == "expo":
        par_dic["theta"] = res_param_vec[2]
    elif kernel == "sum-expo":
        theta_sort = np.sort(res_param_vec[2::]) # sort thetas to avoid identification problems
        for k in range(0, P):
            par_dic[f"theta{k+1}"] = theta_sort[k]
    elif kernel in ["powlaw-cutoff", "powlaw"]:
        par_dic["alpha"] = res_param_vec[2]
        par_dic["tau"] = res_param_vec[3]

    # construct result dictionary
    res_dict = {"Kernel": kernel,
                "succ_flag": True,
                "OptInf": opt_res[2],
                "par": par_dic, # for expo: 1x1 vec of theta, sum-expo: Px1 vec of thetas, powl-cutoff & powl: 2x1 vec [alpha, tau]
                "logL": logL,
                "SampSize": len(sample_vec),
                "kwargs": kwargs,
                "IC": {"BIC": bic,
                       "AIC": aic,
                       "HQ": hq},
                "GridType": "no-grid" # can be: "large-grid", "small-grid", "no-grid" , the grid are from a sibling function below
                }
    return res_dict

def callback_powl(xk):
    print(xk)

def maximize_logL_grid(sample_vec, kernel, grid_type="large-grid", tn=None, rng=rng, **kwargs):
    """
    Estimation of a univariate Hawkes process via maximization of the log-likelihood on a grid of starting values
    The intensity of the process is parameterized as follows: 
        \lambda(t) = mu + eta * sum_{t_i < t} g(t - t_i)
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :param kernel: "expo", "sum-expo", "powlaw-cutoff", "powlaw"
    :param grid_type: "large-grid", "small-grid", "fast-grid"
    :param tn: the end time of the process
    :param rng: random number generator
    :return: a dictionary containing the results and additional values
    """
    # if no end/max time given use the last observation as tn
    if isinstance(tn, type(None)):
        tn = sample_vec[-1]
    
    if grid_type == "large-grid":
        eta_n = 100
        eta0_grid = np.linspace(0.001,0.999, eta_n)
    elif grid_type == "small-grid":
        pre_eta0 = kwargs["pre_eta0"]
        up = max(min(0.999, pre_eta0 + 0.15), 0.001)
        lo = max(min(0.999, pre_eta0 - 0.15), 0.001)
        eta0_grid = np.append(np.linspace(lo, up, 15), rng.uniform(0.01, 0.99, 10))
    elif grid_type == "fast-grid":
        eta0_grid = rng.uniform(0.01, 0.99, 5)

    phi_n = 5 # number of phi starting value random set for each magnitude
    opt_res_ls = [] # list to store the results of the eta_n optimizations
    logL_res_ls = []
    if kernel == "expo":
        the1 = np.append(np.linspace(0.001, 0.0099, phi_n), np.linspace(0.01, 0.099, phi_n))
        the2 = np.append(np.linspace(0.1, 0.99, phi_n), np.linspace(1, 10, phi_n))
        theta0_grid = np.append(the1, the2)
        bnds = [(1e-5, np.inf), (1e-5, 1.0), (1e-5, np.inf)] # parameter bounds
        for eta0 in eta0_grid:
            mu0 = len(sample_vec) * (1 - eta0) / tn 
            idx = rng.integers(0, phi_n*4, 1)[0] # choose a random theta0 from set of theta0_grid values
            param_vec0 = np.array((mu0, eta0, theta0_grid[idx]), dtype=np.float64)
            # minimize negative logL using scipy L-BFGS-B minimization routine
            opt_res = fmin_l_bfgs_b(func=uvhp_expo_logL, x0=param_vec0, fprime=uvhp_expo_logL_grad,
                                    args=(sample_vec, tn), approx_grad=False, bounds=bnds,
                                    m=10, factr=1000, pgtol=1e-05, iprint=- 1, maxfun=15000,
                                    maxiter=15000, disp=None, callback=None, maxls=20)
            if opt_res[2]["warnflag"] == 0: # if optimization exits successfull append result
                opt_res_ls.append(opt_res)
                logL_res_ls.append(-opt_res[1])
        

    elif kernel == "sum-expo":
        P = kwargs.get("P")
        the1 = np.append(np.linspace(0.001, 0.0099, phi_n), np.linspace(0.01, 0.099, phi_n))
        the2 = np.append(np.linspace(0.1, 0.99, phi_n), np.linspace(1, 10, phi_n))
        theta0_grid = np.append(the1, the2)

        bnds = [(1e-5, np.inf), (1e-5, 1)]
        for k in range(0, P):
            bnds.append((1e-5, np.inf))

        for eta0 in eta0_grid:
            mu0 = len(sample_vec) * (1 - eta0) / tn  
            idx = rng.integers(0, phi_n*4, P)
            param_vec0 = np.append(np.append(mu0, eta0), theta0_grid[idx])
            opt_res = fmin_l_bfgs_b(func=uvhp_sum_expo_logL, x0=param_vec0, fprime=uvhp_sum_expo_logL_grad,
                                    args=(sample_vec, tn, P), approx_grad=False, bounds=bnds,
                                    m=10, factr=1000, pgtol=1e-05, iprint=- 1, maxfun=15000,
                                    maxiter=15000, disp=None, callback=None, maxls=20)
            if opt_res[2]["warnflag"] == 0: # if optimization exits successfull append result
                opt_res_ls.append(opt_res)
                logL_res_ls.append(-opt_res[1])

    elif kernel in ["powlaw-cutoff", "powlaw"]:
        bnds = [(1e-8, np.inf), (1e-8, 1.0), (1e-8, 10), (1e-8, np.inf)]
        m = kwargs.get("m")
        M = kwargs.get("M") 
        alpha0_grid = np.append(np.linspace(0.01, 0.99, phi_n*3), np.linspace(1, 2, phi_n))
        tau0_grid = np.append(np.linspace(0.001, 0.099, phi_n*2), np.linspace(0.1, 2, phi_n*2))
        for eta0 in eta0_grid:
            mu0 = len(sample_vec) * (1 - eta0) / tn
            idx = rng.integers(0, 20, 2)
            param_vec0 = np.array([mu0, eta0, alpha0_grid[idx][0], tau0_grid[idx][0]], dtype=np.float64)
            #L-BFGS-B
            # I bounded alpha to 10 and tau to 1e-8 -> np.power(tau0, -alpha0) = 9.999999999999997e+79 < inf
                # the root cause of this has been identified as: overflow in np.power(tau0, -alpha0) -> inf for small tau and large alpha
            if kernel == "powlaw": # uvhp_approx_powl2_logL_test
                opt_res = fmin_l_bfgs_b(func=uvhp_approx_powl1_logL, x0=param_vec0, fprime=None,
                            args=(sample_vec, tn, m, M), approx_grad=True, bounds=bnds,
                            m=10, factr=10000, epsilon=1e-07, pgtol=1e-05, iprint=- 1,
                            maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
            elif kernel == "powlaw-cutoff":
                opt_res = fmin_l_bfgs_b(func=uvhp_approx_powl2_logL, x0=param_vec0, fprime=None,
                                    args=(sample_vec, tn, m, M), approx_grad=True, bounds=bnds,
                                    m=10, factr=1000, epsilon=1e-07, pgtol=1e-05, iprint=- 1,
                                    maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)

            if opt_res[2]["warnflag"] == 0: # if optimization exits successfull append result
                opt_res_ls.append(opt_res)
                logL_res_ls.append(-opt_res[1])
                

    if len(opt_res_ls) == 0:
        print("Grid maximization was not successful")
        return {"succ_flag": False}

    # choose model with largest logL
    idx = logL_res_ls.index(max(logL_res_ls))
    best_opt_res = opt_res_ls[idx]

    res_param_vec = best_opt_res[0]
    logL = -best_opt_res[1]
    # calculate some information criteria
    num_par = len(res_param_vec)
    aic = -2 * logL + 2 * num_par
    bic = -2 * logL + num_par * np.log(num_par)
    hq = -2 * logL + 2 * num_par * np.log(np.log(num_par))

    par_dic = {"mu": res_param_vec[0],
               "eta": res_param_vec[1]
               }
    if kernel == "expo":
        par_dic["theta"] = res_param_vec[2]
    elif kernel == "sum-expo":
        theta_sort = np.sort(res_param_vec[2::])
        for k in range(0, P):
            par_dic[f"theta{k+1}"] = theta_sort[k]
    elif kernel in ["powlaw-cutoff", "powlaw"]:
        par_dic["alpha"] = res_param_vec[2]
        par_dic["tau"] = res_param_vec[3]

    # construct result dictionary
    res_dict = {"Kernel": kernel,
                "succ_flag": True,
                "OptInf": best_opt_res[2],
                "par": par_dic, 
                "logL": logL,
                "SampSize": len(sample_vec),
                "kwargs": kwargs,
                "IC": {"BIC": bic,
                       "AIC": aic,
                       "HQ": hq},
                "GridType": grid_type # can be: "large-grid" for full eta0 grid, "small-grid" for prior eta0 +- 0.2 grid, "no-grid" other function eta0 based on mc estimation no grid search
                }
    return res_dict


@numba.jit(nopython=True)
def uvhp_expo_simulator(T, mu, eta, theta, seed=None):
    """
    Implements Ogata's modified thinning algorithm for the Hawkes process with single exponential kernel
    :param T: end time of the simulated process
    :param mu: background intensity
    :param eta: branching ratio
    :param theta: expo decay
    :param seed: seed for numba's random number generator
    :return: numpy 1d-array containing the simulated timestamps
    """
    if not seed == None:
        np.random.seed(seed) 

    # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return None # for class may put error code here (T to small for given mu)

    prec = np.float64(eta / theta)
    t_arr = np.zeros(1)
    t_arr[0] = np.float64(s)
    tn1 = np.float64(s) # time of the last accepted event
    rk = np.float64(0) # initilize recursion
    ul = np.float64(mu + prec) # initial intensity upper limit
    while s < T:
        # calculate prospective new event time s
        u = 1 - np.random.rand(2) # Note: we set 1 - u to exclude the very unlikely event u=0 -> log(0) = -inf
        s += -np.log(u[0]) / ul  # add exp(ul) distributed interevent time to running event time s

        # evaluate intensity at new event time s
        exn = np.exp(-(s-tn1)/theta)
        l = mu + prec * exn * (rk + 1)
        if u[1] <= (l / ul): # new event accepted
            t_arr = np.append(t_arr, s)

            # update values for next iteration
            rk = exn * (rk + 1)
            ul = mu + prec * (rk + 1)
            tn1 = s

        else: # new event rejected: modify upper limit 
            ul = l

    # check if last accepted event exceeds T
    if t_arr[-1] > T:
        return t_arr[0:-1]
    return t_arr


@numba.jit(nopython=True)
def uvhp_approx_powl_simulator(T, mu, eta, alpha, tau, m, M, seed=None):  
    """
    Implements Ogata's modified thinning algorithm for the Hawkes process with approximate power-law and (no cutoff component)
    :param T: end time of the simulated process
    :param mu: background intensity
    :param seed: seed for numba's random number generator
    :retam alpha, tau, m, M: kernel variables
    :parurn: numpy 1d-array containing the simulated timestamps
    """
    if not seed == None:
        np.random.seed(seed) 
        
     # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return None # for class may put error code here (T to small for given mu)

    z = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    etaoz = eta / z
    ak = tau * m**np.arange(0, M, 1)
    akp1 = ak**(-1-alpha)
    prec = etaoz * np.sum(akp1) # jump size

    t_arr = np.zeros(1)
    t_arr[0] = np.float64(s)
    tn1 = np.float64(s) # time of the last accepted event
    rk = np.zeros(M) # initilize recursion
    ul = np.float64(mu + prec) # initial intensity upper limit
    summ = np.float64(0)
    exps = np.float64(0)
    while s < T:
        # calculate prospective new event time s
        u = 1 - np.random.rand(2) # Note: we set 1 - u to exclude the very unlikely event u=0 -> log(0) = -inf
        s += -np.log(u[0]) / ul  # add exp(ul) distributed interevent time to running event time s

        # evaluate intensity at new event time s
        exps = np.exp(-(s-tn1)/ak)
        summ = 0.
        for k in range(0, M):
            summ += akp1[k] * exps[k] * (rk[k] + 1)

        l = mu + etaoz * summ
        if u[1] <= (l / ul): # new event accepted
            t_arr = np.append(t_arr, s)

            # update values for next iteration
            summ = 0.
            for k in range(0, M):
                rk[k] = exps[k] * (rk[k] + 1) 
                summ += akp1[k] * (rk[k] + 1)

            ul = mu + etaoz * summ
            tn1 = s

        else: # new event rejected: modify upper limit 
            ul = l

    # check if last accepted event exceeds T
    if t_arr[-1] > T:
        return t_arr[0:-1]
    return t_arr


@numba.jit(nopython=True)
def uvhp_approx_powl_cutoff_simulator(T, mu, eta, alpha, tau, m, M, max_n, seed=None):
    """
    Implements Ogata's modified thinning algorithm for the Hawkes process with approximate power-law and (nwith cutoff component)
    :param T: end time of the simulated process
    :param mu: background intensity
    :param seed: seed for numba's random number generator
    :retam alpha, tau, m, M: kernel variables
    :parurn: numpy 1d-array containing the simulated timestamps
    """

    if not seed == None:
        np.random.seed(seed) 
        
    # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return None 

    sc = tau**(-1 - alpha) * (1 - np.power(m, -(1 + alpha) * M)) / (1 - np.power(m, -(1 + alpha)))
    z1 = np.power(tau, -alpha) * (1 - np.power(m, -alpha * M)) / (1 - np.power(m, -alpha))
    z = z1 - sc * tau * np.power(m, -1)
    etaoz = eta / z
    an1 = tau * m**(-1)
    ak = tau * m**np.arange(0, M, 1)
    akp1 = ak**(-1-alpha)
    prec = etaoz * np.sum(akp1)  # not including -sc since upper limit based on modified intensity to insure beeing always greater equal the intensity to be simulated

    t_arr = np.zeros(1)
    t_arr[0] = np.float64(s)
    tn1 = np.float64(s) # time of the last accepted event
    rk = np.zeros(M) # initilize recursion
    rn1 = 0.
    ul = np.float64(mu + prec) # initial intensity upper limit
    summ = np.float64(0)
    exps = np.float64(0)
    while s < T:
        # calculate prospective new event time s
        u = 1 - np.random.rand(2) # Note: we set 1 - u to exclude the very unlikely event u=0 -> log(0) = -inf
        s += -np.log(u[0]) / ul  # add exp(ul) distributed interevent time to running event time s

        # evaluate intensity at new event time s
        exps = np.exp(-(s-tn1)/ak)
        expsn1 = np.exp(-(s-tn1)/an1)
        summ = 0.
        for k in range(0, M):
            summ += akp1[k] * exps[k] * (rk[k] + 1)
        
        l = mu + etaoz * (summ - sc * expsn1 * (rn1 + 1))
        if u[1] <= (l / ul): # new event accepted
            t_arr = np.append(t_arr, s)

            # update values for next iteration
            rn1 = expsn1 * (rn1 + 1)
            summ = 0.
            for k in range(0, M):
                rk[k] = exps[k] * (rk[k] + 1)
                summ += akp1[k] * (rk[k] + 1)

            ul = mu + etaoz * summ # again setting s=0 such that ul >> l for all t
            tn1 = s

        else: # new event rejected: modify upper limit using modified intensity
            ul = mu + etaoz * summ

        if len(t_arr) >= max_n:
            return t_arr

    # check if last accepted event exceeds T
    if t_arr[-1] > T:
        return t_arr[0:-1]
    return t_arr


@numba.jit(nopython=True)
def uvhp_sum_expo_simulator(T, mu, eta, theta_vec, M, seed=None):
    """
    Implements Ogata's modified thinning algorithm for the Hawkes process with sum of exponential kernel
    :param T: end time of the simulated process
    :param mu: background intensity
    :param eta: branching ratio
    :param theta_vec: expo decays
    :param seed: seed for numba's random number generator
    :return: numpy 1d-array containing the simulated timestamps
    """

    if not seed == None:
        np.random.seed(seed) 
        
     # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return None # for class may put error code here (T to small for given mu)

    etaom = eta / M
    prec = etaom * np.sum(1/theta_vec) # jump size

    t_arr = np.zeros(1)
    t_arr[0] = np.float64(s)
    tn1 = np.float64(s) # time of the last accepted event
    rk = np.zeros(M) # initilize recursion
    ul = np.float64(mu + prec) # initial intensity upper limit
    exps = np.float64(0)
    while s < T:
        # calculate prospective new event time s
        u = 1 - np.random.rand(2) # Note: we set 1 - u to exclude the very unlikely event u=0 -> log(0) = -inf
        s += -np.log(u[0]) / ul  # add exp(ul) distributed interevent time to running event time s

        # evaluate intensity at new event time s
        exps = np.exp(-(s-tn1)/theta_vec)
        summ = np.float64(0)
        for k in range(0, M):
            summ += 1 / theta_vec[k] * exps[k] * (rk[k] + 1)
        l = mu + etaom * summ
        if u[1] <= (l / ul): # new event accepted
            t_arr = np.append(t_arr, s)

            # update values for next iteration
            summ = 0.
            for k in range(0, M):
                rk[k] = exps[k] * (rk[k] + 1)
                summ += 1 / theta_vec[k] * (rk[k] + 1)
            ul = mu + etaom * summ
            tn1 = s

        else: # new event rejected: modify upper limit 
            ul = l

    # check if last accepted event exceeds T
    if t_arr[-1] > T:
        return t_arr[0:-1]
    return t_arr


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




def uvhp_residuals_dep(sample_vec, opt_res_dict):
    """
    Function that computes residual process and statistical test (goodness-of-fit) given result dictionary of maxlogL
    :param sample_vec: 1-d ndarray of timestamps. Must be positive and sorted in ascending order 
    :opt_res_dict: result dictionary returned by maxlogL or maxlogL_grid
    :return: an extended result dictionary
    """

    if not opt_res_dict["succ_flag"]:
        return opt_res_dict

    mu = opt_res_dict["par"]["mu"]
    eta = opt_res_dict["par"]["eta"]
    kernel = opt_res_dict["Kernel"]

    if kernel == "expo":
        theta = opt_res_dict["par"]["theta"]
        int_sample_vec = uvhp_expo_int_density(sample_vec, mu, eta, theta)

    elif kernel == "sum-expo":
        P = opt_res_dict["kwargs"]["P"]
        theta_vec = np.zeros(P)
        for k in range(0, P):
            theta_vec[k] = opt_res_dict["par"][f"theta{k+1}"]
        int_sample_vec = uvhp_sum_expo_int_density(sample_vec, mu, eta, theta_vec, P)    

    elif kernel == "powlaw":
        alpha = opt_res_dict["par"]["alpha"]
        tau = opt_res_dict["par"]["tau"]
        M = opt_res_dict["kwargs"]["M"]
        m = opt_res_dict["kwargs"]["m"]
        int_sample_vec = uvhp_approx_powl1_int_density(sample_vec, mu, eta, alpha, tau, m , M)

    elif kernel == "powlaw-cutoff":
        alpha = opt_res_dict["par"]["alpha"]
        tau = opt_res_dict["par"]["tau"]
        M = opt_res_dict["kwargs"]["M"]
        m = opt_res_dict["kwargs"]["m"]
        int_sample_vec = uvhp_approx_powl2_int_density(sample_vec, mu, eta, alpha, tau, m , M)
    
    int_samp_iet = np.diff(int_sample_vec) # inter arrival times of the integrated sample vector
    ed_test = np.sqrt(opt_res_dict["SampSize"]) * ((np.var(int_samp_iet, ddof=1) - 1) / np.sqrt(8))
    ed_pval = stats.norm.sf(abs(ed_test)) # test of excess dispersion (i.e. variance of residuals larger than 1)
    ks_test = stats.kstest(int_samp_iet, "expon", alternative="two-sided")
    lb_test = sm.stats.acorr_ljungbox(int_samp_iet, lags=[20], return_df=True) 

    # construct result dictionary
    opt_res_dict["IntDensity"] = int_sample_vec
    opt_res_dict["Timestamps"] = sample_vec,
    opt_res_dict["KStest"] = {"pval": ks_test[1],
                              "stat": ks_test[0]}
    opt_res_dict["LBtest"] = {"pval": float(lb_test["lb_pvalue"]),
                              "stat": float(lb_test["lb_stat"])}
    opt_res_dict["EDtest"] = {"pval": ed_pval,
                              "stat": ed_test}

    return opt_res_dict


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




