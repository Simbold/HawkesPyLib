# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from num.uvhp import *
from num.sim import *
from matplotlib import pyplot as plt
rng = np.random.default_rng()

def simulate_hp(T, mu, eta, kernel, seed=None, **kwargs):
    # check arguments
    
    # TODO: Check arguments block, burn period, max n simulation
    
    if kernel == "expo":
        theta = kwargs.get("theta")
        sim_proc = uvhp_expo_simulator(T, mu, eta, theta, seed=seed)
    elif kernel == "sum-expo":
        theta_vec = kwargs.get("theta_vec")
        sim_proc = uvhp_sum_expo_simulator(T, mu, eta, theta_vec, M, seed=seed)
    elif kernel == "powlaw":
        alpha = kwargs.get("alpha")
        tau0 = kwargs.get("tau0")
        m = kwargs.get("m")
        M = kwargs.get("M")
        sim_proc = uvhp_approx_powl_simulator(T, mu, eta, alpha, tau0, m, M, seed=None)
    elif kernel == "powlaw-cutoff":
        alpha = kwargs.get("alpha")
        tau0 = kwargs.get("tau0")
        m = kwargs.get("m")
        M = kwargs.get("M")
        sim_proc = uvhp_approx_powl_cutoff_simulator(T, mu, eta, alpha, tau0, m, M, seed=None)
        
    return sim_proc # or return dictionary including simulation paramters
        
        
def ml_estimation(sample_vec, kernel, tn=None, rng=rng, **kwargs):
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
            eta0 = rng.uniform(0, 1, 1)[0] # if optimization not successful choose new random eta0 and try again

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
        par_dic["tau0"] = res_param_vec[3]

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

def ml_estimation_grid(sample_vec, kernel, grid_size=25, grid_type="random", tn=None, rng=rng, **kwargs):
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
    
    if grid_type == "random":
        eta0_grid = rng.uniform(0.001, 0.999, grid_size)
    elif grid_type == "equidistant":
        eta0_grid = np.linspace(0.001, 0.999, grid_size)
    elif grid_type == "custom":
        eta0_grid = kwargs.get("custom-grid")

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
        par_dic["tau0"] = res_param_vec[3]

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
                "GridType": grid_type, 
                "GridSize": grid_size
               }
    return res_dict

              
def exp1_inv(x):
    # quantile function for exponential distribution with lambda=1
    return -np.log(1-x)


def qq_plot(x, F_inv):
    n = len(x)
    x_sorted = np.flip(np.sort(x))
    F_th = F_inv((n-np.arange(1, n+1)+1)/(n+1))  # np.linspace(n/(n+1), 1/(n+1), num= n)
    plt.plot(F_th, x_sorted, 'ob')
    plt.ylabel("empirical quantile")
    plt.xlabel("theoretical quantile")

    # plot two lines: 1: linear regression line through points, 2: unit diagonal line
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x_sorted, F_th)
    #plt.plot(intercept+slope*x_sorted, x_sorted, '-r')
    plt.plot(x_sorted, x_sorted)
    plt.show()
    return F_th
              

def get_residual_process(sample_vec, opt_res_dict, QQplot=False):
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
        tau = opt_res_dict["par"]["tau0"]
        M = opt_res_dict["kwargs"]["M"]
        m = opt_res_dict["kwargs"]["m"]
        int_sample_vec = uvhp_approx_powl1_int_density(sample_vec, mu, eta, alpha, tau, m , M)

    elif kernel == "powlaw-cutoff":
        alpha = opt_res_dict["par"]["alpha"]
        tau = opt_res_dict["par"]["tau0"]
        M = opt_res_dict["kwargs"]["M"]
        m = opt_res_dict["kwargs"]["m"]
        int_sample_vec = uvhp_approx_powl2_int_density(sample_vec, mu, eta, alpha, tau, m , M)
    
    int_samp_iet = np.diff(int_sample_vec) # inter arrival times of the integrated sample vector
    #ed_test = np.sqrt(opt_res_dict["SampSize"]) * ((np.var(int_samp_iet, ddof=1) - 1) / np.sqrt(8))
    #ed_pval = stats.norm.sf(abs(ed_test)) # test of excess dispersion (i.e. variance of residuals larger than 1)
    #ks_test = stats.kstest(int_samp_iet, "expon", alternative="two-sided")
    #lb_test = sm.stats.acorr_ljungbox(int_samp_iet, lags=[20], return_df=True) 
    
    # plots for visual analysis
    if QQplot:
        qq_plot(int_samp_iet, exp1_inv)
              
    # construct result dictionary
    opt_res_dict["IntDensity"] = int_sample_vec
    opt_res_dict["Timestamps"] = sample_vec,
    #opt_res_dict["KStest"] = {"pval": ks_test[1],
    #                         "stat": ks_test[0]}
    #opt_res_dict["LBtest"] = {"pval": float(lb_test["lb_pvalue"]),
    #                          "stat": float(lb_test["lb_stat"])}
    #opt_res_dict["EDtest"] = {"pval": ed_pval,
    #                          "stat": ed_test}

    return opt_res_dict
