import numpy as np
import numba

@numba.jit(nopython=True, fastmath=True)
def uvhp_approx_powl_cutoff_simulator(T, mu, eta, alpha, tau, m, M, seed=None):
    # T, mu, theta must be positive, eta must be in (0, 1)

    # returns timestamps of the generated process

    # implements Ogata's modified thinning algorithm as described in algorithm 2 in Ogata 1981 

    if not seed == None:
        np.random.seed(seed) 
        
    # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return None # for class may put error code here (T to small for given mu)

    # calculate fixed values
    sc = tau**(-1 - alpha) * (1 - m**(-(1 + alpha) * M)) / (1 - m**(-(1 + alpha)))
    z1 = tau**(-alpha) * (1 - m**(-alpha * M)) / (1 - m**(-alpha))
    z = z1 - sc * tau * m**(-1)
    
    etaoz = eta / z
    an1 = tau * m**(-1)
    ak = tau * m**np.arange(0, M, 1)
    akp1 = ak**(-1-alpha)
    prec = etaoz * np.sum(akp1)  # not including -sc since upper limit based on modified intensity to insure beeing always greater equal the intensity to be simulated
    
    avg_n = mu / (1-eta) * T
    # allocate 5 times the number of expected observations (to be on the safe side for the mc)
    t_arr = np.empty(np.int64(avg_n*5), np.float64) 
    i = 0
    #t_arr = np.zeros(1)
    tn1 = np.float64(s) # time of the last accepted 
    t_arr[i] = tn1
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
            #t_arr = np.append(t_arr, s)
            i += 1
            t_arr[i] = s

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
    
    
    # check if last accepted event exceeds T
    if t_arr[i] > T:
        return t_arr[0:i]
    return t_arr[0:(i+1)]



@numba.jit(nopython=True, fastmath=True)
def uvhp_expo_simulator(T, mu, eta, theta, seed=None):
    # T, mu, theta must be positive, eta must be in (0, 1)

    # returns timestamps of the generated process

    # implements Ogata's modified thinning algorithm as described in algorithm 2 in Ogata 1981 
    if not seed == None:
        np.random.seed(seed) 

    # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return None # for class may put error code here (T to small for given mu)

    prec = np.float64(eta / theta)
    
    avg_n = mu / (1-eta) * T
    # allocate 5 times the number of expected observations (to be on the safe side for the mc)
    t_arr = np.empty(np.int64(avg_n*5), np.float64) 
    i = 0
    #t_arr = np.zeros(1)
    tn1 = np.float64(s) # time of the last accepted event
    t_arr[i] = tn1
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
            #t_arr = np.append(t_arr, s)
            i += 1
            t_arr[i] = s

            # update values for next iteration
            rk = exn * (rk + 1)
            ul = mu + prec * (rk + 1)
            tn1 = s

        else: # new event rejected: modify upper limit 
            ul = l

    # check if last accepted event exceeds T
    if t_arr[i] > T:
        return t_arr[0:i]
    return t_arr[0:(i+1)]


@numba.jit(nopython=True, fastmath=True)
def uvhp_approx_powl_simulator(T, mu, eta, alpha, tau, m, M, seed=None):
    # T, mu, theta must be positive, eta must be in (0, 1)

    # returns timestamps of the generated process

    # implements Ogata's modified thinning algorithm as described in algorithm 2 in Ogata 1981 

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

    avg_n = mu / (1-eta) * T
    # allocate 5 times the number of expected observations (to be on the safe side for the mc)
    t_arr = np.empty(np.int64(avg_n*5), np.float64) 
    i = 0
    #t_arr = np.zeros(1)
    tn1 = np.float64(s) # time of the last accepted event
    t_arr[i] = tn1
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
            #t_arr = np.append(t_arr, s)
            i += 1
            t_arr[i] = s

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
    if t_arr[i] > T:
        return t_arr[0:i]
    return t_arr[0:(i+1)]


@numba.jit(nopython=True, fastmath=True)
def uvhp_sum_expo_simulator(T, mu, eta, theta_vec, M, seed=None):
    # T, mu, theta must be positive, sum alpha must be smaller 1

    # returns timestamps of the generated process

    # implements Ogata's modified thinning algorithm as described in algorithm 2 in Ogata 1981 

    if not seed == None:
        np.random.seed(seed) 
        
     # generate the first event
    u1 = 1 - np.random.rand(1)[0]
    s = -np.log(u1) / mu
    if s > T:
        return None # for class may put error code here (T to small for given mu)

    etaom = eta / M
    prec = etaom * np.sum(1/theta_vec) # jump size

    avg_n = mu / (1-eta) * T
    # allocate 5 times the number of expected observations (to be on the safe side for the mc)
    t_arr = np.empty(np.int64(avg_n*5), np.float64) 
    i = 0
    #t_arr = np.zeros(1)
    
    tn1 = np.float64(s) # time of the last accepted event
    t_arr[i] = tn1
    
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
            #t_arr = np.append(t_arr, s)
            i += 1
            t_arr[i] = s

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
    if t_arr[i] > T:
        return t_arr[0:i]
    return t_arr[0:(i+1)]

