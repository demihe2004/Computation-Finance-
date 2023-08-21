# 1. 
Calculate price, POD and expected excersice time of default option 
``` python
def default_option_price (v0, l0, mu, sigma, gamma, lambda1, lambda2, T, r0, delta, alpha, epsilon, path, pmt):

    total_pmt = T * pmt
    dt = 1 / pmt
    dt_path = np.arange(dt, T + dt, dt)

    # Generate vt process - N paths
    vt = np.zeros((path, total_pmt+1))
    vt [:,0] = v0
    #reshape((path, total_pmt)) reshapes the 1D array obtained from np.random.normal() into a 2D array with dimensions (path, total_pmt). 
    #The reshaping is done by specifying the desired shape as a tuple (path, total_pmt)
    z = np.random.normal(0, 1, path * total_pmt).reshape((path, total_pmt))
    wt = z * dt ** 0.5
    jt = np.random.poisson(lambda1 * dt, path * total_pmt).reshape((path, total_pmt))
    for s in range(1, total_pmt +1):
        vt[:, s] = vt[:, s - 1] + vt[:, s - 1] * (mu * dt + sigma * wt[:, s - 1]) 
        vt[:, s] += vt[:, s] * gamma * jt[:, s - 1]

    # Generate Lt process - 1 path
    R = r0 + delta * lambda2
    r = R / pmt
    pmt_amount = (l0 * r) / (1 - 1 / np.power(1 + r, total_pmt))
    a = pmt_amount / r
    b = pmt_amount / (r * np.power(1 + r, total_pmt))
    c = 1 + r
    lt = a - b * np.power(c, dt_path * pmt)   

    # Stop time Q - 1*N array, one time for each path
    beta = (epsilon - alpha) / T
    qt = alpha + beta * dt_path
    cv_t = lt * qt
    Q_default_bool = vt[:, 1:] <= np.tile(cv_t, (path, 1))
    Q = np.argmax(Q_default_bool, axis=1)
    ## adding a value > T for paths with no default
    Q[Q == 0] = total_pmt + 1

    # Stop time Q - 1*N array, one time for each path
    N_default_bool = np.random.poisson(lambda2 * dt, path * total_pmt).reshape((path, total_pmt))
    S = np.argmax(N_default_bool, axis=1)
    ## adding a value > T for paths with no default
    S[S == 0] = total_pmt + 1

    # optimal excution time tau
    tau = np.minimum(Q, S)

    # payoff paths determining when Q default or s default happens first 
    payoff_Q = np.where(Q < S, 1, 0)
    Q_default_path = np.where((payoff_Q == 1) & (tau != total_pmt + 1))[0]
    S_default_path = np.where((payoff_Q == 0) & (tau != total_pmt + 1))[0]

    # get payoffs and take mean as the default option price
    payoff = np.zeros(path)
    for n in Q_default_path:
        payoff[n] = np.maximum(lt[tau[n]] - epsilon * vt[n, tau[n]+1],0)

    for n in S_default_path:
        payoff[n] = np.abs(lt[tau[n]] - epsilon * vt[n, tau[n]+1]) 

    D_price = payoff.mean()
    
    # get POD and expected exercise time Et
    pod = (len(S_default_path) + len(Q_default_path)) / path
    Et =  (np.mean(tau[tau != total_pmt + 1])) * dt

    return D_price, pod, Et
```
Change of price given parameter:

![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/64ce600f-c927-4708-be57-eddedbb2c121)

Change of POD given parameter:

![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/0c5062e4-0aa1-48b2-81bd-853a105981a5)

