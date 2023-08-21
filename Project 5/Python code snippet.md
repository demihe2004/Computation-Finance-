# 1. 
LSMC method to price american put
``` python
def LSMC_American_put (K,s0,T,r,sigma,n,k,method):
    """
    Takes in all inputs for a put option and calculate each path and optional excute time by
    comparing EV and CV at each dt and assigning and updating indexes with value of 1 and 0 
    output the mean of all path discounted to t=0 and the standard deviation 
    """
    
    # set up step, dt and discount for future use
    step = int(n**0.5)
    dt = T / step
    discount = np.exp(-r * dt)
    ECV = np.zeros((n, step+1))
    
    # get underlying path 
    st = St (s0,T,r,sigma,n)
    
    # set up EV payoff tree
    payoff_tree_EV = st
    payoff_tree_EV = np.maximum(k - st, 0)
    payoff_tree_EV
    
    # set up index tree and populate t=T 
    index = np.zeros((n, step+1))
    index[:,-1] = st[:,-1]
    index[:,-1] = np.where(index[:,-1] < k, 1, 0)
    
    # construct payoff tree and index 
    for t in range(step-1):
        # set up discount factor for Y
        discount_vector = np.power(discount, np.arange(1,t+2))
        
        # get Y
        cv = payoff_tree_EV[:,step-t:] * index[:,step-t:] * discount_vector
        y_t = np.sum(cv, axis=1, keepdims=True)
        
        # get x
        x_t = st[:,step-t-1:step-t]
        x_t[x_t > k] = 0
        y_t[x_t == 0] = 0
        
        # keep only non-zero x and Y at t = t
        x_t_nonzero = x_t[np.nonzero(x_t)]
        mask = x_t[:, 0] != 0
        y_matrix = y_t[mask].reshape(-1, 1)
        
        # get F b A a with matrix calculation 
        if method == 'laguerre':
            F_t = laguerre(x_t_nonzero.T, K)
            #F_t = laguerre(x_t.T, K)

        if method == 'hermite':
            F_t = hermite(x_t_nonzero.T, K)

        if method == 'simple':
            F_t = simple(x_t_nonzero.T, K)

        #F_t = laguerre(x_t_nonzero.T, K)
        b_t = np.dot(F_t, y_matrix)
        A_t = np.dot(F_t, F_t.T)
        a_t = np.dot(np.linalg.inv(A_t), b_t)
        
        # get ECV 
        ECV_t = np.sum(F_t * a_t, axis=0).reshape(-1, 1)
        ECV_t = ECV_t.T
        ECV_map = np.zeros(mask.shape)
        ECV_map[np.where(mask)[0]] = ECV_t.flatten()
        ECV_map = ECV_map.reshape(-1,1)
        ECV [:, step-t-1] = ECV_map.flatten()
        
        # update index tree
        mask_1 = payoff_tree_EV[:, step-t-1:step-t] > ECV_map
        index[:, step-t-1:step-t] = np.where(mask_1, 1, 0)
        
        # make sure every path is excuted only once 
        mask_2 = index[:, step-t-1:] == 1
        row_sum = np.sum(mask_2, axis=1)
        mask_2[row_sum > 1, 1:] = False
        index[:, step-t-1:] = np.where(mask_2, 1, 0)
        payoff_tree_EV[:, step-t-1:step-t] = np.maximum(payoff_tree_EV[:, step-t-1:step-t], ECV_map)
    
    # generate PV by multiplying payoff at each nod with whether or not to excercise
    PV = np.multiply(payoff_tree_EV,index)
    indices = np.nonzero(PV)
    PV_array = PV[indices]
    PV_put = PV_array.sum()/n*np.exp(-r) 
    PV_put_std = PV_array.std()/np.sqrt(n)
    
    return PV_put, PV_put_std
```
- Laguerre

![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/c702169f-64aa-49b5-bdcc-9be7a1659c45)

- Hermite
  
![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/86a65d0d-066d-4f69-bca7-979632f86134)

- Simple
  
![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/86b1f8f2-6de3-4afa-af29-711a02be3c94)

Observations:
1. Compared with the price calculated using binomial method, looks like Hermite and Monomials method did a great job, but Laguerre Polynomials result is quite off. It seems like exponential is not working too well in this situation. 
2. Looking into price and std of using different k, I found out that the estimation did the best when k = 3. One explanation could be that since we used higher order polynomials, the estimation is more accurate. There also exist tradeoff of using larger k means larger calculation time.

  
