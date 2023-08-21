# 1. 
Convergence using different binomial trees
- a: $u = \frac{1}{d}, d =c - \sqrt{c^2 - 1}, c = \frac{1}{2}(e^{-r\Delta} + e^{(r+ \sigma^2)\Delta}), p = \frac{e^{r\Delta } - d}{u - d}$
- b: $u = e^{r \Delta} (1 + \sqrt{e^{\sigma^2\Delta} - 1}), d = e^{r \Delta} (1 - \sqrt{e^{\sigma^2\Delta} - 1}), p = \frac{1}{2}$
- c: $u = e^{(r - \frac{\sigma^2}{2})\Delta + \sigma\sqrt{\Delta}}, d = e^{(r - \frac{\sigma^2}{2})\Delta - \sigma\sqrt{\Delta}}, p = \frac{1}{2}$
- d: $u = e^{ \sigma\sqrt{\Delta}}, d =  e^{ -\sigma\sqrt{\Delta}}, p = \frac{1}{2} + \frac{1}{2}\bigg(\frac{(r - \frac{1}{2}\sigma^2)\sqrt{\Delta} }{\sigma}\bigg)$
  
  ![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/56f93ca9-6ffe-4e4d-80f6-f141f1cb394e)

# 2. 
American call with binomial tree:
``` python
def binominal_underlying_tree_american (s0, u, d, n):
    """
    Takes in specific u, d, n and returns the full underlying binomial tree 
    """
    stock_tree = np.zeros((n+1, n+1))
    stock_tree[0,0] = s0

    for column in range(1, n + 1):
        stock_tree[0, column] = stock_tree[0, column - 1] * u
        for row in range(1, column + 1):
            stock_tree[row, column] = stock_tree[row - 1, column - 1] * d

    return stock_tree

def payoff_tree_american_call (s0, k, u, d, n, p, discount):
    """
    Takes in the full underlying binomial tree and compared max[EV,CV] at each node, 
    returns option value at t = 0
    """

    payoff_tree_EV = binominal_underlying_tree_american (s0, u, d, n)
    payoff_tree_EV = np.maximum(payoff_tree_EV - k, 0)

    payoff_tree = np.zeros((n+1, n+1))
    payoff_tree[:,-1] = payoff_tree_EV[:,-1]

    for column in range(n-1, -1, -1):
        for row in range(column, -1, -1):
            payoff_tree [row, column] = np.maximum (payoff_tree_EV[row, column], 
                                                    (payoff_tree[row, column + 1] * p + payoff_tree[row + 1, column + 1] * (1 - p)) * discount)

    return payoff_tree[0,0]
```
Greeks
``` python
def CRR_greeks(s0, T, k, r, sigma, epsilon, u, d, n, p, discount):
    
    #delta
    s0_plus = s0 + epsilon
    s0_minus = s0 - epsilon
    delta = (payoff_tree_american_call (s0_plus, k, u, d, n, p, discount) - payoff_tree_american_call (s0_minus, k, u, d, n, p, discount))/(2*epsilon)

    #gamma
    gamma = ((payoff_tree_american_call (s0_plus, k, u, d, n, p, discount) - 2 * payoff_tree_american_call (s0, k, u, d, n, p, discount)) + (payoff_tree_american_call (s0_minus, k, u, d, n, p, discount)))/(epsilon**2)
    
    #theta
    T_plus = T + epsilon
    T_minus = T - epsilon
    u_plus = np.exp(sigma * np.sqrt(T_plus/n))
    u_minus = np.exp(sigma * np.sqrt(T_minus/n))
    d_plus = np.exp(-sigma * np.sqrt(T_plus/n))
    d_minus = np.exp(-sigma * np.sqrt(T_minus/n))
    p_plus = 1/2 + 1/2 * ((r - sigma ** 2/2) * np.sqrt(T_plus/n)/sigma)
    p_minus = 1/2 + 1/2 * ((r - sigma ** 2/2) * np.sqrt(T_minus/n)/sigma)
    discount_plus = np.exp(-r * T_plus/n)
    discount_minus = np.exp(-r * T_minus/n)
    
    theta = -(payoff_tree_american_call (s0, k, u_plus, d_plus, n, p_plus, discount_plus) - payoff_tree_american_call (s0, k, u_minus, d_minus, n, p_minus, discount_minus))/(2*epsilon)
    
    #vega
    sigma_plus = sigma + epsilon
    sigma_minus = sigma - epsilon
    u_plus = np.exp(sigma_plus * np.sqrt(dt))
    u_minus = np.exp(sigma_minus * np.sqrt(dt))
    d_plus = np.exp(-sigma_plus * np.sqrt(dt))
    d_minus = np.exp(-sigma_minus * np.sqrt(dt))
    p_plus = 1/2 + 1/2 * ((r - sigma_plus ** 2/2) * np.sqrt(dt)/sigma_plus)
    p_minus = 1/2 + 1/2 * ((r - sigma_minus ** 2/2) * np.sqrt(dt)/sigma_minus)
                    
    vega = (payoff_tree_american_call (s0, k, u_plus, d_plus, n, p_plus, discount)- payoff_tree_american_call (s0, k, u_minus, d_minus, n, p_minus, discount))/(2*epsilon)
    
    return delta,gamma,theta,vega
```
![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/ee0ad16b-97cb-435d-8f33-9dad7b3f606b)

# 3. 
Trinomial tree for pricing Euro call
Use the trinomial method applied to the stock price-process ($S_t$) in which : $u = \frac{1}{d}, d = e^{-\sigma\sqrt{3\Delta}}$

$p_d = \frac{r\Delta(1 - u) + (r\Delta)^2 + \sigma^2\Delta}{(u-d)(1-d)}$ ,
$p_u = \frac{r\Delta(1 - d) + (r\Delta)^2 + \sigma^2\Delta}{(u-d)(u-1)}$, $p_m = 1 - p_u - p_d$

```python
def trinomial_underlying_tree (s0, u, d, n):
    """
    Takes in specific s0, u, d, n and returns the terminal underlying value 
    """
    stock_tree = np.zeros((2*n+1, n+1))
    stock_tree[0,0] = s0
    stock_tree[:3,1] = s0 * u, s0, s0 * d

    for column in range(2, n + 1):
        last = 2 * column 
        # get the reduntant part of the trinominal tree
        stock_tree[1:,column] = stock_tree[:-1, column - 1] 
        # calculate the first and last node
        stock_tree[0,column] = stock_tree[1,column] * u
        stock_tree[last,column] = stock_tree[last-1,column] * d
   
    return stock_tree[:,-1]

def payoff_tree_tri_euro_call (s0, u, d, n, pu, pd, k, discount):
    """
    Takes in st calculated from trinomial_underlying_tree and create payoff tree by discounting from T
    Returns the [0,0] value of tree, which is payoff at t=0
    """

    #payoff_tree = np.zeros((n+1, n+1))
    payoff_tree = trinomial_underlying_tree (s0, u, d, n)
    payoff_tree = np.maximum(payoff_tree - k, 0)

    for i in range(n):
        payoff_tree = discount * (payoff_tree[:-2] * pu + payoff_tree[1:-1] * (1 - pu - pd) + payoff_tree[2:] * pd )

    return payoff_tree[0]
```
# 4.
Halton’s Low-Discrepancy Sequences to price European Call options
``` python
def GetHalton(HowMany, Base):
    """
    Funtion form project 3, adapted from matlab code provided in class
    """
    Seq = np.zeros(HowMany)
    NumBits = 1 + np.ceil(np.log(HowMany)/np.log(Base))
    VetBase = Base ** (-np.arange(1, NumBits + 1, 1))
    WorkVet = np.zeros(int(NumBits))
    
    for i in range(HowMany):
        j = 0
        ok = 0
        
        while ok == 0:
            WorkVet[j] += 1
            if WorkVet[j] < Base:
                ok = 1
            else:
                WorkVet[j] = 0
                j += 1
                
        Seq[i] = np.dot(WorkVet, VetBase)
        
    return Seq

def Box_Muller(HowMany, base1, base2):
    """
    Function from project 1, 
    takes in uniform generated from GetHalton and generate normal dist
    """
    H1 = GetHalton(HowMany, base1)
    H2 = GetHalton(HowMany, base2)
    
    BM_z1= np.sqrt(-2*np.log(H1)) * np.cos(2*np.pi*H2)
    BM_z1 = np.sqrt(-2*np.log(H1)) * np.sin(2*np.pi*H2)
    return BM_z1, BM_z1


def Halton_BM_euro_call(s0, k, r, T, HowMany,sigma, base1, base2):
    """
    Takes in normal dist generated by Halton and BM
    takes average of all simulations and generate avergae call price
    """
    z_i = np.hstack(Box_Muller(HowMany, base1, base2))
    st = s0 * np.exp((r - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * z_i)
    payoff = np.maximum(st - k, np.zeros(HowMany*2))
    call_price = np.exp(-r * T) * np.mean(payoff)
    
    return call_price
```

