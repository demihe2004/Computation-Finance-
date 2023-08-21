# 1. 
Vasicek model: $dr_t = \mathcal{k}(\overline r - r_t)dt + \sigma dW_t$

- r paths 
```python
def Vasicek(day, T, t, r0, sigma, k, r_bar, N):
    
    '''
    Vasicek model generator:
    Takes in intial assumptions and set T and steps for simulation of r
    return N*step matirx of r paths 
    '''
    #taking each time step to be a day
    steps = int((T - t) * day)
    dt = (T - t) / steps

    # create N paths with T divided into steps
    r = np.zeros((N, steps + 1))
    r[:,0] = r0
    
    for i in range(1, steps+1):
        wt = np.random.normal(size = N) * np.sqrt(dt)
        r[:, i] = r[:, i-1] + k * (r_bar - r[:, i-1]) * dt + sigma * wt
                
    return r
```
- simulate PV
```python
def bond_price_vasicek (payment, coupon_freq, day, T, t, r0, sigma, k, r_bar, N, bond_type):
    
    """
    ZCB and CB price generator with int rate predicted using vasiceck model:
    Takes in initial assumptions and bond type of coupon paying or not
    return 1*1 vector of bond price as sum of PVs for each payment 
    """
    
    #set T according to payment interval 
    if bond_type == "ZCB":
        T_list = [T]
    if bond_type == "CB":
        T_list = np.arange (coupon_freq, T/coupon_freq + coupon_freq, coupon_freq)
    
    steps = int((T - t) * day)
    dt = (T - t) / steps
    
    # get PV for each payment discounted to t = 0
    discount = np.zeros(len(payment))
    
    for i in range(len(payment)):
        
        r = Vasicek(day, T_list[i], t, r0, sigma, k, r_bar, N)
        discount[i] = np.mean(np.exp(- np.sum(r * dt, axis = 1)))
        #PV[i] = payment[i] * np.mean(np.exp(-np.sum(r * dt, axis=1)))
   
    PV = np.dot(payment, discount)
    #return sum(PV) 
    return PV
```
- Close form for ZCB:
The explicit price function for ZCB (maturing @ S) fitted with vacicek model per lecture notes is:

$$
P(t, S, r_r) = par \cdot A(t, S) \cdot e^{-B(t,S) * r_t}
$$
Given
$$
\begin{cases}
A(t, S) & = e^{(\bar{r} - \frac{\sigma^2}{2k^2})[B(t, S) - (S-t)] - \frac{\sigma^2}{4k}B^2(t,S)}\\
B(t, S) & = \frac{1}{k}(1 - e^{-k(S-t)})
\end{cases}
$$

Options (maturing @ T) on Pure Discount Bonds(maturing @ S): c(0,T,S) = $E(e^{-\int_0^T{r_s ds}}\cdot (P(T, S, r_r)-K)^+)$

``` python
def ZCB_Vasicek_explicit (r_bar, r0, k, sigma, t, S, par):
    
    '''
    explicit solution for ZCB:
    Takes in initial assumptions and par value of bond
    returns: 1*1 vector of price of such ZCB maturing @ S and discounted to t = t
    '''
    
    # define A, B
    B = 1 / k * (1 - np.exp(-k * (S - t)))
    A = np.exp((r_bar - sigma ** 2 / (2 * k ** 2)) * (B - (S - t)) - sigma ** 2 / (4 * k) * B ** 2)
    
    # compute for price
    price = par * A * np.exp(-B * r0)
    
    return price
```

- Close form for option price:
For call option on coupon paying bond, define price as: $C = (t,T,{c_i}_{i=1}^n,{T_i}_{i=1}^n,K)$ where

$$
\begin{cases}
K = \text{strike price} \\
T = \text{option maturity time} \\
c_i = \text{coupon payment} \\
T_i = \text{coupon paying time} \\
\end{cases}
$$

Based on Jamshidian(1989), further deduct $C = \sum_{i=1}^n {c_i\cdot C(t,T,T_i,K_i)}$ where

$$
\begin{cases}
n : \text{number of coupon due after T} \\
C(t,T,T_i,K_i): \text{price of euro call @ T on ZCB maturing @ Ti and stike price Ki} \\
K_i = P(T,T_i,r^*) \text{which is the pure dicount bond price discounted by r\*} \\
r\* : \text{Optimzied short rate that} \sum^n_{i=1} {c_i \cdot P(T,T_i,r^\*) = K}
\end{cases}
$$

Further decution, we get close form solution:

$$
\begin{align*}
C = \sum_{i=1}^n {c_i \cdot C(t, T, T_i, P(T,T_i,r^\*))}\\
= \sum_{i=1}^n {c_i \cdot P(t,T_i)\cdot N(d_{i,+}) - K_i P(t,T)N(d_{i,-})}
\end{align*}
$$

where

$$
\begin{cases}
d_{i,+} & = \frac{1}{\sigma_p}\cdot ln(\frac{P(t, T_i, r_t)}{K_i\cdot P(t, T, r_t)}) + \frac{\sigma_p}{2}\\
d_{i,-} & = d_{i,+} - \sigma_p\\
\sigma_p & = \frac{\sigma}{k}(1 - e^{-k(T_i-T)}) \sqrt{\frac{1}{2k}(1- e^{-2k(T-t)})}
\end{cases}
$$

```python
def EU_call_CB_Vasicek_explicit(r0, trial_step, max_trial, payment, coupon_freq, day, T, t, sigma, k, K, r_bar,par = 1):
    
    """
    Function that returns the close form solution of an euro call option on coupon bond:
    Takes t as the option mature time and T as bond mature time
    Also taker r_star and close form ZCB results
    Return 1*1 variable of euro call option on specific coupon paying bond
    """
    
    T_list = np.arange (coupon_freq, T/coupon_freq + coupon_freq, coupon_freq)

    steps = int((T - t) * day)
    dt = (T - t) / steps

    r_star = short_rate (r0, trial_step, max_trial, payment, coupon_freq, day, T, t, sigma, k, K)

    discount = np.zeros(len(payment))

    for i in range(len(payment)):

        # ki: ith ZCB both from mature of option to mature of bond, discounted at r_star
        Ki = ZCB_Vasicek_explicit(r_bar, r_star, k, sigma, t, T_list[i], par)
        # pt: ZCB price from 0 to mature of option, discounted at r0
        p_T = ZCB_Vasicek_explicit (r_bar, r0, k, sigma, 0, t, par)
        # ps: ZCB price from 0 to mature of bond, discounted at r0
        p_S = ZCB_Vasicek_explicit (r_bar, r0, k, sigma, 0, T_list[i], par)
        # BS parameters
        sigmap = np.sqrt((1 - np.exp(-2 * k * (t - 0))) / (2 * k)) * (1 - np.exp(-k * (T_list[i] - t))) / k * sigma
        d1 = np.log(par * p_S / (Ki * p_T)) / sigmap + sigmap / 2
        d2 = d1 - sigmap
        # discount factor array
        discount_i = par * p_S * norm.cdf(d1, 0, 1) - Ki * p_T * norm.cdf(d2, 0, 1)
        discount[i] = discount_i 

    PV = np.dot(payment, discount)
    return PV
```

# 2.
CIR model: $dr_t = \mathcal{k}(\overline r - r_t)dt + \sigma\sqrt{r_t} dW_t$

# 3. 
G2++ model:

$$
\begin{cases}
dx_t = −ax_tdt + \sigma dWt^1\\
dy_t = −by_tdt + \eta dWt^2\\
r_t = x_t + y_t + \phi_t
\end{cases}
$$
