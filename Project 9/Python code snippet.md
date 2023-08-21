# MBS

$$P_0 = E^\*\bigg(\sum_{t = 1}^N PV(CF_t) \bigg) = E^\*\bigg(\sum_{t = 1}^N d_t \cdot c_t\bigg)$$

1. Discount factor dt: $dt =\text{exp}^{\int^t_0{(r_u)du}} $
2. Cash flow factor ct

$$
\begin{aligned}
c_t &= TPP_t + IP_t = SP_t + PP_t + IP_t = MP_t + PP_t\\
& = \frac{PV_{t-1}r}{1 - (1 + r)^{-N+(t-1)}} + (PV_{t-1} - PV_{t-1}r[\frac{1}{1 - (1+r)^{-N + (t-1)}} - 1])(1 - (1 - CPR_t)^{\frac{1}{12}})
\end{aligned}
$$

3. CPR

$$
CPR_t = RI_t \cdot BU_t \cdot SG_t \cdot SY_t
$$

where

$$
\begin{aligned}
RI_t & = 0.28 + 0.14 \arctan(-8.57 + 430 (R - r_{t-1}(10)))\\
BU_t & = 0.3 + 0.7\frac{PV_{t-1}}{PV_0}\\
SG_t & = \min(1, \frac{t}{30})\\
SY_t & = \{0.94, 0.76, 0.74, 0.95, 0.98, 0.92, 0.98, 1.10, 1.18, 1.22, 1.23, 0.98\}
\end{aligned}
$$

and the 10yr US treasury rate can be simulated by 

$$
r_{t-1}(10)^i = -\frac{1}{T}\log(P_{t-1}(10))^i
$$

with $P_{t-1}(10)$ simulated by close form solution of ZCB bonds using CIR model for r

```python
def CIR (T, t, r0, sigma, k, r_bar, n, month = 12, interval = 10):
    
    '''
    CIR model generator:
    Takes in intial assumptions and set T and steps for simulation of r
    return n*step matirx of r paths 
    '''
    np.random.seed(4059)
    
    steps = int((T - t) * month * 10)
    dt = (T - t) / steps
    #dt = 1 / np.sqrt(n)
    #steps = int((T - t) / dt)

    # create N paths with T divided into steps
    r = np.zeros((n, steps + 1))
    r[:,0] = r0
    
    for i in range(1, steps+1):
        wt = np.random.normal(size = n) * np.sqrt(dt)
        r[:, i] = np.maximum(r[:, i-1], 0) + k * (r_bar - np.maximum(r[:, i-1], 0)) * dt + sigma * np.sqrt(np.maximum(r[:, i-1], 0)) * wt
                
    return r

def rf_10 (r_bar, r0_list, k, sigma, t, T, par = 1, maturity = 10):
    
    '''
    Takes in initial assumptions and par value of bond
    takes in n*1 matirx of r0_list (by CIR function)
    calculate explicit solution for ZCB with r following CIR (maturing @ T and discounted to t = t)
    returns: n*1 vector of annulized 10yr rf 
    '''
    
    h1 = np.sqrt(k ** 2 + 2 * sigma ** 2)
    h2 = (k + h1) / 2
    h3 = (2 * k * r_bar) / sigma ** 2
    
    A = (h1 * np.exp(h2 * (T-t)) / (h2 * (np.exp(h1 * (T-t)) - 1) + h1)) ** h3
    B = (np.exp(h1 * (T-t)) - 1) / (h2 * (np.exp(h1 * (T-t)) - 1) + h1)    
    
    bond_price = par * A * np.exp(-B * r0_list)
    
    rf_10_lag = - np.log(bond_price) / maturity
    
    return rf_10_lag


def CPR_Numerix (rf_10_lag, WAC, PV_lag, i, t_mon, PV0):
    
    """
    Numerix Prepayment Model of CPR 
    Takes in 10 year US treasury yield: rf_10_lag (from rf_10 function)
    Annulized mortage rate: WAC
    PV of t-1: PV_lag, 
    period: t,
    month of the specific payment at t: t_mon
    payment at t = 0: PV0
    Returnes n*1 vector of CPR_t
    """
    
    #factors
    RI = 0.28 + 0.14 * np.arctan(-8.57 + 430 * (WAC - rf_10_lag))
    BU = 0.3 + 0.7 * PV_lag / PV0
    SG = np.min([1, i / 30])
    seasonality = [0.94,0.76,0.74,0.95,0.98,0.92,0.98,1.10,1.18,1.22,1.23,0.98]
    SY = seasonality[t_mon-1]

    CPR_t = RI * BU * SG * SY
    
    return CPR_t
```
4. PV of MBS:
```python
def MBS(r_bar, r0, k, sigma, t, T, WAC, PV0, n, interval = 10, month = 12, oas = 0):
    
    '''
    Funtion to calculate PVCF of MBS 
    takes in initial value needed to calculate dt and ct 
    Return 1*1 vector of MBS price
    '''
    
    steps = int((T - t) * month * 10)
    dt = (T - t) / steps
    mort_r = WAC / month
    N = T * month
    PV_lag = PV0
    PVCF = 0 
    
    # simulate rf path for n simulations and each path have 10*12*30 steps 
    rf = CIR (T, t, r0, sigma, k, r_bar, n, month = 12, interval = 10)
    
    # calculate PVCF_t for each payment (30*12 payment in total)
    # expectation is taken within each time passing the loop
    for i in range(1, N + 1):

        # expected discount factor by taking the mean of rf from 0 - payment time (i*12)
        #d_t = np.mean(discount_factor(rf[:, :int(interval * i)] + oas, dt))
        d_t = np.mean(np.exp(-np.sum((rf[:, :int(interval * i)] + oas) * dt,axis=1)))
        
        # expected lag rf10 by pass r0 as the (i-1)*12 column of the simulated rf matrix 
        rf_10_lag = np.mean(rf_10(r_bar, rf[:, int(interval * (i-1))], k, sigma, (i-1) / 12, (i-1) / 12 + 10, par = 1, maturity = 10))
        # 1*1 CPR_t value given expected lag rf10
        t_mon = i % 12
        CPR_t = CPR_Numerix (rf_10_lag, WAC, PV_lag, i, t_mon, PV0)
        c_t = cashflow_t (PV_lag, CPR_t, mort_r, i, N)
        
        # cumalate PVCF
        PVCF += c_t * d_t
        
        # update PV_lag
        PV_lag = PV_lag * (1 + mort_r) - c_t 

    return PVCF
```
6. OAS-Adjusted Duration and Convexity
- $OAS_{duration} = \frac{P_- - P_+}{2yP_0}$
- $OAS_{convexity} = \frac{P_+ + P_- - 2 P_0}{2P_0y^2}$

```python
oas_d = oas - dr
oas_u = oas + dr

p_d = MBS(r_bar, r0, k, sigma, t, T, WAC, PV0, n, interval = 10, month = 12, oas = oas_d)
p_u = MBS(r_bar, r0, k, sigma, t, T, WAC, PV0, n, interval = 10, month = 12, oas = oas_u)

oas_dur = (p_d - p_u)/(2 * dr * p_mkt)
oas_conv = (p_u + p_d - 2 * p_mkt)/(2 * p_mkt * dr ** 2)
```

7. PO and IO
   
![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/88f8b253-8efa-4aea-9409-9e3b43043a74)

- Prices of PO tranches are inversely related to mortage rate and IO tranches are directly related to mortgage rates.


