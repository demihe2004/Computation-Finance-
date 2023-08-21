# 1. 
Using the LGM method generate 10,000 Uniformly distributed random numbers on [0,1] and compute the empirical mean and the standard deviation of the sequence.
```python
def LGM(x0, m, a, b, n):
    """
    x0: starting number
    m,a,b: parameters
    n: # of generations
    return: 1*n vectore of Uniformly distributed random numbers
    """
    lgm = np.zeros(n)
    lgm[0] = x0
    for i in range(1, n):
        lgm[i] = (a * lgm[i-1] + b) % m
    lgm = lgm / m
    return lgm
```
 # 2. 
Generate 10,000 random numbers with specific distribution:
```python
GDD = np.zeros(10000)
lgm = LGM(X_0,m,a,0,10000)
for i in range(10000):
    if lgm[i] < 0.3:
        GDD[i] = -1
    elif lgm[i] < 0.65:
        GDD[i] = 0
    elif lgm[i] < 0.85:
        GDD[i] = 1
    else:
        GDD[i] = 2
```
# 3.
Binomial distribution
```python
p = 0.64
n = 44
binomial =  pd.DataFrame(np.empty((1000, n)))
uniform = LGM(X_0, m, a, 0, n*1000)

for i in range(n):
    binomial.loc[:, i] = uniform[(i*1000) : ((i+1)*1000)]
    for j in range(1000):
        if binomial.iloc[j,i] <= p:
            binomial.iloc[j,i] = 1
        else:
            binomial.iloc[j,i] = 0
```
# 4. 
Exponentially distributed
```python
gamma = 1.5
Exp = np.zeros(10000)
uniform = LGM(X_0, m, a, 0, 10000)
Exp = -np.log(uniform)/gamma
```
# 5. 
Box- Muller Method
```python
def BM(x0, m, a, n):
    """
    x0, m, a: para of LGM
    n: generator 2 pairs of n normally distributed random numbers
    return: n*4 dataframe with U1,U2,Z1,Z2
    """
    uniform = LGM(x0, m, a, 0, 2*n)
    x = int(n)
    BM = pd.DataFrame(index=range(x), columns=["U1", "U2"])
    BM["U1"] = uniform[0:x]
    BM["U2"] = uniform[x:(2*x)]
    BM['Z1'] = np.sqrt(-2*np.log(BM["U1"])) * np.cos(2*np.pi*BM["U2"])
    BM['Z2'] = np.sqrt(-2*np.log(BM["U1"])) * np.sin(2*np.pi*BM["U2"])
    return BM
```
# 6. 
Polar-Marsaglia method 
```python
def PM (x0, m, a, n, pr):
    """
    x0, m, a: para of LGM
    pr: prob of w<=1
    n: generator 2 pairs of n normally distributed random numbers
    return: 1*(2n) vector of Normally distributed random numbers
    """
    PM_n = math.ceil(n/pr)
    uniform = LGM(x0, m, a, 0, 2*PM_n)
    x = int(PM_n)
    PM = pd.DataFrame(index=range(x), columns=["U1", "U2"])
    PM["U1"] = uniform[0:x]
    PM["U2"] = uniform[x:(2*x)]
    PM['V1'] = 2*PM["U1"]-1
    PM['V2'] = 2*PM["U2"]-1
    PM['W'] = PM['V1']**2 + PM['V2']**2
    
    PM_generated = []
    for i in range(2*n-1):
        if PM['W'][i] <= 1:
            PM_generated.append(PM['V1'][i] * math.sqrt(-2 * math.log(PM['W'][i])/PM['W'][i]))
            PM_generated.append(PM['V2'][i] * math.sqrt(-2 * math.log(PM['W'][i])/PM['W'][i]))
        if len(PM_generated) > 2*n-1:
            break
    return PM_generated
```
