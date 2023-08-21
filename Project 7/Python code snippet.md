# 1. 
Explicit Finite-Difference method

$$
A = \begin{bmatrix}
a_2 & a_1 & 0 & 0 & \dots & 0 & 0 & 0 & 0\\
a_3 & a_2 & a_1 & 0 & \dots & 0 & 0 & 0 & 0\\
0 & a_3 & a_2 & a_1 & \dots & 0 & 0 & 0 & 0\\
\vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots\\
0 & 0 & 0 & 0 & \dots & a_3 & a_2 & a_1 & 0\\
0 & 0 & 0 & 0 & \dots & 0 & a_3 & a_2 & a_1\\
0 & 0 & 0 & 0 & \dots & 0 & 0 & a_3 & a_2\\
\end{bmatrix}
\qquad
C_i = \begin{bmatrix}
C_{i, N-1}\\
C_{i, N-2}\\
C_{i, N-3}\\
\vdots\\
C_{i, 3}\\
C_{i, 2}\\
C_{i, 1}\\
\end{bmatrix}
\qquad 
B = \begin{bmatrix}
d_{i+1, N-1}\\
d_{i+1, N-2}\\
d_{i+1, N-3}\\
\vdots\\
d_{i+1, 3}\\
d_{i+1, 2}\\
d_{i+1, 1}\\
\end{bmatrix}
$$

Where $AC_i = B_{i+1}$

$$
\begin{cases}
C_{i, N} & = C_{i, N-1}\\
C_{i, N-1} & = P_u C_{i+1, N} + P_m C_{i+1, N - 1} + P_dC_{i+1, N - 2}\\
&\vdots\\
C_{i, -N+1} & = P_u C_{i+1, -N+2} + P_m C_{i+1, -N+1} + P_dC_{i+1, -N}\\
C_{i, -N} & = C_{i, -N+1} + (S_{i, -N+1} - S_{i, -N})\\
\end{cases}
$$

Where

$$
\begin{cases}
P_u & = \Delta t(\frac{\sigma^2}{2(\Delta X)^2} + \frac{r - \frac{\sigma^2}{2}}{2\Delta X})\\
P_m & = 1 - \Delta t\frac{\sigma^2}{\Delta X^2} - r\Delta t\\
P_d & = \Delta t(\frac{\sigma^2}{2(\Delta X)^2} - \frac{r - \frac{\sigma^2}{2}}{2\Delta X})\\
\end{cases}
$$

``` python
def EFD_euro_put_x (st_range, dx, dt, T, K, sigma, r):
    
    step = int(T/dt)
    
    Pu = dt* (sigma**2/dx**2 + (r - sigma**2/2)/dx)/2
    Pm = 1 - dt* sigma**2/ dx**2 - r * dt
    Pd = dt* (sigma**2/dx**2 - (r - sigma**2/2)/dx)/2
    
    x_range = np.arange(np.log(st_range[-1]) + 100 * dx, np.log(st_range[0])- 100 * dx, -dx)
    
    F = np.maximum(K - np.exp(x_range),0)
    
    N = len(F)
    
    B = np.zeros((N))
    B[-1] = np.exp(x_range[-2])-np.exp(x_range[-1])
    
    A = np.zeros((N,N))
    A[0, 0:3] = [Pu, Pm, Pd]
    for i in range(1, N-1):
        start = int(i-1)
        A[i, start:start+3] = [Pu, Pm, Pd]
    A[-1] = A[-2]
    
    F = F.T
    for i in range(step):
        F = np.dot(A,F) + B
    
    optimal_j = [np.abs(x_range - np.log(st)).argmin() for st in st_range]
    F0 = F[optimal_j]
    return F0
```
![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/a564efd3-d410-47d0-b5c4-9d69c729b11c)

# 2. 
Implicit Finite-Difference method

$$
\begin{cases}
C_{i, N} - C_{i, N-1} & = 0\\
P_u C_{i, N} + P_m C_{i+1, N - 1} + P_dC_{i, N - 2} &= C_{i+1, N-1}\\
&\vdots\\
P_u C_{i, -N+2} + P_m C_{i, -N+1} + P_dC_{i, -N} &= C_{i+1, -N+1}\\
C_{i, -N} - C_{i, -N+1} & = S_{i, -N+1} - S_{i, -N}
\end{cases}
$$

where

$$
\begin{cases}
P_u & = -\frac{1}{2}\Delta t(\frac{\sigma^2}{\Delta X^2} + \frac{v}{\Delta X})\\
P_m &= 1 + \Delta t \frac{\sigma^2}{\Delta X^2} + r \Delta t\\
P_d &= -\frac{1}{2}\Delta t(\frac{\sigma^2}{\Delta X^2} - \frac{v}{\Delta X})
\end{cases}
$$

where 
$$
v = r - \frac{\sigma^2}{2}
$$

```python
def IFD_euro_put_x (st_range, dx, dt, T, K, sigma, r):
    
    step = int(T/dt)
    
    Pu = - dt* (sigma**2/dx**2 + (r - sigma**2/2)/dx)/2
    Pm = 1 + dt* sigma**2/ dx**2 + r * delta_t
    Pd = - dt* (sigma**2/dx**2 - (r - sigma**2/2)/dx)/2
    
    x_range = np.arange(np.log(st_range[-1]) + 100 * dx, np.log(st_range[0])- 100 * dx, -dx)
    N = len(x_range)
    
    F = np.zeros((N, step + 1))
    F[:,-1] = np.maximum(K - np.exp(x_range), 0)

    A = np.zeros((N,N))
    A[0, 0:2] = [-1, 1]
    for i in range(1, N-1):
        start = int(i-1)
        A[i, start:start+3] = [Pu, Pm, Pd]
    A[-1, N-2:N] = [-1, 1]
    
    for i in range(step - 1, -1, -1):
        B = F[:, i+1]
        B[0] = 0
        B[-1] = np.exp(x_range[-2]) - np.exp(x_range[-1])
        F[:,i] = np.linalg.inv(A).dot(B)
    
    optimal_j = [np.abs(x_range - np.log(st)).argmin() for st in st_range]
    
    F0 = F[:,1][optimal_j]
    return F0
```
![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/5c63f9cf-59cf-43de-a0cf-1b23bda624a7)

# 3. 
Crank-Nicolson Finite-Difference method

$$
\begin{cases}
C_{i, N} - C_{i, N-1} & = z_{i+1, N} = 0\\
P_u C_{i, N} + P_m C_{i+1, N - 1} + P_dC_{i, N - 2} &= z_{i+1, N-1}\\
&\vdots\\
P_u C_{i, -N+2} + P_m C_{i, -N+1} + P_dC_{i, -N} &= z_{i+1, -N+1}\\
C_{i, -N} - C_{i, -N+1} & = z_{i+1, -N}\\
\end{cases}
$$

Where

$$
\begin{cases}
z_{i+1, N} &= 0\\
z_{i+1, N-1} & = - P_u C_{i+1, N} - (P_m -2)C_{i+1, N-1} - P_d C_{i+1, N - 2}\\
&\vdots\\
z_{i+1, -N+1} & =  - P_u C_{i+1, -N+2} - (P_m -2)C_{i+1, -N+1} - P_d C_{i+1, -N}\\
z_{i+1, -N} & = S_{i, -N + 1} - S_{i, -N}\\
\end{cases}
$$

$$
\begin{cases}
P_u & = -\frac{1}{4}\Delta t(\frac{\sigma^2}{\Delta X^2} + \frac{v}{\Delta X})\\
P_m &= 1 + \Delta t \frac{\sigma^2}{2(\Delta X)^2} + \frac{r \Delta t}{2}\\
P_d &= -\frac{1}{4}\Delta t(\frac{\sigma^2}{\Delta X^2} - \frac{v}{\Delta X})
\end{cases}
$$

where 
$$
v = r - \frac{\sigma^2}{2}
$$

```python
def CNFD_euro_put_x (st_range, dx, dt, T, K, sigma, r):
    
    step = int(T/dt)
    
    Pu = -dt* (sigma**2/dx**2 + (r - sigma**2/2)/dx)/4
    Pm = 1 + dt* sigma**2/ (2 * dx**2) + r * dt/2
    Pd = -dt* (sigma**2/dx**2 - (r - sigma**2/2)/dx)/4
    
    x_range = np.arange(np.log(st_range[-1]) + 100 * dx, np.log(st_range[0])- 100 * dx, -dx)
    N = len(x_range)
    
    F = np.zeros((N, step + 1))
    F[:,-1] = np.maximum(K - np.exp(x_range), 0)

    A = np.zeros((N,N))
    A[0, 0:2] = [-1, 1]
    for i in range(1, N-1):
        start = int(i-1)
        A[i, start:start+3] = [Pu, Pm, Pd]
    A[-1, N-2:N] = [-1, 1]
    
    for i in range(step - 1, -1, -1):
        B = np.zeros(N)
        c = F[:, i+1]
        B[1:-1] = -Pu * c[:-2] - (Pm - 2) * c[1:-1] - Pd * c[2:]
        B[0] = 0
        B[-1] = np.exp(x_range[-2]) - np.exp(x_range[-1])
        
        F[:,i] = np.linalg.inv(A).dot(B)
    
    optimal_j = [np.abs(x_range - np.log(st)).argmin() for st in st_range]
    
    F0 = F[:,1][optimal_j]
    return F0
```
![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/ec24b213-c10e-4d1f-8052-0646ce3bc83f)

**Observations**
- All of the Finite-Difference methods are quite accurate when comparing result to BSM prices. 
- According to the total difference chart above, IFD with 𝛥𝑋=$\sqrt{\Delta t}$ seems to be doing the best estimation
- 𝛥𝑋=$\sqrt{\Delta 3t}$ seems be doing the worst estimation

# 4. 
generalized finite-difference

$$
A = \begin{bmatrix}
a_2 & a_1 & 0 & 0 & \dots & 0 & 0 & 0 & 0\\
a_3 & a_2 & a_1 & 0 & \dots & 0 & 0 & 0 & 0\\
0 & a_3 & a_2 & a_1 & \dots & 0 & 0 & 0 & 0\\
\vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots\\
0 & 0 & 0 & 0 & \dots & a_3 & a_2 & a_1 & 0\\
0 & 0 & 0 & 0 & \dots & 0 & a_3 & a_2 & a_1\\
0 & 0 & 0 & 0 & \dots & 0 & 0 & a_3 & a_2\\
\end{bmatrix}
\qquad
C_i = \begin{bmatrix}
C_{i, N-1}\\
C_{i, N-2}\\
C_{i, N-3}\\
\vdots\\
C_{i, 3}\\
C_{i, 2}\\
C_{i, 1}\\
\end{bmatrix}
\qquad 
B = \begin{bmatrix}
d_{i+1, N-1}\\
d_{i+1, N-2}\\
d_{i+1, N-3}\\
\vdots\\
d_{i+1, 3}\\
d_{i+1, 2}\\
d_{i+1, 1}\\
\end{bmatrix}
$$

```python
def GFD_American_option(sigma, r, dt, dS, N, alpha, K, st_range,T, call_or_put):
    
    step = int(T / dt)
    ds_step = np.arange(0, N * dS, dS)

    # get parameter a1 a2 a3 b1 b2 b3
    a1, a2, a3 = [], [], []
    b1 = np.zeros(N-2)
    b2 = np.zeros(N-2)
    b3 = np.zeros(N-2)

    for j in range(1, N - 1):
        a1.append((sigma**2 * j**2 - r * j) * (1 - alpha) / 2)
        a2.append(-1 /dt - (sigma**2 * j**2 + r) * (1-alpha))
        a3.append((sigma**2 * j**2 + r * j) * (1 - alpha) / 2)

        b1[j-1] = (sigma**2 * j**2 - r * j) * alpha / 2
        b2[j-1] = 1 / dt - (sigma**2 * j**2 + r) * alpha
        b3[j-1] = (sigma**2 * j**2 + r * j) * alpha / 2

    # get A matrix 
    A = np.zeros((N-2,N-2))
    A[0, 0:2] = [a2[0], a3[0]]
    for i in range(1, N-3):
        start = int(i-1)
        A[i, start:start+3] = [a1[i], a2[i], a3[i]]
    A[-1, N-4:N-2] = [a1[-1], a2[-1]]

    # get V as payoff matrix
    V = np.zeros((N, step + 1))
    if call_or_put == 'call':
        V[:,-1] = np.maximum(ds_step - K, 0)
    elif call_or_put == 'put':
        V[:,-1] = np.maximum(K - ds_step, 0)

    for i in range(steps - 1, -1, -1): 
        Vi = V[:, i+1]
        # get B matrix
        B = np.zeros(M - 2)
        B = -b1 * Vi[:-2] - b2 * Vi[1:-1] - b3 * Vi[2:]

        # get c matrix for each time step
        Ci = np.zeros(N)
        Ci[1:-1] = np.linalg.inv(A).dot(B)
        if call_or_put == 'call':
            Ci[0] = Ci[1] + S[0] - S[1]
            Ci[-1] = Ci[-2]
        elif call_or_put == 'put':
            Ci[0] = Ci[1]
            Ci[-1] = Ci[-2] + S[-2] - S[-1]

        #compare cv and ev at each step
        V[:,i] = np.maximum(Ci, V[:,-1])

    optimal_j = [abs(ds_step - st).argmin() for st in st_range]
    V0 = V[:,1][optimal_j]   
    return V0
```
![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/7996c307-56e0-400c-92cb-63948a3c50f3)



