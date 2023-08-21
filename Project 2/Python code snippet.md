# 1. 
Correlated pair of 1,000 Uniformly distributed random numbers
``` python
def Bivariate_rho(x0, m, lgm_a, n, mu1, mu2, sigma1, sigma2, a):
    """
    x0, m, lgm_a: para of LGM
    mu1, mu2, sigma1, sigma2: mean and vol vector
    return: 1*1 vetor of corr btw x and y generated 
    """
    rho=a/(sigma1*sigma2)
    biv = pd.DataFrame(index=range(n), columns=["Z1", "Z2"])
    biv["Z1"] = BM(x0, m, lgm_a, n).loc[:,['Z1']]
    biv["Z2"] = BM(x0, m, lgm_a, n).loc[:,['Z2']]
    biv["X"] = mu1 + sigma1 * biv["Z1"]
    biv["Y"] = mu2 + sigma2 * rho * biv["Z1"] + sigma2 * np.sqrt(1 - rho ** 2) * biv["Z2"]
    mu_X = np.mean(biv["X"])
    mu_Y = np.mean(biv["Y"])
    # close form for corr
    rho_a = ((np.sum((biv["X"] - mu_X) * (biv["Y"] - mu_Y)))/(n - 1))/(np.sqrt(np.sum((biv["X"] - mu_X) ** 2)/(n - 1))*np.sqrt(np.sum((biv["Y"] - mu_Y) ** 2)/(n - 1)))
    return rho_a
```
# 2. 
Expected value of Correlated random numbers
``` python
def Bivariate_E(x0, m, lgm_a, n, mu1, mu2, sigma1, sigma2, rho):
    """
    x0, m, lgm_a: para of LGM
    n: mc paths
    mu1, mu2, sigma1, sigma2, rho: mean and var-cov matrix 
    return: 1*1 vetor of mc expected value  
    """
    biv = pd.DataFrame(index=range(n), columns=["Z1", "Z2"])
    biv["Z1"] = BM(x0, m, lgm_a, n).loc[:,['Z1']]
    biv["Z2"] = BM(x0, m, lgm_a, n).loc[:,['Z2']]
    biv["X"] = mu1+sigma1*biv["Z1"]
    biv["Y"] = mu2+sigma2*rho*biv["Z1"]+sigma2*np.sqrt(1-rho**2)*biv["Z2"]
    biv["MC"] = np.maximum(0, (biv["Y"] ** 3 + np.sin(biv["Y"]) + biv["X"] ** 2 * biv["Y"]))
    expected_value = np.mean(biv["MC"])
    return expected_value
```

# 3. 
Generate Wt
``` python
def weiner_generator (time):
    output = pd.DataFrame(index=time, columns=["E_At", "E_Bt"])
    genarator = pd.DataFrame(index=range(n), columns=["Z"])
    genarator["Z"] = BM(x0, m, lgm_a, n).loc[:,['Z1']]
    for t in time:
        genarator["wt"] = genarator.iloc[:,[0]] * np.sqrt(t)
        genarator["At"] = genarator["wt"]**2+np.sin(genarator["wt"])
        genarator["Bt"] = np.exp(t/2)*np.cos(genarator["wt"])
        output.loc[t,"E_At"] = np.mean(genarator["At"])
        output.loc[t,"E_Bt"] = np.mean(genarator["Bt"])
    return output
```
Reduce variance uisng control variates
``` python
def weiner_generator_3c (time):
    output = pd.DataFrame(index=time, columns=["E_At", "E_Bt"])
    genarator = pd.DataFrame(index=range(n), columns=["Z"])
    x0 = np.random.randint(1, 100)
    genarator["Z"] = BM(x0, m, lgm_a, n).loc[:,['Z1']]
    for t in time:
        genarator["wt"] = genarator.iloc[:,[0]] * np.sqrt(t)
        genarator["At"] = genarator["wt"]**2+np.sin(genarator["wt"])
        genarator["Bt"] = np.exp(t/2)*np.cos(genarator["wt"])
        output.loc[t,"E_At"] = np.mean(genarator["At"])
        output.loc[t,"E_Bt"] = np.mean(genarator["Bt"])
    return genarator
```
# 4. 
MC simulation to calculate payoff
``` python
def MC_call (r,s0,sigma,x):
    genarator = pd.DataFrame(index=range(n), columns=["Z"])
    genarator["Z"] = BM(x0, m, lgm_a, n).loc[:,['Z1']]
    genarator["wt"] = genarator.iloc[:,[0]] * np.sqrt(5)
    genarator["st"] = s0*np.exp(sigma*genarator["wt"]+(r-sigma**2/2)*5)
    genarator["payoff"] = np.maximum(0, (genarator["st"]-x))
    call_MC = np.exp(-r*5)*np.mean(genarator["payoff"] )
    return call_MC
```
Reduce variance using Antithetic Variate
``` python
def MC_call_anti (r,s0,sigma,x):
    genarator = pd.DataFrame(index=range(n), columns=["Z"])
    genarator["Z"] = BM(x0, m, lgm_a, n).loc[:,['Z1']]
    #get Antithetic Variates by using -zt
    genarator["wt_1"] = genarator.iloc[:,[0]] * np.sqrt(5)
    genarator["wt_2"] = -genarator.iloc[:,[0]] * np.sqrt(5)
    genarator["st_1"] = s0*np.exp(sigma*genarator["wt_1"]+(r-sigma**2/2)*5)
    genarator["st_2"] = s0*np.exp(sigma*genarator["wt_2"]+(r-sigma**2/2)*5)
    genarator["payoff_1"] = np.maximum(0, (genarator["st_1"]-x))
    genarator["payoff_2"] = np.maximum(0, (genarator["st_2"]-x))
    # get (payoff1+ payoff2)/2 
    genarator["anti"] = (genarator["payoff_1"]+genarator["payoff_2"])/2
    corr_anti = genarator["payoff_1"].corr(genarator["payoff_2"])
    call_MC_anti = np.exp(-r*5)*np.mean(genarator["anti"])
    return corr_anti,call_MC_anti
```
# 5. 
Simulate underlying path

![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/a1621d91-4ca3-4f63-bebf-9be35d4527ac)

Increase underlying vol 

![image](https://github.com/demihe2004/Computation-Finance-/assets/135466801/6243766e-0a78-45b8-9cbe-9f4cd365c5a5)

