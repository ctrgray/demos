'''
Simulate data from fake nudging experiment, then use STAN to estimate means with shrinkage.
'''

from IPython import embed
import numpy as np
import pandas as pd

def simulate_data(*, N=10000, K=10, sigma_alpha=0.05,sigma_beta=10):

    # Generate treatments (D) as randomly assigned multinomial with K categories
    D = np.random.choice(range(K), size=N, p=np.ones(K)/K)  # differentially popular

    # Generate K action dummies (nudged by treatment D)
    alpha = np.random.normal(loc=0, scale=sigma_alpha, size=K)  # dsitributional assumption
    A = np.zeros((N,K))
    for k in range(K):
        A_prob = np.clip(0.1 + alpha[k]*(D==k), 0, 1)
        A[:, k] = np.random.binomial(n=1, p=A_prob, size=N)

    # Generate outcomes driven by those actions
    beta = np.random.normal(loc=0, scale=sigma_beta, size=K)
    Y_noise =  np.random.normal(loc=0, scale=1, size=N)
    Y = 10 + np.sum(A*beta, axis=1) + Y_noise

    # Put {D, A, Y} into dataframe
    micro_A = pd.DataFrame(A, columns=[f'A{k}' for k in range(K)]).astype(int)
    micro = pd.concat([pd.Series(D, name='D'), micro_A], axis=1)
    micro['Y'] = Y

    # Collapse data
    clp = micro.groupby('D').mean().reset_index()
    clp['n'] = micro.groupby('D').count()['Y'] 

    return micro, clp, alpha, beta

##
np.random.seed(1234)
K=10
micro, clp, alpha, beta = simulate_data(K=K)

###
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

with pm.Model() as model:
    intercept = pm.Normal('intercept', mu=0, sigma=100)
    slope = pm.Normal('slope', mu=0, sigma=100)
    Y_mean = intercept 
    for k in range(K):
        Y_mean = Y_mean + slope*clp[f'A{k}']
    Y_sigma = pm.HalfNormal('Y_sigma', sigma=1)
    likelihood = pm.Normal('Y', mu=Y_mean, sigma=Y_sigma, observed=clp['Y'])
    idata = pm.sample(1000)

az.summary(idata)

fig = az.plot_trace(idata)
plt.savefig('trace.png')

fig = az.plot_posterior(idata)
plt.savefig('posterior.png')