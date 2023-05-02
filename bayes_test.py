'''
Simulate data from fake nudging experiment, then use STAN to estimate means with shrinkage.
'''

from IPython import embed
import numpy as np
import pandas as pd

def simulate_data(*, N=10000, K=3, sigma_alpha=0.1, sigma_beta=10):

    # Generate treatments (D) as randomly assigned multinomial with K categories
    p = np.ones(K) / K
    #D = np.random.multinomial(n=1, pvals=p, size=N)
    D = np.random.choice(range(K), size=N, p=np.ones(K)/K)

    # Generate K action dummies (nudged by treatment D)
    alpha = np.random.normal(loc=0.1, scale=sigma_alpha, size=K)
    alpha[0] = 0 # control group
    A_noise = np.random.uniform(-0.1, 0.1, size=(N,K))
    A = np.zeros((N,K))
    for k in range(K):
        A_prob = np.clip(0.1 + alpha[k]*(D==k) + A_noise[:, k], 0, 1)
        A[:, k] = np.random.binomial(n=1, p=A_prob, size=N)

    # Generate outcomes driven by those actions
    beta = np.random.normal(loc=0, scale=sigma_beta, size=K)
    beta[0] = 0
    Y_noise =  np.random.normal(loc=0, scale=1, size=N)
    Y = 10 + np.sum(A*beta, axis=1) + Y_noise

    # Put {D, A, Y} into dataframe
    #df_D = pd.DataFrame(D, columns=[f'D{i+1}' for i in range(K)])
    micro_A = pd.DataFrame(A, columns=[f'A{k}' for k in range(K)]).astype(int)
    micro = pd.concat([pd.Series(D, name='D'), micro_A], axis=1)
    micro['Y'] = Y

    # Collapse data
    clp = micro.groupby('D').mean().reset_index()
    clp['n'] = micro.groupby('D').count()['Y'] 

    return micro, clp, alpha, beta

##
np.random.seed(123)
micro, clp, alpha, beta = simulate_data()

# Collapse data
# Run STAN
# Plot results