'''
Simulate data from fake nudging experiment, then use STAN to estimate means with shrinkage.
'''

from IPython import embed
import numpy as np
import pandas as pd

def simulate_data(*, N=10000, K=3, sigma_alpha=0.05,sigma_beta=10):

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
micro, clp, alpha, beta = simulate_data()

###
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
y, N = clp['Y'], len(clp)

# Model
with pm.Model() as model:
    mu = pm.Normal('mu', 0, 10)
    sigma = pm.HalfNormal('sigma', 1)
    y = pm.Normal('y', mu, sigma, observed=clp['Y'])
    idata = pm.sample(1000)

fig = az.plot_trace(idata)
plt.savefig('idata.png')

az.summary(idata)


trace 
# Displaying the parameters
print('Mean:', trace['mu'].mean())
print('Variance:', trace['sigma'].var())


###
# Run STAN
import pystan

y, N = clp['Y'], len(clp)
# Write the Stan model.
model = """
data {
  int<lower=0> N;
  real y[N];
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  y ~ normal(mu, sigma);
}
"""

# Compile the Stan model.
model_code = pystan.StanModel(model_code=model)

# Sample from the posterior distribution.
samples = model_code.sample(data={"N": 100, "x": [1, 2, 3, 4, 5]})

# Estimate the parameters of the Gaussian distribution.
mean = samples["mu"].mean()
variance = samples["sigma"].var()

# Print the results.
print("Mean:", mean)
print("Variance:", variance)

# Plot results



