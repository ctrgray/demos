import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

np.random.seed(42)

def simulate_data(*, N, weights, means, vars):
    assert len(weights) == len(means) == len(vars), "weights, means, and vars must be same length"
    K = len(weights)
    for k in range(K):
        tmp = pd.DataFrame({'y': np.random.normal(means[k], vars[k], int(weights[k]*N)), 'grp': k})
        if k ==0:
           data = tmp 
        else:
           data = pd.concat([data, tmp], axis=0)
    return data


#
data = simulate_data(N=1000, weights=[0.8,0.2], means=[0,10], vars=[1,1])
    
data

#
N, K = data.shape
y = data['y']

# Initialize guesses
means = np.zeros(K)
vars = np.ones(K)
weights = np.ones(K) / K

i = 0
# E Step: Calculate expectations of latent data ("responsibilities")
responsibilities = np.zeros((N, K))
for k in range(K):
    responsibilities[:, k] = weights[k] * multivariate_normal.pdf(y, means[k], vars[k])
responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
responsibilities

# M Step: Update parameters conditional on latent data



x=1
x/=2
x


from scipy.stats import multivariate_normal

def em_algorithm(data, num_components, num_iterations):
    num_points, data_dim = data.shape

    # Initialize random means and covariances for the Gaussian components
    means = [np.random.rand(data_dim) for _ in range(num_components)]
    covariances = [np.eye(data_dim) for _ in range(num_components)]
    weights = np.ones(num_components) / num_components

    for _ in range(num_iterations):
        # E-step: Calculate responsibilities (posterior probabilities) of data points
        responsibilities = np.zeros((num_points, num_components))
        for i in range(num_components):
            responsibilities[:, i] = weights[i] * multivariate_normal.pdf(data, means[i], covariances[i])

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        # M-step: Update means, covariances, and weights based on responsibilities
        for i in range(num_components):
            total_weight = np.sum(responsibilities[:, i])
            means[i] = np.dot(responsibilities[:, i], data) / total_weight
            covariances[i] = np.dot(responsibilities[:, i] * (data - means[i]).T, (data - means[i])) / total_weight
            weights[i] = total_weight / num_points

    return means, covariances, weights
