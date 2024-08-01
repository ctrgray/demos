"""

source venv/bin/activate
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

def generate_data(N, *, confound_strength=1, Y_corr=0.5, lift_1=1, lift_2=1):
    "Generate synthetic data with confounding and endogeneity."

    # Generate correlated X columns
    X_mean, X_cov = [0, 0], [[1, 0.5], [0.5, 1]]
    X1, X2 = np.random.multivariate_normal(X_mean, X_cov, N).T

    # Generate endogenous treatment (D)
    D_err = np.random.normal(0, 1, N)
    D = (confound_strength*(X1 + X2**2) + D_err > 2).astype(int)

    # Generate two noisy correlated outcomes (Y1,Y2)
    Y_err_mean, Y_err_cov = [0, 0], [[1, Y_corr], [Y_corr, 1]]
    Y_err_1, Y_err_2 = np.random.multivariate_normal(Y_err_mean, Y_err_cov, N).T
    Y1 = sigmoid(X1) + np.sin(X2) + lift_1*D + Y_err_1
    Y2 = np.sin(X1) + sigmoid(X2) + lift_2*D + Y_err_2
    return pd.DataFrame({'idx': np.arange(N), 'X1': X1, 'X2': X2, 'D': D, 'Y1': Y1, 'Y2': Y2})

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_heatmap(data, colm):
    "Generate heatmap of outcome wrt X1, X2"
    df = data.copy()
    # Pivot data to capture average Y2 in each cell
    bins_x1 = np.linspace(df['X1'].min(), df['X1'].max(), 50)
    bins_x2 = np.linspace(df['X2'].min(), df['X2'].max(), 50)
    df['X1_bin'] = pd.cut(df['X1'], bins=bins_x1)
    df['X2_bin'] = pd.cut(df['X2'], bins=bins_x2)
    pivot_table = df.pivot_table(values=colm, index='X1_bin', columns='X2_bin')
    # Plot
    cmap = plt.cm.viridis
    cmap.set_under(color='white')  # Set color for zero values
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot_table.values, origin='lower', extent=[bins_x1[0], bins_x1[-1], bins_x2[0], bins_x2[-1]], cmap=cmap, norm=colors.LogNorm(vmin=0.1))
    plt.colorbar(label=f'Average {colm}')
    plt.title(f'Heatmap of average {colm} with respect to X1 and X2')
    plt.savefig('temp.png')

####

np.random.seed(0)
data = generate_data(N=100000)
data.describe()
data.groupby('D').mean()
np.corrcoef(data['Y1'], data['Y2'])
generate_heatmap(data, 'Y1')