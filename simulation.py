"""
Simple simulations to test drive some tools
CG Aug 2024

source venv/bin/activate
"""


from keras.models import Sequential
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

def generate_data(N, *, Y_corr=0.5, Y_noise=0.2, lift_1=1, lift_2=1):
    "Generate synthetic data with confounding and endogeneity."

    # Generate correlated X columns
    X_mean, X_cov = [0, 0], [[1, 0.5], [0.5, 1]]
    X1, X2 = np.random.multivariate_normal(X_mean, X_cov, N).T

    # Generate endogenous treatment (D)
    D = (X1 + X2 > np.random.normal(0, 1, N)).astype(int)

    # Generate two noisy correlated outcomes (Y1,Y2)
    Y_err_mean, Y_err_cov = [0, 0], [[1, Y_corr], [Y_corr, 1]]
    Y_err_1, Y_err_2 = np.random.multivariate_normal(Y_err_mean, Y_err_cov, N).T
    Y1 = 2*sigmoid(X1) + np.sin(2*X2) + lift_1*D + Y_noise*Y_err_1
    Y2 = np.sin(2*X1) + 2*sigmoid(X2) + lift_2*D + Y_noise*Y_err_2
    return pd.DataFrame({'idx': np.arange(N), 'X1': X1, 'X2': X2, 'D': D, 'Y1': Y1, 'Y2': Y2})

def check_data():
    data.describe()
    data.groupby('D').mean()
    np.corrcoef(data['Y1'], data['Y2'])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_heatmap(data, colm, filename='temp.png'):
    "Generate heatmap of outcome wrt X1, X2"
    fig = plt.scatter(data['X1'], data['X2'], c=data['Y1'], cmap='coolwarm', alpha=0.3)
    plt.colorbar(fig, label='Y1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig(filename)
    plt.close()


def predict_with_neural_net(df, yvar, *, layers, epochs, plot_figure=True):
    "Predict yvar using X1, X2 with a neural network."

    # Define the model
    model = Sequential()
    model.add(Input(shape=(2,)))
    for i in range(layers):
        print(f"Add layer {i}")
        model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model
    X = np.array(df[['X1', 'X2']]).astype(np.float32)
    Y = np.array(df[yvar].values).astype(np.float32)
    model.fit(X, Y, epochs=epochs, batch_size=32)
    df['pred'] = model.predict(X)

    # Optional plot
    if plot_figure is True:
        plt.scatter(df['pred'], df[yvar], alpha=0.3)
        plt.plot(df['pred'], df['pred'], color='black')
        plt.xlabel(f'Predicted {yvar}')
        plt.ylabel(f'Actual {yvar}')
        plt.savefig('compare.png')
        plt.close()

    return df

####

np.random.seed(0)
data = generate_data(N=10000)

control_data = data.loc[data['D']==0].copy()
generate_heatmap(control_data, 'Y1', filename='heatmap.png')
predict_with_neural_net(control_data.copy(), 'Y1', layers=10, epochs=20)


