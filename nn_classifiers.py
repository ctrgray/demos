"""
Building intuition for backpropagation in classifiers
CG Jan 2025

I spend some of my time after surgery (Jan 2025) building intuition about transformers.
A key prereq of really understanding transformers is to visualize how multi-classifiers work in neural nets.
I built up some increasingly complex examples to get a feel for how this can be modularized & scaled up.

My notation: For each layer, l[i] = sigmoid(b[i] + l[i-1].dot(w[i]))
Backprop will work by feeding forward to store layers {l[i]},
    then backpropagating to store dL/dl[i] for each layer, 
        then using dL/dw[i] = dL/dl[i] * dl[i]/dw[i].
From there, most of the trick is not getting mixed up in the matrix algebra.
I haven't yet extended this to ReLUs or deeply investigated stability.
"""

##############################
## VERSION GIVEN BY CHATGPT ##

COPY_DATA = 1000
learning_rate = 1 / COPY_DATA
epochs = 10001
np.random.seed(42)

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


X = np.array(
    [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]]
)
y = np.array([[0], [1], [1], [0], [0], [1], [1], [0], [0], [0]])
X = np.tile(X, (COPY_DATA, 1))
y = np.tile(y, (COPY_DATA, 1))

# Initialize weights and biases
input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1

weights_input_hidden = np.random.rand(input_layer_neurons, hidden_layer_neurons)
weights_hidden_output = np.random.rand(hidden_layer_neurons, output_neurons)
bias_hidden = np.random.rand(hidden_layer_neurons)
bias_output = np.random.rand(output_neurons)

# Training parameters

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = (
        np.dot(X, weights_input_hidden) + bias_hidden
    )  # X.dot(W[0]) + b[0]
    hidden_layer_output = sigmoid(hidden_layer_input)  # L[0]

    output_layer_input = (
        np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    )  # L[0].dot(W[1])
    predicted_output = sigmoid(output_layer_input)  # L[1]

    # Calculate error
    error = y - predicted_output
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {mse_loss(y, predicted_output)}")

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(
        predicted_output
    )  # (y-p)*s'(L[1]) = dloss_darg[1]
    error_hidden_layer = d_predicted_output.dot(
        weights_hidden_output.T
    )  # derr[1] * W[1].T = dloss_dlayer[1]
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(
        hidden_layer_output
    )  # dloss_dlayer[1] * s'(L[0]) = dloss_darg[0]

    # Update weights and biases
    weights_hidden_output += (
        hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    )  # L[0].T.dot(dloss_darg[1]) * lr
    weights_input_hidden += (
        X.T.dot(d_hidden_layer) * learning_rate
    )  # X.T.dot(dloss_darg[0]) * lr
    bias_output += np.sum(d_predicted_output, axis=0) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0) * learning_rate


###############################
## WORKING SINGLE CLASSIFIER ##

import numpy as np
import pandas as pd
import random


def get_data():
    X = np.random.normal(size=[N, num_inputs])
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape([-1, 1])
    return X, y


def initialize_weights():
    weights = [np.random.normal(size=(num_inputs, num_nodes_per_layer))]
    bias = [np.random.normal(size=(num_nodes_per_layer, 1))]
    for i in range(num_layers - 2):
        weights.append(
            np.random.normal(size=(num_nodes_per_layer, num_nodes_per_layer))
        )
        bias.append(np.random.normal(size=(num_nodes_per_layer, 1)))
    weights.append(np.random.normal(size=(num_nodes_per_layer, num_outputs)))
    bias.append(np.random.normal(size=(num_outputs, 1)))
    return bias, weights


def feed_forward(weights):
    layers = []
    for i in range(num_layers):
        if i == 0:  # first layer
            layers += [sigmoid(X.dot(weights[i]) + bias[i].T)]
        elif i == num_layers - 1:  # last layer
            layers += [sigmoid(layers[i - 1].dot(weights[i]) + bias[i].T)]
        else:  # interior layers
            layers += [sigmoid(layers[i - 1].dot(weights[i]) + bias[i].T)]
    if num_outputs == 1:
        p = layers[-1].reshape([-1, 1])
        loss = np.mean((y - p) ** 2)
    return p, layers, loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(s):
    return s * (1 - s)


N = 1000
EPOCHS = 10001
learning_rate = 10 / N
num_inputs = 2
num_layers = 5
num_nodes_per_layer = 5
num_outputs = 1
np.random.seed(101)

X, y = get_data()
bias, weights = initialize_weights()

for iter in range(EPOCHS):
    p, layers, loss = feed_forward(weights)
    for lyr in np.arange(num_layers - 1, -1, -1):
        if lyr == num_layers - 1:  # final layer
            dloss_dlayer = y - p
        if lyr < num_layers - 1:  # interior layers
            dloss_dlayer = dloss_dlayer * sigmoid_derivative(layers[lyr + 1]).dot(
                weights[lyr + 1].T
            )
        if lyr > 0:
            dloss_dweight = layers[lyr - 1].T.dot(
                dloss_dlayer * sigmoid_derivative(layers[lyr])
            )
        if lyr == 0:
            dloss_dweight = X.T.dot(dloss_dlayer * sigmoid_derivative(layers[lyr]))
        dloss_dbias = np.sum(
            dloss_dlayer * sigmoid_derivative(layers[lyr]), axis=0
        ).reshape([-1, 1])
        weights[lyr] += learning_rate * dloss_dweight
        bias[lyr] += learning_rate * dloss_dbias
    if iter % 1000 == 0:
        print(f"{iter}: {loss}")


##############################
## WORKING MULTI-CLASSIFIER ##

import numpy as np
import pandas as pd
import random


def one_hot_encode(inputs, output):
    df = pd.DataFrame(inputs, columns=["input"])
    df["output"] = output
    X = np.array(
        pd.get_dummies(df["input"].apply(lambda x: pd.Series(list(x))))
    ).astype(int)
    y = np.array(
        pd.get_dummies(df["output"].apply(lambda x: pd.Series(list(x))))
    ).astype(int)
    return X, y


def initialize_weights():
    weights = [np.random.normal(size=(num_inputs, num_nodes_per_layer))]
    bias = [np.random.normal(size=(num_nodes_per_layer, 1))]
    for i in range(num_layers - 2):
        weights.append(
            np.random.normal(size=(num_nodes_per_layer, num_nodes_per_layer))
        )
        bias.append(np.random.normal(size=(num_nodes_per_layer, 1)))
    weights.append(np.random.normal(size=(num_nodes_per_layer, num_outputs)))
    bias.append(np.random.normal(size=(num_outputs, 1)))
    return bias, weights


def feed_forward(weights):
    layers = []
    for i in range(num_layers):
        if i == 0:  # first layer
            layers += [sigmoid(X.dot(weights[i]) + bias[i].T)]
        elif i == num_layers - 1:  # last layer is value that goes into softmax
            layers += [sigmoid(layers[i - 1].dot(weights[i]) + bias[i].T)]
        else:  # interior layers
            layers += [sigmoid(layers[i - 1].dot(weights[i]) + bias[i].T)]
    # Softmax
    denom = np.sum(np.exp(layers[-1]), axis=1)
    p = np.zeros((N, num_outputs))
    for i in range(num_outputs):
        p[:, i] = np.exp(layers[-1][:, i]) / denom
    # Loss
    loss = -np.sum(y * np.log(p)) / N
    return p, layers, loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(s):
    return s * (1 - s)


def get_accuracy(p, choices, output):
    predictions = [choices[int(x.argmax())] for x in p]
    accuracy = np.mean(np.array(predictions) == np.array(output))
    return float(accuracy)


N = 1000
learning_rate = 10 / N  # learning rate should decrease with N
EPOCHS = 5001
num_nodes_per_layer = 9  # quick converge with 9+ nodes, given our functional form
num_layers = 3
# ^ with <9 nodes per layer, more layers helps a little but doesn't substitute
np.random.seed(99)

choices = "abc"
inputs = ["".join(random.choices(choices, k=3)) for _ in range(N)]
output = [x[1] for x in inputs]
X, y = one_hot_encode(inputs, output)
num_inputs = X.shape[1]
num_outputs = y.shape[1]
bias, weights = initialize_weights()

for iter in range(EPOCHS):
    p, layers, loss = feed_forward(weights)
    for lyr in np.arange(num_layers - 1, -1, -1):
        # final layer: derivative of loss wrt softmax argument, dL/dl[-1]
        if lyr == num_layers - 1:
            dloss_dlayer = p - y
        # interior layers: l[i+1] = sigmoid(... + l[i]*w[i] + ...)
        # --> dL/dl[i] = dL/dl[i+1] * dl[i+1]/dl[i] = dL/dl[i+1] * sigmoid'(l[i+1]) * w[i+1]
        if lyr < num_layers - 1:  # interior layers:
            dloss_dlayer = (dloss_dlayer * sigmoid_derivative(layers[lyr + 1])).dot(
                weights[lyr + 1].T
            )
        # dL/dw[i] = dL/dl[i] * dl[i]/dw[i] = dL/dl[i] * sigmoid'(l[i]) * l[i-1]
        if lyr > 0:
            dloss_dweight = layers[lyr - 1].T.dot(
                dloss_dlayer * sigmoid_derivative(layers[lyr])
            )
        if lyr == 0:
            dloss_dweight = X.T.dot(dloss_dlayer * sigmoid_derivative(layers[lyr]))
        dloss_dbias = np.sum(
            dloss_dlayer * sigmoid_derivative(layers[lyr]), axis=0
        ).reshape([-1, 1])
        weights[lyr] += -learning_rate * dloss_dweight
        bias[lyr] += -learning_rate * dloss_dbias
    if iter % 1000 == 0:
        accuracy = get_accuracy(p, choices, output)
        print(f"{iter}: {accuracy}")
