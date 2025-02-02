

for i in range(1000):
    # Forward pass
    layers = [np.maximum(0, X.dot(w[0]))]
    for j in range(num_layers - 2):
        layers.append(np.maximum(0, layers[-1].dot(w[j + 1])))
    layers.append(layers[-1].dot(w[-1]))

    # Stable softmax
    exps = np.exp(layers[-1] - layers[-1].max(axis=1))  # Remove keepdims
    denom = exps.sum(axis=1)
    pred = exps / denom
    
    # Compute loss
    loss = -np.mean(np.sum(y * np.log(pred + 1e-9), axis=1))  # Stability
    if i % 100 == 0:
        print(f"LOSS: {loss}")

    # Backpropagation
    loss_at_stake = (pred - y) / N  # Softmax derivative
    _dw = [layers[-2].T.dot(loss_at_stake)]  # Last layer gradient

    for j in range(num_layers - 2, 0, -1):  # Hidden layers
        loss_at_stake = (loss_at_stake.dot(w[j + 1].T)) * relu_derivative(layers[j])
        _dw.insert(0, layers[j - 1].T.dot(loss_at_stake))

    loss_at_stake = (loss_at_stake.dot(w[1].T)) * relu_derivative(layers[0])  # First layer
    _dw.insert(0, X.T.dot(loss_at_stake))

    # Update weights
    for j in range(len(w)):
        w[j] -= learning_rate * _dw[j]
