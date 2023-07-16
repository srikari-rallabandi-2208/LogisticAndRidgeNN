def SGD(X, y, n_epochs, batch_size, eta, lmb=0):
    datapoints = X.shape[0]
    beta = np.random.randn(X.shape[1], 1)
    eta0 = eta
    for epoch in range(n_epochs):
        perm = np.random.permutation(datapoints)
        X_shuffled = X[perm, :]
        y_shuffled = y[perm, :]
        for i in range(0, datapoints, batch_size):
            x_i = X_shuffled[i:i+batch_size, :]
            y_i = y_shuffled[i:i+batch_size, :]
            gradient = (2 / batch_size) * (x_i.T @ ((x_i @ beta) - y_i) + lmb * beta)
            beta = beta - gradient * eta
    return beta
