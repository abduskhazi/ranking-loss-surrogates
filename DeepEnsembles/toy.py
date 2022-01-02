# This is the first step to implement Deep Ensemble surrogate for a toy example.

import math
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot

from DeepEnsemble import Estimator, regression_criterion, train_estimator

rng = np.random.default_rng()


# Objective function
def objective(x, noise=0.1):
    y = x ** 2 * math.sin(5 * math.pi * x) ** 6
    y_noise = rng.normal(loc=0, scale=noise)
    return y + y_noise


# Returns the approx maximizer and the maxima of the objective function
# This is only for testing. It does not exist in the real world.
def get_maxima(obj_func):
    X = np.linspace(0.0, 1.0, 5000)
    Y = [objective(x, 0.0) for x in X]
    i = np.argmax(Y)
    maximizer = X[i]
    maxima = Y[i]
    return maximizer, maxima


maximizer, maxima = get_maxima(objective)
print("Actual values:")
print("Maximizer =", maximizer, ", maxima =", maxima)

# Reason for random samples
#     - In the actual problems we have access to only random samples and their objective evaluations
X = np.array(np.random.random(1000), dtype=np.float32)
Y = np.array([objective(x) for x in X], dtype=np.float32)  # Noisy evaluations.
X_copy = np.copy(X)
Y_copy = np.copy(Y)

X = X.reshape((-1, 1))
Y = Y.reshape((-1, 1))
estim = Estimator()
optimizer = optim.Adam(estim.parameters(), lr=0.01)
train_estimator(estim, X, Y, optimizer)

# Plotting the results
pyplot.scatter(X_copy, Y_copy, marker='.')
X = np.array(np.linspace(0.0, 1.0, 1000), dtype=np.float32)
Y = np.array([objective(x, 0.0) for x in X], dtype=np.float32)
pyplot.plot(X, Y)

Y_NN = estim(torch.from_numpy(X.reshape(-1, 1)))
mean_NN = Y_NN[:, 0].detach().numpy()  # Getting the mean out
variance_NN = Y_NN[:, 1]
std_dev = torch.sqrt(variance_NN).detach().numpy()
pyplot.plot(X, mean_NN)
pyplot.fill_between(X, mean_NN - std_dev, mean_NN + std_dev, alpha=0.45)
pyplot.savefig("results.png")
pyplot.show()

# Implementing Deep Ensembles.
M = 5  # Number of Neural Networks
