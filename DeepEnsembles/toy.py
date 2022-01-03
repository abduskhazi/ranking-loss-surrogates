# This is the first step to implement Deep Ensemble surrogate for a toy example.

import math
import numpy as np
import torch
from matplotlib import pyplot
from scipy.stats import norm
from DeepEnsemble import DeepEnsemble

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

# We need to use Deep Ensemble as a surrogate to predict the maxima of the objective function.
# The Deep Ensemble takes over the functionality of a gaussian process model for estimating the mean
# and the variance of the loss surface at a given input.

# Getting a small list of random samples:
#    In the actual problems we have access to only random samples and their objective evaluations
#    These random samples serve as a beginning data to train the Deep Ensemble with.
X = np.array(np.random.random(50), dtype=np.float32)
Y = np.array([objective(x) for x in X], dtype=np.float32)  # Noisy evaluations of objective function.

# Deep Ensembles = surrogate for our optimization problem.
DE = DeepEnsemble(M=5)  # M = Number of Neural Networks
DE.train(X.reshape(-1, 1), Y.reshape(-1, 1), epochs=100, batch_size=10)

def plot(X, Y, model):
    # Plotting the results after the optimisation cycle.
    pyplot.scatter(X, Y, marker='.') # Plotting noisy data points
    X = np.array(np.linspace(0.0, 1.0, 1000), dtype=np.float32)
    Y = np.array([objective(x, 0.0) for x in X], dtype=np.float32)
    pyplot.plot(X, Y) # Plotting true objective function
    mean, variance = model.predict(torch.from_numpy(X.reshape(-1, 1)))
    mean = mean.detach().numpy()
    std_dev = torch.sqrt(variance).detach().numpy()
    pyplot.plot(X, mean)
    pyplot.fill_between(X, mean - std_dev, mean + std_dev, alpha=0.45)
    # pyplot.savefig("results_initial_fit_DE.png")
    pyplot.show()

# Defining the acquisition function for deep ensembles.
# Using probability of improvement for the acquisition function
def acquisition_PI(Y, X_samples, surrogate_model):
    # Find the best value of the objective function so far according to data.
    best_y = np.max(Y)
    # Calculate the predicted mean & variance values of all the required samples
    mean, variance = surrogate_model.predict(X_samples)
    mean = mean.detach().numpy()
    std_dev = torch.sqrt(variance).detach().numpy()
    return norm.cdf((mean - best_y) / (std_dev + 1E-9))

# Upper Confidence Bound acquisistion function
def acquisition_UCB(X_samples, surrogate_models):
    mean, variance = surrogate_models.predict(X_samples)
    mean = mean.detach().numpy()
    std_dev = torch.sqrt(variance).detach().numpy()
    ucb = mean + 2 * std_dev
    return ucb


# Defining a routine that finds the next best input sample according to the acquisition function.
def optimize_acquisition(Y, surrogate_model):
    X_samples = np.array(np.random.random(10000), dtype=np.float32)
    X_samples = torch.from_numpy(X_samples.reshape(-1, 1))
    # scores = acquisition_PI(Y, X_samples, surrogate_model)
    scores = acquisition_UCB(X_samples, surrogate_model)
    ind = np.argmax(scores)
    return X_samples[ind]


# Plotting just before the optmization cycle
plot(np.copy(X), np.copy(Y), DE)

# Running the optimisation cycle
#   Get the next best input sample
#   Evaluate the sample on the objective function (expensive step)
#   Append the data to the existing data points
#   Refit our surrogate (i.e Deep Ensembles)
print("Running optimisation cycle")
for _ in range(100):
    print("Iteration:", _, end="\r")
    x_opt = optimize_acquisition(Y, DE)
    x_opt = x_opt.detach().numpy()
    y_opt = objective(x_opt)
    X = np.append(X, x_opt, axis=0)
    Y = np.append(Y, y_opt, axis=0)
    # Training the neural networks only for a small number of epochs
    # Rationale - Most of the training has been completed, only slight modification needs to
    #             be done due to an additional data point.
    DE.train(X.reshape(-1, 1), Y.reshape(-1, 1), epochs=100, batch_size=20)
    #if _ % 10 == 0:
    #    plot(np.copy(X), np.copy(Y), DE)

plot(np.copy(X), np.copy(Y), DE)
print()
print("After optimization")
print("Maximizer =", X[np.argmax(Y)], ", Maxima =", np.max(Y))
