import math
import numpy as np
from matplotlib import pyplot
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm

rng = np.random.default_rng()


# Objective function
def objective(x, noise=0.0):
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


# Can be used to ignore warnings as well.
def surrogate(model, X):
    yhat_mean, yhat_std = model.predict(X, return_std=True)
    return yhat_mean, yhat_std


# Using probability of improvement for the acquisition function.
#    1. Calculate the best sample so far.
#    2. Use the mean and std of all samples to calculate the respective PIs
#    3. PI = cumulative probability.
def acquisition(Y, X_samples, model):
    best = np.max(Y)
    mean, std = surrogate(model, X_samples)
    return norm.cdf((mean - best) / (std + 1E-9))

def acquisition_EI(Y, X_samples, model):
    # https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec
    # Find the best value of the objective function so far according to data.
    # Is this according to the gaussian fit or according to the actual values.???
    # For now using the best according to the actual values.
    best_y = np.max(Y)
    # Calculate the predicted mean & variance values of all the required samples
    mean, variance = surrogate(model, X_samples)
    std_dev = np.sqrt(variance)
    z = (mean - best_y) / (std_dev + 1E-9)
    return (mean - best_y) * norm.cdf(z) + (std_dev + 1E-9) * norm.pdf(z)

# Upper Confidence Bound acquisistion function
# Beta defines the number of std_dev to take into account.
def acquisition_UCB(X_samples, model, beta):
    mean, std_dev = surrogate(model, X_samples)
    ucb = mean + beta * std_dev
    return ucb

# Optimization of the acquisition function
#     1. Search strategy for samples - Here it is random sampling
#     2. Run the acquisition function for all samples
#     3. Find the best sample and then return the maximizer
def opt_aquisition(Y, X, model):
    # Sampling from the whole domain instead of randomly sampling domain values
    X_samples = np.array(np.linspace(0.0, 1.0, 100000), dtype=np.float32)
    X_samples = X_samples.reshape((-1, 1))
    # scores = acquisition_UCB(X_samples, model, beta=4)
    scores = acquisition_EI(Y, X_samples, model)
    # scores = acquisition(Y, X_samples, model)
    i = np.argmax(scores)
    return X_samples[i]


def plot(X, Y, model):
    # Plotting the observed data
    pyplot.scatter(X, Y, marker='.')

    # Plotting the true objective
    grid_i = np.array(np.linspace(0.0, 1.0, 10000))
    grid_o = np.array([objective(x, 0.0) for x in grid_i])
    pyplot.plot(grid_i, grid_o)

    # Plotting the uncertainty prediction of model.
    mean, std_dev = model.predict(grid_i.reshape(-1, 1), return_std=True)
    pyplot.plot(grid_i, mean)
    pyplot.fill_between(grid_i, mean - std_dev, mean + std_dev, alpha=0.45)

    pyplot.show()


maximizer, maxima = get_maxima(objective)
print("Actual values:")
print("Maximizer =", maximizer, ", maxima =", maxima)

# Fit a GP model with random samples.
# Reason for random samples
#     - In the actual problems we have access to only random samples and their objective evaluations
X = np.random.random(5)
Y = np.array([objective(x) for x in X])  # Noisy evaulations.
kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=500)
X = X.reshape((-1, 1))
model.fit(X, Y)
plot(X, Y, model)

#   Add incumbent data for plotting
incumbent = []
# Now the optimization cycle.
for _ in range(20):
    x = opt_aquisition(Y, X, model)
    y = objective(x)
    x = x.reshape((1, 1))

    X = np.append(X, x, axis=0)
    Y = np.append(Y, y, axis=0)

    model.fit(X, Y)
    # plot(X, Y, model)  # Uncomment this to see how the model evolves over time.

    incumbent += [np.max(Y)]

print("After optimization")
print("Maximizer =", X[np.argmax(Y), 0], ", Maxima =", np.max(Y))
plot(X, Y, model)

# Plotting the incumbent graph
pyplot.plot(np.array(range(1, len(incumbent)+1)), np.array(incumbent))
pyplot.show()
