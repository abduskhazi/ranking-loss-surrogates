import math
import numpy as np
from matplotlib import pyplot
from sklearn.gaussian_process import GaussianProcessRegressor

rng = np.random.default_rng()

# Objective function
def objective(x, noise=0.1):
    y = x**2 * math.sin(5 * math.pi * x)**6
    y_noise = rng.normal(loc=0, scale=noise)
    return y + y_noise

# Returns the approx maximizer and the maxima of the objective function
# This is only for testing. It does not exist in the real world.
def get_maxima(obj_func):
    X = np.linspace(0.0, 1.0, 3000)
    Y = [objective(x, 0.0) for x in X]
    i = np.argmax(Y)
    maximizer = X[i]
    maxima = Y[i]
    return maximizer, maxima

# Can be used to ignore warnings as well.
def surrogate(model, X):
    yhat_mean, yhat_std = model.predict(X, return_std=True)
    return yhat_mean, yhat_std

def plot(X, Y, model):
    pyplot.scatter(X, Y, marker='.')

    grid_i = np.linspace(0.0, 1.0, 1000)
    grid_o = model.predict(grid_i.reshape(-1,1))
    pyplot.plot(grid_i, grid_o)

    pyplot.show()

maximizer, maxima = get_maxima(objective)
print("Maximizer =", maximizer, ", maxima =", maxima)

# Taking random samples from the domain.
# Reason - In the actual problems we have access to only random samples and their objective evaluations
X = np.random.random(100)
Y = np.array([objective(x) for x in X]) # Noisy evaulations.

model = GaussianProcessRegressor()
X = X.reshape((-1, 1))
model.fit(X,Y)
plot(X, Y, model)
