import math
import numpy as np
from matplotlib import pyplot

rng = np.random.default_rng()

# Objective function
def objective(x, noise=0.1):
    y = x**2 * math.sin(5 * math.pi * x)**6
    y_noise = rng.normal(loc=0, scale=noise)
    return y + y_noise

X = np.linspace(0.0, 1.0, 400)
Y = [objective(x, 0.0) for x in X]
Y_noise = [objective(x) for x in X]

# Finding the true maximizer and maxima
i = np.argmax(Y)
maximizer = X[i]
maxima = Y[i]

print("Maximizer =", maximizer, ", maxima =", maxima)

# Plot with noise
pyplot.scatter(X,Y_noise, marker='.')
# Plot without noise
pyplot.plot(X,Y)

pyplot.show()
