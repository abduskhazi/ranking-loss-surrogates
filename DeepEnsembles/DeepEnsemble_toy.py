# This is the first step to implement Deep Emsemble surrogate for a toy example.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot
import sklearn

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


# Implementing a single neural network
class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        # Here fc is an abbreviation fully connected
        self.fc1 = nn.Linear(1, 10)  # Input dimension of the objective function = 1
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 30)
        self.fc4 = nn.Linear(30, 20)
        self.fc5 = nn.Linear(20, 10)
        self.fc6 = nn.Linear(10, 1)  # Output dimension of the objective function = 1

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)

        x = self.fc6(x)

        return x


maximizer, maxima = get_maxima(objective)
print("Actual values:")
print("Maximizer =", maximizer, ", maxima =", maxima)

# Reason for random samples
#     - In the actual problems we have access to only random samples and their objective evaluations
X = np.array(np.random.random(500), dtype=np.float32)
Y = np.array([objective(x) for x in X], dtype=np.float32)  # Noisy evaluations.

# Plotting for debugging
pyplot.scatter(X, Y, marker='.')
pyplot.savefig("actual.png")
pyplot.show()

estim = Estimator()
optimizer = optim.Adam(estim.parameters(), lr=0.01)
criterion = nn.MSELoss()

X = X.reshape((-1, 1))
Y = Y.reshape((-1, 1))

# Training the estimator for the objective function for 100 epochs
for _ in range(500):
    print("epoch", _)
    X, Y = sklearn.utils.shuffle(X, Y)  # Randomly shuffle the data for each epoch
    batch_size = 25
    i = 0
    while i < X.shape[0]:
        optimizer.zero_grad()
        #  Sample a Mini-batch of 50 points
        X_batch = X[i: i + batch_size]
        Y_batch = Y[i: i + batch_size]
        X_batch = torch.from_numpy(X_batch)
        Y_batch = torch.from_numpy(Y_batch)
        i = i + batch_size
        #  Calculate the output
        Y_pred = estim(X_batch)
        #  Calculate the MSE loss
        loss = criterion(Y_pred, Y_batch)
        #  back propagate the loss
        loss.backward()
        #  Update the parameters using optimizer.
        optimizer.step()

X = np.array(np.random.random(500), dtype=np.float32)
Y = estim(torch.from_numpy(X.reshape(-1, 1)))

# Plotting the values obtained from the estimator
pyplot.scatter(X, Y.detach().numpy(), marker='.')
pyplot.savefig("estimator.png")
pyplot.show()

# Implementing Deep Ensembles.
M = 5  # Number of Neural Networks
