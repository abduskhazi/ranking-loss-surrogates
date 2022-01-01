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


# Implementing a single neural network that estimates uncertainty
# Uncertainty can be estimated using a mean and a variance.
class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        # Here fc is an abbreviation fully connected
        self.fc1 = nn.Linear(1, 10)  # Input dimension of the objective function = 1
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 30)
        self.fc4 = nn.Linear(30, 20)
        self.fc5 = nn.Linear(20, 10)
        # For mean and variance the output dimension is 2.
        # First output is the mean, second output is the variance
        self.fc6 = nn.Linear(10, 2)

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

        # Enforcing the positivity of the variance as mentioned in the paper.
        # Also adding 1e-6 for numerical stability
        x[:, 1] = torch.log(1 + torch.exp(x[:, 1])) + 1e-6

        return x


maximizer, maxima = get_maxima(objective)
print("Actual values:")
print("Maximizer =", maximizer, ", maxima =", maxima)

# Reason for random samples
#     - In the actual problems we have access to only random samples and their objective evaluations
X = np.array(np.random.random(1000), dtype=np.float32)
Y = np.array([objective(x) for x in X], dtype=np.float32)  # Noisy evaluations.
X_copy = np.copy(X)
Y_copy = np.copy(Y)

estim = Estimator()
optimizer = optim.Adam(estim.parameters(), lr=0.01)
# Defining the criterion proposed in the paper instead of using MSE for regression task.
def criterion(predicted, Y):
    mean, variance = predicted[:, 0], predicted[:, 1]
    mean = mean[:, None]
    variance = variance[:, None]
    vals = torch.log(variance) / 2 + torch.pow(Y - mean, 2) / (2 * variance)
    return torch.mean(vals)

X = X.reshape((-1, 1))
Y = Y.reshape((-1, 1))

# Training the estimator for the objective function for 100 epochs
for _ in range(1000):
    print("epoch", _)
    X, Y = sklearn.utils.shuffle(X, Y)  # Randomly shuffle the data for each epoch
    batch_size = 50
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

# Plotting the results
pyplot.scatter(X_copy, Y_copy, marker='.')
X = np.array(np.linspace(0.0, 1.0, 1000), dtype=np.float32)
Y = np.array([objective(x, 0.0) for x in X], dtype=np.float32)
pyplot.plot(X, Y)

Y_NN = estim(torch.from_numpy(X.reshape(-1, 1)))
mean_NN = Y_NN[:, 0].detach().numpy() # Getting the mean out
pyplot.plot(X, mean_NN)
pyplot.savefig("results.png")
pyplot.show()

# Implementing Deep Ensembles.
M = 5  # Number of Neural Networks
