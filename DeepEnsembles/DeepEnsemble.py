import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn


# Defining the criterion proposed in the paper instead of using MSE for regression task.
def regression_criterion(predicted, Y):
    mean, variance = predicted[:, 0], predicted[:, 1]
    mean = mean[:, None]
    variance = variance[:, None]
    return_vals = torch.log(variance) / 2 + torch.pow(Y - mean, 2) / (2 * variance)
    return torch.mean(return_vals)


# Implementing a single neural network that estimates uncertainty
# Uncertainty can be estimated using a mean and a variance.
class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        # Here fc is an abbreviation fully connected
        self.fc1 = nn.Linear(1, 15)  # Input dimension of the objective function = 1
        self.fc2 = nn.Linear(15, 25)
        self.fc3 = nn.Linear(25, 15)
        # For mean and variance the output dimension is 2.
        # First output is the mean, second output is the variance
        self.fc4 = nn.Linear(15, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        # Enforcing the positivity of the variance as mentioned in the paper.
        # Also adding 1e-6 for numerical stability
        x[:, 1] = torch.log(1 + torch.exp(x[:, 1])) + 1e-6

        return x

    # Generation of adversarial examples
    # Using eps as 1% as recommended in the paper
    def get_adversarial(self, X_batch, Y_batch, eps=0.01):
        # Setting requires grad to true for the input tensor
        X_batch.requires_grad = True
        Y_pred = self.forward(X_batch)
        loss = regression_criterion(Y_pred, Y_batch)
        loss.backward()
        X_batch_adv = X_batch + eps * torch.sign(X_batch.grad.data)

        return X_batch_adv

    # Training the estimator (with uncertainty) for the objective function
    def train(self, X, Y, optimizer, epochs=1000, batch_size=50):
        for _ in range(epochs):
            print("Epoch", _)
            X, Y = sklearn.utils.shuffle(X, Y)  # Randomly shuffle the data for each epoch
            i = 0
            while i < X.shape[0]:
                #  Sample a Mini-batch of data points based on batch size
                X_batch = X[i: i + batch_size]
                Y_batch = Y[i: i + batch_size]
                X_batch = torch.from_numpy(X_batch)
                Y_batch = torch.from_numpy(Y_batch)
                i = i + batch_size

                # Get the adversarial batch, append the adv to the X_batch
                X_batch_adv = self.get_adversarial(X_batch.clone().detach(), Y_batch.clone().detach())
                X_batch = torch.cat((X_batch, X_batch_adv))
                Y_batch = torch.cat((Y_batch, Y_batch))

                # Run the optmization procedure on the whole data
                optimizer.zero_grad()
                Y_pred = self.forward(X_batch)
                loss = regression_criterion(Y_pred, Y_batch)
                loss.backward()
                optimizer.step()

class DeepEnsemble():
    def __init__(self, M=5):
        self.M = M
        self.nn_list = []
        for _ in range(self.M):
            self.nn_list += [Estimator()]

    def train(self, X, Y):
        for nn in self.nn_list:
            optimizer = optim.Adam(nn.parameters(), lr=0.01)
            nn.train(X, Y, optimizer, batch_size=X.shape[0]//10)

    def predict(self, X):
        mean_list = []
        variance_list = []

        for nn in self.nn_list:
            Y_NN = nn(X)
            mean_NN = Y_NN[:, 0]  # Getting the mean out
            variance_NN = Y_NN[:, 1]
            mean_list += [mean_NN]
            variance_list += [variance_NN]

        # Returning the mean of the gaussian mixture
        sum = mean_list[0]
        for i in range(1, self.M):
            sum = sum + mean_list[i]
        mean = sum / self.M

        # Returning variance of the gaussian mixture
        sum = variance_list[0] + torch.pow(mean_list[0], 2)
        for i in range(1, self.M):
            sum = sum + variance_list[i] + torch.pow(mean_list[i], 2)
        variance = sum / self.M - torch.pow(mean, 2)

        return mean, variance
