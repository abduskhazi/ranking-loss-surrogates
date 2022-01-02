import torch
import torch.nn as nn
import torch.nn.functional as F

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