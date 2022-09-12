# Global imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
import multiprocessing as mp
import numpy as np
from scipy.stats import norm

# Local repo imports parent folder path
import sys
sys.path.append('../../../ranking-loss-surrogates')
from HPO_B.hpob_handler import HPOBHandler
from study_hpo import get_all_combinations, store_object, evaluate_combinations

# =======================================================================================
# Quick Configuration
non_transfer = True
transfer = False
# =======================================================================================

def acquisition_EI(Y, X_samples, surrogate_model):
    # https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec
    # Find the best value of the objective function so far according to data.
    # Is this according to the gaussian fit or according to the actual values.???
    # For now using the best according to the actual values.
    best_y = np.max(Y)
    # Calculate the predicted mean & variance values of all the required samples
    mean, variance = surrogate_model.predict(X_samples)
    mean = mean.detach().numpy()
    std_dev = torch.sqrt(variance).detach().numpy()
    z = (mean - best_y) / (std_dev + 1E-9)
    return (mean - best_y) * norm.cdf(z) + (std_dev + 1E-9) * norm.pdf(z)

# Defining the criterion proposed in the paper instead of using MSE for regression task.
def regression_criterion(predicted, Y):
    errors = (Y - predicted)
    squared_errors = errors ** 2
    mean_squared_errors = torch.sum(squared_errors, dim=1)
    rmses = torch.sqrt(mean_squared_errors)
    return torch.mean(rmses)

# Implementing a single neural network that estimates uncertainty
# Uncertainty can be estimated using a mean and a variance.
class Estimator(nn.Module):
    def __init__(self, input_dim=1):
        super(Estimator, self).__init__()
        # Here fc is an abbreviation fully connected
        self.fc1 = nn.Linear(input_dim, 32)  # Input dimension of the objective function
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 1)

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

        return x

    # Training the estimator (with uncertainty) for the objective function
    def train(self, X, Y, search_space_range=1, epochs=1000, lr=0.001, batch_size=100):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            X, Y = sklearn.utils.shuffle(X, Y)  # Randomly shuffle the data for each epoch
            i = 0
            while i < X.shape[0]:
                #  Sample a Mini-batch of data points based on batch size
                X_batch = X[i: i + batch_size]
                Y_batch = Y[i: i + batch_size]
                X_batch = torch.from_numpy(X_batch)
                Y_batch = torch.from_numpy(Y_batch)
                i = i + batch_size

                # Run the optimization procedure on the whole data
                optimizer.zero_grad()
                Y_pred = self.forward(X_batch)
                loss = regression_criterion(Y_pred, Y_batch)
                loss.backward()
                optimizer.step()

def nn_train(args):
    nn, X, Y, search_space_range, epochs, lr, batch_size = args
    nn.train(X, Y, search_space_range, epochs=epochs, lr=lr, batch_size=batch_size)

class DeepEnsembleRegression():
    def __init__(self, input_dim=1, M=10, parallel_training=False):
        self.parallel_training = parallel_training
        self.M = M
        self.nn_list = []
        for _ in range(self.M):
            # Using the default weight initialization in pytorch according to the paper.
            self.nn_list += [Estimator(input_dim)]

    def train(self, X, Y, search_space_range=1, epochs=1000, lr=0.001, batch_size=100):
        if self.parallel_training:
            with mp.Pool(len(self.nn_list)) as p:
                p.map(nn_train, [(nn, X, Y, search_space_range, epochs, lr, batch_size) for nn in self.nn_list])
        else:
            i = 1
            for nn in self.nn_list:
               nn_train((nn, X, Y, search_space_range, epochs, lr, batch_size))
               i += 1

    def predict(self, X):
        predictions = []

        for nn in self.nn_list:
            Y_NN = nn(X)
            predictions += [Y_NN]

        predictions = torch.stack(predictions)
        mean = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0, keepdim=True)[0]

        return mean, variance

class DE_regression:

    def __init__(self, input_dim):
        print("Using Deep Ensembles as method...")
        self.DE = DeepEnsembleRegression(M=10, input_dim=input_dim, parallel_training=False)
        self.input_dim = input_dim

    # Assuming that we are dealing with only discrete case ==> X_pen is None.
    # First fitting the model based on observations.
    # Predicting then the pending configuration for evaluation.
    # Returning the index that gives us the best results.
    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        # Doing random starts like GPs
        self.DE = DeepEnsembleRegression(M=10, input_dim=self.input_dim, parallel_training=False)
        X_obs = np.array(X_obs, dtype=np.float32)
        y_obs = np.array(y_obs, dtype=np.float32)
        X_pen = np.array(X_pen, dtype=np.float32)
        self.DE.train(X_obs, y_obs, epochs=1000, lr=0.02, batch_size=200)

        X_pen = torch.from_numpy(X_pen)
        scores = acquisition_EI(y_obs, X_pen, self.DE)
        idx = np.argmax(scores)
        return idx

def evaluate_DE_regression(hpob_hdlr, keys_to_evaluate):
    performance = []
    for key in keys_to_evaluate:
        search_space, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space, dataset)
        method = DE_regression(input_dim=input_dim)
        res = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=[key])
        performance += res

    return performance

def non_transfer_DE(i, run):
    hpob_hdlr = HPOBHandler(root_dir="../../HPO_B/hpob-data/", mode="v3-test")
    de_keys = get_all_combinations(hpob_hdlr, 100)
    print("Evaluating", i, "of ", len(de_keys))
    de_keys = de_keys[i:i+1]  # Only executing the required keys.
    de_performance = evaluate_DE_regression(hpob_hdlr, keys_to_evaluate=de_keys)
    store_object(de_performance, "./" + str(run) + "/results/DE_REG_OPT" + str(i))


if __name__ == '__main__':
    i = int(sys.argv[1])
    run = int(sys.argv[2])

    if non_transfer:
        print("Non Transfer: Evaluating DE with regression loss");
        non_transfer_DE(i, run)

    if transfer:
        print("HPO Transfer: Evaluating DE with regression loss (Not yet implemented)");
