# Global imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
import multiprocessing as mp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

# Local repo imports parent folder path
import sys
sys.path.append('../../../ranking-loss-surrogates')
from HPO_B.hpob_handler import HPOBHandler
from study_hpo import get_all_combinations, store_object, evaluate_combinations
from fsbo import convert_meta_data_to_np_dictionary, get_input_dim


# =======================================================================================
# Quick Configuration
non_transfer = False
transfer = True
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

def get_batch_HPBO_single(meta_data, batch_size):
    query_X = []
    query_y = []
    for i in range(1):
        data = meta_data[np.random.choice(list(meta_data.keys()))]
        X = data["X"]
        y = data["y"]
        idx = np.random.choice(X.shape[0], size=batch_size, replace=True)
        query_X += [torch.from_numpy(X[idx])]
        query_y += [torch.from_numpy(y[idx].flatten())]

    # Viewing everything as 2D
    query_X = torch.stack(query_X)
    query_X = query_X.view(-1, query_X.shape[-1])
    query_y = torch.stack(query_y)
    query_y = query_y.flatten()

    return query_X, query_y

def get_batch_HPBO_all(meta_data, batch_size):
    query_X = []
    query_y = []
    for k in list(meta_data.keys()):
        data = meta_data[k]
        X = data["X"]
        y = data["y"]
        idx = np.random.choice(X.shape[0], size=batch_size, replace=True)
        query_X += [torch.from_numpy(X[idx])]
        query_y += [torch.from_numpy(y[idx].flatten())]

    # Viewing everything as 2D
    query_X = torch.stack(query_X)
    query_X = query_X.view(-1, query_X.shape[-1])
    query_y = torch.stack(query_y)
    query_y = query_y.flatten()

    return query_X, query_y

# Implementing a single neural network that estimates uncertainty
# Uncertainty can be estimated using a mean and a variance.
class Estimator(nn.Module):
    def __init__(self, input_dim=1):
        super(Estimator, self).__init__()
        # Here fc is an abbreviation fully connected
        self.fc1 = nn.Linear(input_dim, 32)  # Input dimension of the objective function
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x

    def generate_meta_loss(self, prediction, y_true):
        prediction = prediction[None, :]
        y_true = y_true[None, :]
        loss = regression_criterion(prediction, y_true)
        return loss

    def meta_train(self, meta_train_data, meta_val_data, epochs, lr, batch_size):
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}, ])
        loss_list = []
        val_loss_list = []
        for _ in range(epochs):
            self.train()

            for __ in range(100):   # Doing the grad step for batch size number of steps.
                optimizer.zero_grad()
                train_X, train_y = get_batch_HPBO_single(meta_train_data, batch_size)

                prediction = self.forward(train_X).flatten()
                loss = self.generate_meta_loss(prediction, train_y)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.eval()
                # Calculating training loss
                train_X, train_y = get_batch_HPBO_all(meta_train_data, batch_size)
                prediction = self.forward(train_X).flatten()
                loss = self.generate_meta_loss(prediction, train_y)

                # Calculating validation loss
                val_X, val_y = get_batch_HPBO_all(meta_val_data, batch_size)
                pred_val = self.forward(val_X).flatten()
                val_loss = self.generate_meta_loss(pred_val, val_y)

            loss_list += [loss.item()]
            val_loss_list += [val_loss.item()]
            #  Removing this as the output file will be too long.
            #  if (_+1) % 250 == 0:
            #      print("Epoch[", _, "] ==> Loss =", loss.item(), "; Val_loss =", val_loss.item())

        return loss_list, val_loss_list

    # Training the estimator (with uncertainty) for the objective function
    def train_finetune(self, X, Y, search_space_range=1, epochs=1000, lr=0.001, batch_size=100):
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
    nn.train_finetune(X, Y, search_space_range, epochs=epochs, lr=lr, batch_size=batch_size)

class DeepEnsembleRegression(nn.Module):
    def __init__(self, input_dim=1, M=10, parallel_training=False):
        super(DeepEnsembleRegression, self).__init__()
        self.parallel_training = parallel_training
        self.M = M
        self.nn_list = []
        for _ in range(self.M):
            # Using the default weight initialization in pytorch according to the paper.
            self.nn_list += [Estimator(input_dim)]
        # To make it easier to load and store results
        self.nn_list = nn.ModuleList(self.nn_list)

    def meta_train(self, meta_train_data, meta_val_data, epochs=1000, lr=0.001, batch_size=100):
        loss_list = []
        val_loss_list = []
        for nn in self.nn_list:
            l, vl = nn.meta_train(meta_train_data, meta_val_data, epochs, lr, batch_size)
            loss_list += [l]
            val_loss_list += [vl]

        loss_list = np.array(loss_list, dtype=np.float32)
        val_loss_list = np.array(val_loss_list, dtype=np.float32)
        loss_list = np.mean(loss_list, axis=0).tolist()
        val_loss_list = np.mean(val_loss_list, axis=0).tolist()

        return loss_list, val_loss_list

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

class DE_regression_transfer:
    def __init__(self, input_dim, ssid, load=False):
        print("Using Deep Ensembles as method...")
        self.input_dim = input_dim
        self.ssid = ssid
        self.M = 10

        self.save_folder = "./" + str(sys.argv[2]) + "/results/";
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        if load:
            self.load()
        else:
            self.DE = DeepEnsembleRegression(M=self.M, input_dim=input_dim, parallel_training=False)

    def save(self):
        file_name = self.save_folder + self.ssid
        torch.save({"input_dim": self.input_dim,
                    "ssid": self.ssid,
                    "M": self.M,
                    "DE": self.DE.state_dict(),
                    "save_folder": self.save_folder},
                   file_name)

    def load(self):
        file_name = self.save_folder + self.ssid
        dict = torch.load(file_name)
        self.input_dim = dict["input_dim"]
        self.ssid = dict["ssid"]
        self.M = dict["M"]
        self.save_folder = dict["save_folder"]

        # Creating and initializing the DE
        self.DE = DeepEnsembleRegression(M=self.M, input_dim=self.input_dim, parallel_training=False)
        self.DE.load_state_dict(dict["DE"])


    # Assuming that we are dealing with only discrete case ==> X_pen is None.
    # First fitting the model based on observations.
    # Predicting then the pending configuration for evaluation.
    # Returning the index that gives us the best results.
    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        # Doing restarts
        self.load()
        X_obs = np.array(X_obs, dtype=np.float32)
        y_obs = np.array(y_obs, dtype=np.float32)
        X_pen = np.array(X_pen, dtype=np.float32)
        self.DE.train(X_obs, y_obs, epochs=1000, lr=0.001, batch_size=200)

        X_pen = torch.from_numpy(X_pen)
        scores = acquisition_EI(y_obs, X_pen, self.DE)
        idx = np.argmax(scores)
        return idx

def evaluate_DE_regression(hpob_hdlr, keys_to_evaluate):
    performance = []
    for key in keys_to_evaluate:
        search_space, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space, dataset)
        method = DE_regression_transfer(input_dim=input_dim, ssid=search_space, load=True)
        res = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=[key])
        performance += res

    return performance

def DE(i, run):
    hpob_hdlr = HPOBHandler(root_dir="../../HPO_B/hpob-data/", mode="v3-test")
    de_keys = get_all_combinations(hpob_hdlr, 100)
    print("Evaluating", i, "of ", len(de_keys))
    de_keys = de_keys[i:i+1]  # Only executing the required keys.
    de_performance = evaluate_DE_regression(hpob_hdlr, keys_to_evaluate=de_keys)
    store_object(de_performance, "./" + str(run) + "/results/DE_REG_T_32x32x10_E1000_l0_02_OPT" + str(i))

def meta_train_on_HPOB(i):
    hpob_hdlr = HPOBHandler(root_dir="../../HPO_B/hpob-data/", mode="v3")

    # Pretrain Ranking loss surrogate with a single search spaces i
    for search_space_id in hpob_hdlr.get_search_spaces()[i:i+1]:
        meta_train_data = hpob_hdlr.meta_train_data[search_space_id]
        meta_val_data = hpob_hdlr.meta_validation_data[search_space_id]

        input_dim = get_input_dim(meta_train_data)
        print("Input dim of", search_space_id, "=", input_dim)

        meta_train_data = convert_meta_data_to_np_dictionary(meta_train_data)
        meta_val_data = convert_meta_data_to_np_dictionary(meta_val_data)

        epochs = 5000
        batch_size = 100
        de_regression = DE_regression_transfer(input_dim=input_dim, ssid=search_space_id)
        loss_list, val_loss_list = \
            de_regression.DE.meta_train(meta_train_data, meta_val_data, epochs, 0.001, batch_size)

        de_regression.save()
        de_regression.load()  # just for testing the functionality

        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        plt.plot(np.array(val_loss_list, dtype=np.float32))
        legend = ["Loss",
                  "Validation Loss"
                  ]
        plt.legend(legend)
        plt.title("SSID: " + search_space_id + "; Input dim: " + str(input_dim))
        plt.savefig(de_regression.save_folder + "loss_" + search_space_id + ".png")

if __name__ == '__main__':
    i = int(sys.argv[1])
    run = int(sys.argv[2])

    if non_transfer:
        print("Non Transfer: Evaluating DE with regression loss");
        non_transfer_DE(i, run)

    if transfer:
        print("HPO Transfer: Evaluating DE with regression loss");
        if sys.argv[3] == "train":
            print("Meta training", i)
            meta_train_on_HPOB(i)
        elif sys.argv[3] == "evaluate":
            print("Evaluating", i)
            DE(i, run)
        else:
            print("Unknown option specified")