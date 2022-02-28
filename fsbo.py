import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from DKT import DKT
import argparse
from scipy.stats import norm

from HPO_B.hpob_handler import HPOBHandler

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def acquisition_EI(X, y, X_query, surrogate_model):
    # https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec
    # Find the best value of the objective function so far according to data.
    # Is this according to the gaussian fit or according to the actual values.???
    # For now using the best according to the actual values.
    best_y = np.max(y.detach().cpu().numpy())
    # Calculate the predicted mean & variance values of all the required samples
    prediction = surrogate_model.predict(X, y, X_query)
    mean, variance = prediction.mean, prediction.variance
    mean = mean.detach().cpu().numpy()
    std_dev = torch.sqrt(variance).detach().cpu().numpy()
    z = (mean - best_y) / (std_dev + 1E-9)
    return (mean - best_y) * norm.cdf(z) + (std_dev + 1E-9) * norm.pdf(z)


class NN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Here fc is an abbreviation fully connected
        self.fc1 = nn.Linear(input_dim, 64)  # Input dimension of the objective function
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x


def parse_args_regression(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--seed', default=0, type=int, help='Seed for Numpy and pyTorch. Default: 0 (None)')
    parser.add_argument('--model', default='DNN', help='model: DNN')
    parser.add_argument('--method', default='DKT', help='DKT')
    parser.add_argument('--dataset', default='HPOB', help='HPOB meta data set')
    parser.add_argument('--spectral', action='store_true', help='Use a spectral covariance kernel function')

    if script == 'train_regression':
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=100, type=int,
                            help='Stopping epoch')  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume', action='store_true',
                            help='continue from previous trained model with largest epoch')
    elif script == 'test_regression':
        parser.add_argument('--n_support', default=5, type=int,
                            help='Number of points on trajectory to be given as support points')
        parser.add_argument('--n_test_epochs', default=10, type=int, help='How many test people?')
    return parser.parse_args()


def get_min_max(data_dict):
    labels = np.array([], dtype=np.float32)
    for d in data_dict.values():
        vals = np.array(d["y"], dtype=np.float32)
        labels = np.append(labels, vals)
    return np.min(labels), np.max(labels)


def get_input_dim(meta_data):
    dataset_key = list(meta_data.keys())[0]
    dim = np.array(meta_data[dataset_key]["X"]).shape[1]
    return dim

class FSBO:
    # Note: num_batches is referred to as b_n in the paper.
    def __init__(self, ssid, input_dim, latent_dim, batch_size, num_batches):
        # print("Using FSBO as method...") # Not printing this for now.
        self.ssid = ssid
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.params = self.get_params(ssid)

        # backbone for latent feature calculation.
        backbone = NN(input_dim=self.input_dim, output_dim=self.latent_dim)
        self.dkt = DKT(backbone, batch_size=batch_size).to(device)
        self.optimizer = torch.optim.Adam([{'params': self.dkt.model.parameters(), 'lr': 0.0001}, # 0.001 used for long name HPOB simulation
                                           {'params': self.dkt.feature_extractor.parameters(), 'lr': 0.0001}])

    def get_val_data(self, val_data):
        batch = []
        batch_labels = []

        for i in val_data.keys():
            X = np.array(val_data[i]["X"], dtype=np.float32)
            y = np.array(val_data[i]["y"], dtype=np.float32)
            batch += [torch.from_numpy(X)]
            batch_labels += [torch.from_numpy(y).flatten()]

        batch, batch_labels = torch.cat(batch, 0), torch.cat(batch_labels, 0)

        if(batch_labels.shape[0] > 2000):
            idx = np.random.choice(np.arange(len(X)), 1000)
            batch, batch_labels = batch[idx], batch_labels[idx]

        return (batch, batch_labels)

    def train(self, meta_data, meta_val_data):
        loss_list = []
        val_loss_list = []
        meta_val_data = self.get_val_data(meta_val_data)
        # for reference check https://arxiv.org/pdf/2101.07667.pdf algorithm 1
        # Running the outer loop a 100 times as this is not specified exactly in the paper.
        # Finding y_min and y_max for creating a scale invariant model
        y_min, y_max = get_min_max(meta_data)
        for epoch in range(1000):  # params.stop_epoch
            # Sample a task and its data at random
            data_task_id = np.random.choice(list(meta_data.keys()))
            data = meta_data[data_task_id]
            # Calculate l and u for random scaling such that l < u
            # Issue: l can be very close to u and create issues.
            l = np.random.uniform(low=y_min, high=y_max)
            u = np.random.uniform(low=l, high=y_max)
            # Run the model training loop for set number of times.
            self.dkt.train_loop(epoch, self.optimizer,
                                data, l, u, b_n=self.num_batches,
                                batch_size=self.batch_size)

        self.dkt.save_checkpoint(self.params.checkpoint_dir)

    # Fine tuning is done for a few points in the data set, to make it
    # a little bit more specific to this.
    def finetune(self, X, y):
        self.dkt.load_checkpoint(self.params.checkpoint_dir)

        for epoch in range(200):
            # Run the model training loop for a smaller number of times for finetuning.
            # Do not use scaling when doing fine tuning.
            self.dkt.fine_tune_loop(epoch, self.optimizer, X, y)

    def get_params(self, ssid):
        params = parse_args_regression('train_regression')
        # optional seeding property
        # np.random.seed(params.seed)
        # torch.manual_seed(params.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # Creating folder and filename to store the trained model.
        save_dir = './save/'
        params.checkpoint_dir = '%scheckpoints/%s/' % (save_dir, params.dataset)
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (
            save_dir, params.dataset, params.method, params.model, ssid)

        return params

    def predict(self, x_support, y_support, x_query):
        return self.dkt.predict(x_support, y_support, x_query)

    # Assuming that we are dealing with only discrete case ==> X_pen is None.
    # First fine tuning the model based on observations.
    # Predicting then the pending configuration for evaluation.
    # Returning the index that gives us the best results.
    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        X_obs = np.array(X_obs, dtype=np.float32)
        y_obs = np.array(y_obs, dtype=np.float32)
        X_pen = np.array(X_pen, dtype=np.float32)
        X_obs = torch.from_numpy(X_obs)
        y_obs = torch.from_numpy(y_obs).flatten()
        X_pen = torch.from_numpy(X_pen)

        # Doing restarts from the checkpoint
        fsbo = FSBO(ssid=self.ssid,
                    input_dim=self.input_dim,
                    latent_dim=self.latent_dim,
                    batch_size=self.batch_size,
                    num_batches=self.num_batches)
        fsbo.finetune(X_obs, y_obs)
        scores = acquisition_EI(X_obs, y_obs, X_pen, fsbo)

        # If we want to reuse self object
        # self.finetune(X_obs, y_obs)
        # scores = acquisition_EI(X_obs, y_obs, X_pen, self)

        idx = np.argmax(scores)
        return idx

if __name__ == '__main__':
    # mp.freeze_support()
    # Pretrain hpob with a single search space (hardcoded for now)
    search_space_id = '4796'

    # Pretrain DKT
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3")
    meta_train_data = hpob_hdlr.meta_train_data[search_space_id]
    fsbo = FSBO(search_space_id, input_dim=get_input_dim(meta_train_data),
                latent_dim=10, batch_size=70, num_batches=50)
    fsbo.train(meta_train_data)

    # Running the Fine tuning loop...
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
    meta_test_data = hpob_hdlr.meta_test_data[search_space_id]
    fsbo = FSBO(search_space_id, input_dim=get_input_dim(meta_test_data),
                latent_dim=10, batch_size=70, num_batches=50)
    for d_id in meta_test_data:
        X = meta_test_data[d_id]["X"]
        y = meta_test_data[d_id]["y"]
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y).flatten()
        fsbo.finetune(X, y)
