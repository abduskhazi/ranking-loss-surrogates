import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from DKT import DKT
import argparse

from HPO_B.hpob_handler import HPOBHandler

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


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


class FSBO:
    # Note: num_batches is referred to as b_n in the paper.
    def __init__(self, meta_data, ssid, latent_dim, batch_size, num_batches):
        print("Using FSBO as method...")
        self.meta_data = meta_data
        self.ssid = ssid
        self.input_dim = self.get_input_dim(meta_data)
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.params = self.get_params(ssid)

        # Finding y_min and y_max for creating a scale invariant model
        self.y_min, self.y_max = get_min_max(self.meta_data)

        # backbone for latent feature calculation.
        backbone = NN(input_dim=self.input_dim, output_dim=self.latent_dim)
        self.dkt = DKT(backbone, batch_size=batch_size).to(device)
        self.optimizer = torch.optim.Adam([{'params': self.dkt.model.parameters(), 'lr': 0.001},
                                           {'params': self.dkt.feature_extractor.parameters(), 'lr': 0.001}])

    def train(self):
        # for reference check https://arxiv.org/pdf/2101.07667.pdf algorithm 1
        # Running the outer loop a 100 times as this is not specified exactly in the paper.
        for epoch in range(1000):  # params.stop_epoch
            # Sample a task and its data at random
            data_task_id = np.random.choice(list(self.meta_data.keys()))
            data = self.meta_data[data_task_id]
            # Calculate l and u for random scaling such that l < u
            # Issue: l can be very close to u and create issues.
            l = np.random.uniform(low=self.y_min, high=self.y_max)
            u = np.random.uniform(low=l, high=self.y_max)
            # Run the model training loop for set number of times.
            self.dkt.train_loop(epoch, self.optimizer,
                                data, l, u, b_n=self.num_batches,
                                batch_size=self.batch_size)

        self.dkt.save_checkpoint(self.params.checkpoint_dir)

    def finetune(self):
        self.dkt.load_checkpoint(self.params.checkpoint_dir)

        for epoch in range(70):
            # Sample a task and its data at random
            data_task_id = np.random.choice(list(self.meta_data.keys()))
            data = self.meta_data[data_task_id]
            # Run the model training loop for a smaller number of times for finetuning.
            # Do not use scaling when doing fine tuning.
            self.dkt.train_loop(epoch, self.optimizer, data, self.y_min,
                                self.y_max, b_n=50,
                                batch_size=self.batch_size,
                                scaling=False)

    def get_input_dim(self, meta_data):
        dataset_key = list(meta_data.keys())[0]
        dim = np.array(meta_data[dataset_key]["X"]).shape[1]
        return dim

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

if __name__ == '__main__':
    # mp.freeze_support()
    # Pretrain hpob with a single search space (hardcoded for now)
    search_space_id = '4796'

    # Pretrain DKT
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3")
    meta_train_data = hpob_hdlr.meta_train_data[search_space_id]
    fsbo = FSBO(meta_train_data, search_space_id, latent_dim=10, batch_size=70, num_batches=50)
    fsbo.train()

    # Running the Fine tuning loop...
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
    meta_test_data = hpob_hdlr.meta_test_data[search_space_id]
    fsbo = FSBO(meta_test_data, search_space_id, latent_dim=10, batch_size=70, num_batches=50)
    fsbo.finetune()
