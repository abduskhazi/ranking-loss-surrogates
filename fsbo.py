import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from DKT import DKT
import argparse

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
    parser.add_argument('--model', default='Conv3', help='model: Conv{3} / MLP{2}')
    parser.add_argument('--method', default='DKT', help='DKT / transfer')
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

# Training the fsbo for a defined search space.
def train_fsbo(meta_data):
    params = parse_args_regression('train_regression')
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_dir = './save/'
    params.checkpoint_dir = '%scheckpoints/%s/' % (save_dir, params.dataset)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (save_dir, params.dataset, params.model, params.method)

    dataset_key = list(meta_data.keys())[0]
    dim = np.array(meta_data[dataset_key]["X"]).shape[1]

    bb = NN(input_dim=dim, output_dim=10)  # backbone.Conv3().to(device) # this has be different than conv3d
    batch_size = 70
    model = DKT(bb, batch_size=batch_size).to(device)

    optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                                  {'params': model.feature_extractor.parameters(), 'lr': 0.001}])

    # for reference check https://arxiv.org/pdf/2101.07667.pdf algorithm 1
    # First finding y_min and y_max for creating a scale invariant model
    y_min, y_max = get_min_max(meta_data)
    # Running the outer loop a 100 times as this is not specified exactly in the paper.
    for epoch in range(1000):  # params.stop_epoch
        # Sample a task and its data at random
        data_task_id = np.random.choice(list(meta_data.keys()))
        data = meta_data[data_task_id]
        # Calculate l and u for random scaling such that l < u
        l = np.random.uniform(low=y_min, high=y_max)
        u = np.random.uniform(low=l, high=y_max)
        # Run the model training loop for set number of times.
        model.train_loop(epoch, optimizer, data, l, u, b_n=50, batch_size=batch_size)

    model.save_checkpoint(params.checkpoint_dir)

if __name__ == '__main__':
    #mp.freeze_support()
    train_fsbo()

