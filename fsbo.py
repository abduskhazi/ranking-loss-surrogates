import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from DKT import DKT
import argparse
from scipy.stats import norm
import matplotlib.pyplot as plt
# For memory management
import gc
import sys
from collections import namedtuple
import pickle
import time
import gpytorch

from HPO_B.hpob_handler import HPOBHandler
# from study_hpo import get_all_combinations, store_object, evaluate_combinations

def store_object(obj, obj_name):
    with open(obj_name, "wb") as fp:
        pickle.dump(obj, fp)

def load_object(obj_name):
    with open(obj_name, "rb") as fp:
        return pickle.load(fp)

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
    def __init__(self, input_dim=1, output_dim=1, n_hidden=2):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Here fc is an abbreviation fully connected
        self.fc1 = nn.Linear(input_dim, 32)  # Input dimension of the objective function
        self.hidden_layers = []
        for i in range(n_hidden):
            self.hidden_layers += [nn.Linear(32, 32)]
            # Not using a module list here i think is the problem of not getting good results.
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.fc_last = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        for fc in self.hidden_layers:
            x = fc(x)
            x = F.relu(x)

        x = self.fc_last(x)
        # The last layer must not be passed through relu

        return x


def parse_args_regression(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--seed', default=0, type=int, help='Seed for Numpy and pyTorch. Default: 0 (None)')
    parser.add_argument('--model', default='DNN', help='model: DNN')
    parser.add_argument('--method', default='DKT', help='DKT')
    parser.add_argument('--dataset', default='all_HPOB_32x4_matern_multistep_epoch', help='HPOB meta data set')
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

def convert_meta_data_to_np_dictionary(meta_data):
    temp_meta_data = {}
    for k in meta_data.keys():
        X = np.array(meta_data[k]["X"], dtype=np.float32)
        y = np.array(meta_data[k]["y"], dtype=np.float32)
        temp_meta_data[k] = {"X": X, "y": y}
        # temp_meta_data[k] = {"X": torch.from_numpy(X), "y": torch.from_numpy(y)} ... ?

    return temp_meta_data

def add_query_points(meta_val_data, n_support_points):
    n_query_points = 3000

    for k in meta_val_data.keys():
        X = meta_val_data[k]["X"]
        idx_support = np.random.choice(X.shape[0], size=n_support_points, replace=False)
        meta_val_data[k]["support"] = idx_support

        if n_query_points > X.shape[0] - n_support_points:
            n_query_points = X.shape[0] - n_support_points
        query_choice = np.delete(np.arange(X.shape[0]), idx_support)
        idx_query = np.random.choice(query_choice, size=n_query_points, replace=False)
        meta_val_data[k]["query"] = idx_query

    return meta_val_data

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
        # Total layers = n_hidden + 1 (input layer) + 1 (output layer)
        backbone = NN(input_dim=self.input_dim, output_dim=self.latent_dim, n_hidden=2)
        self.dkt = DKT(backbone, batch_size=batch_size).to(device)

    def train(self, meta_data, meta_val_data):
        loss_list = []
        val_loss_list = []
        # for reference check https://arxiv.org/pdf/2101.07667.pdf algorithm 1
        # Running the outer loop a 100 times as this is not specified exactly in the paper.
        # Finding y_min and y_max for creating a scale invariant model
        y_min, y_max = get_min_max(meta_data)
        epochs = 5000

        meta_data = convert_meta_data_to_np_dictionary(meta_data)
        meta_val_data = convert_meta_data_to_np_dictionary(meta_val_data)
        meta_val_data = add_query_points(meta_val_data, 20)

        div_count = 0
        for epoch in range(epochs):  # params.stop_epoch
            # Calculate l and u for random scaling such that l < u
            # Issue: l can be very close to u and create issues.
            l = np.random.uniform(low=y_min, high=y_max)
            u = np.random.uniform(low=l, high=y_max)
            # Run the model training loop for set number of times.
            loss, val_loss = self.dkt.train_loop(epoch, None, # giving optimiser as none.
                                meta_data, meta_val_data, l, u, b_n=self.num_batches,
                                batch_size=self.batch_size, scaling=False)

            # Save the best model.
            if val_loss < min(val_loss_list + [np.inf]):
                self.dkt.save_checkpoint(self.params.checkpoint_dir)

            # Compare how many times the current loss is higher than the lowest loss
            # Break if divergence occurs in the last 10 epochs
            if val_loss > min(val_loss_list + [np.inf]):
                div_count += 1
            else:
                div_count = 0
            if div_count > 10:  # Maybe 30? probably not a good idea
                # break
                no_breaking = True

            loss_list += [loss]
            val_loss_list += [val_loss]

        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        plt.plot(np.array(val_loss_list, dtype=np.float32))
        legend = ["Loss",
                  "Validation Loss"
                  ]
        plt.legend(legend)
        plt.title("SSID: " + self.ssid + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.params.checkpoint_dir + "_loss.png")

        return loss_list, val_loss_list

    # Fine tuning is done for a few points in the data set, to make it
    # a little bit more specific to this.
    def finetune(self, X, y):
        self.dkt.load_checkpoint(self.params.checkpoint_dir)
        epochs = 5000  # ....

        scheduler_fn = lambda x, y: torch.optim.lr_scheduler.CosineAnnealingLR(x, y, eta_min=1e-4)
        # 500 with 0.1 giving best results
        optimizer = torch.optim.Adam([{'params': self.dkt.model.parameters(), 'lr': 0.001},
                                      {'params': self.dkt.feature_extractor.parameters(), 'lr': 0.001}])
        scheduler = scheduler_fn(optimizer, epochs)

        loss_list = []
        for epoch in range(epochs):
            # Run the model training loop for a smaller number of times for finetuning.
            # Do not use scaling when doing fine tuning.
            loss = self.dkt.fine_tune_loop(epoch, optimizer, X, y)
            loss_list += [loss]
            scheduler.step()

        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        legend = ["Fine tune Loss"]
        plt.legend(legend)
        plt.title("SSID: " + self.ssid + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.params.checkpoint_dir + "_fine_tune_loss.png")
        plt.close()

        return loss_list

    def get_params(self, ssid):
        # params = parse_args_regression('train_regression')
        # optional seeding property
        # np.random.seed(params.seed)
        # torch.manual_seed(params.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        params = namedtuple("params", "checkpoint_dir")

        # Creating folder and filename to store the trained model.
        save_dir = './FSBO_save/'
        fsbo_config = "all_HPOB_32x4_matern_multistep_epoch"
        params.checkpoint_dir = '%scheckpoints/%s/' % (save_dir, fsbo_config)
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s' % (
            save_dir, fsbo_config, "DKT", "DNN", ssid)

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

# created as a stub for parallel evaluations.
def evaluation_worker(hpob_hdlr, method, args):
    search_space, dataset, seed, n_trials = args
    print(search_space, dataset, seed, n_trials)
    res = []
    try:
        t_start = time.time()
        res = hpob_hdlr.evaluate(method,
                                  search_space_id=search_space,
                                  dataset_id=dataset,
                                  seed=seed,
                                  n_trials=n_trials)
        t_end = time.time()
        print(search_space, dataset, seed, n_trials, "Completed in", t_end - t_start, "s")
    # This exception needs to be ignored due to issues with GP fitting the HPO-B data
    except gpytorch.utils.errors.NotPSDError:
        print("Ignoring the error and not recording this as a valid evaluation combination")
        res = []
    return (search_space, dataset, seed, n_trials), res

def get_all_combinations(hpob_hdlr, n_trials):
    # A total of 430 combinations are present in this if all seeds are used.
    seed_list = ["test0", "test1", "test2", "test3", "test4"]
    evaluation_list = []
    for search_space in hpob_hdlr.get_search_spaces():
        for dataset in hpob_hdlr.get_datasets(search_space):
            for seed in seed_list: # ["test2"]:  # seed_list: # use this for running on all possible seeds
                evaluation_list += [(search_space, dataset, seed, n_trials)]

    return evaluation_list

def evaluate_combinations(hpob_hdlr, method, keys_to_evaluate):

    print("Evaluating for", method)

    evaluation_list = []
    for key in keys_to_evaluate:
        search_space, dataset, seed, n_trials = key
        evaluation_list += [(search_space, dataset, seed, n_trials)]

    performance = []
    run_i = 0
    for eval_instance in evaluation_list:
        result = evaluation_worker(hpob_hdlr, method, eval_instance)
        performance.append(result)
        run_i = run_i + 1
        print("Completed Running", run_i, end="\n")
        gc.collect()

    return performance


def evaluate_FSBO(hpob_hdlr, keys_to_evaluate):
    performance = []
    for key in keys_to_evaluate:
        search_space_id, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space_id, dataset)
        method_fsbo = FSBO(search_space_id, input_dim=input_dim,
             latent_dim=10, batch_size=70, num_batches=50)
        #method_fsbo = FSBO(search_space_id, input_dim=input_dim,
        #                  latent_dim=32, batch_size=100, num_batches=1000)
        res = evaluate_combinations(hpob_hdlr, method_fsbo, keys_to_evaluate=[key])
        performance += res
    return performance

def evaluate(i, run):
    print("Evaluate test set (testing meta dataset)")
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3-test")
    dkt_keys = get_all_combinations(hpob_hdlr, 100)[i:i+1]
    dkt_performance = evaluate_FSBO(hpob_hdlr, keys_to_evaluate=dkt_keys)
    store_object(dkt_performance, "./FSBO_save/" + str(run) + "dkt_evaluation_32x4_100_03_cosAnn_" + str(i))

if __name__ == '__main__':
    # mp.freeze_support()
    i = int(sys.argv[1])
    run = int(sys.argv[2])
    mode = sys.argv[3]  # train/evaluate

    if mode == "train":
        # Pretrain DKT
        hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3")
        search_space_id = hpob_hdlr.get_search_spaces()[i]
        print("Training FSBO for " + search_space_id)
        meta_train_data = hpob_hdlr.meta_train_data[search_space_id]
        meta_val_data = hpob_hdlr.meta_validation_data[search_space_id]
        fsbo = FSBO(search_space_id, input_dim=get_input_dim(meta_train_data),
                    latent_dim=10, batch_size=70, num_batches=50)
        fsbo.train(meta_train_data, meta_val_data)

    if mode == "evaluate":
        evaluate(i, run)
"""
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
"""
