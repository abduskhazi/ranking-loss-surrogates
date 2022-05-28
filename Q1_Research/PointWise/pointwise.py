import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

# Local repo imports parent folder path
import sys
sys.path.append('../../../ranking-loss-surrogates')
from HPO_B.hpob_handler import HPOBHandler
from study_hpo import get_all_combinations, store_object, evaluate_combinations
from rankNet import rankNet
from fsbo import convert_meta_data_to_np_dictionary, get_input_dim

# =======================================================================================
# Quick Configuration
non_transfer = True
transfer = False
# =======================================================================================

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1

def pointwise_rmse(y_pred, y_true, no_of_levels, padded_value_indicator=PADDED_Y_VALUE):
    """
    Pointwise RMSE loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param no_of_levels: number of unique ground truth values
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    valid_mask = (y_true != padded_value_indicator).type(torch.float32)

    y_true[mask] = 0
    y_pred[mask] = 0

    errors = (y_true - no_of_levels * y_pred)

    squared_errors = errors ** 2

    mean_squared_errors = torch.sum(squared_errors, dim=1) / torch.sum(valid_mask, dim=1)

    rmses = torch.sqrt(mean_squared_errors)

    return torch.mean(rmses)

def average_ranks(X_query, rl_model):
    # Calculating the average rank of all input points.
    score_list = []
    for nn in rl_model.sc:
        score_list += [nn(X_query).detach().numpy().flatten()]

    # Rank them and return the average rank.
    score_list = np.stack(score_list)
    ranks = scipy.stats.rankdata(-1 * score_list, axis=-1)
    mean_rank = np.mean(ranks, axis=0)

    return mean_rank

# Defining our ranking model as a DNN.
# Keeping the model simple for now.
class Scorer(nn.Module):
    # Output dimension by default is 1 as we need a real valued score.
    def __init__(self, input_dim=1):
        super(Scorer, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class RankingLossPoint(nn.Module):
    def __init__(self, input_dim, file_name=None, ssid=None):
        super(RankingLossPoint, self).__init__()
        self.save_folder = "./" + str(sys.argv[2]) + "/results/";
        self.file_name = file_name
        self.ssid = ssid
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        if file_name:
            self.load(file_name)
        else:
            self.input_dim = input_dim
            self.sc = self.create_embedder_scorers_uncertainty(self.input_dim)

    def create_embedder_scorers_uncertainty(self, in_dim):
        sc_list = []
        for i in range(10):
            sc_list += [Scorer(input_dim=in_dim)]
        # Using Module List to make it easier to store and retrieve the neural networks.
        return nn.ModuleList(sc_list)

    def save(self, file_name):
        file_name = self.save_folder + file_name
        state_dict = self.sc.state_dict()
        torch.save({"input_dim": self.input_dim,
                    "scorer": state_dict},
                   file_name)

    def load(self, file_name):
        file_name = self.save_folder + file_name
        state_dict = torch.load(file_name)
        self.input_dim = state_dict["input_dim"]
        # Creating and initializing the scorer
        self.sc = self.create_embedder_scorers_uncertainty(self.input_dim)
        self.sc.load_state_dict(state_dict["scorer"])


    def flatten_for_loss_list(self, pred, y):
        flatten_from_dim = len(pred.shape) - 2
        pred = torch.flatten(pred, start_dim=flatten_from_dim)
        y = torch.flatten(y, start_dim=flatten_from_dim)
        return pred, y

    def fine_tune_single(self, nn, X_obs, y_obs, epochs, lr):
        epochs = epochs
        loss_list = []
        optimizer = torch.optim.Adam([{'params': nn.parameters(), 'lr': lr},])
        for i in range(epochs):
            nn.train()
            optimizer.zero_grad()

            prediction = nn.forward(X_obs)
            prediction, y_obs = self.flatten_for_loss_list(prediction, y_obs)

            # Viewing everything as a 2D tensor.
            y_obs = y_obs.view(-1, y_obs.shape[-1])
            prediction = prediction.view(-1, prediction.shape[-1])

            loss = pointwise_rmse(prediction, y_obs, y_obs.shape[-1])

            loss.backward()
            optimizer.step()

            loss_list += [loss.item()]

        # Plotting fine tune loss
        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        legend = ["Fine tune Loss for pointwise Ranking loss"]
        plt.legend(legend)
        plt.title("SSID: " + self.ssid + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.save_folder + self.ssid + "_fine_tune_loss.png")
        plt.close()

    def fine_tune(self, X_obs, y_obs, epochs, lr):
        for nn in self.sc:
            self.fine_tune_single(nn, X_obs, y_obs, epochs, lr)

    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        X_obs = np.array(X_obs, dtype=np.float32)
        y_obs = np.array(y_obs, dtype=np.float32)
        X_pen = np.array(X_pen, dtype=np.float32)
        X_obs = torch.from_numpy(X_obs)
        y_obs = torch.from_numpy(y_obs)
        X_pen = torch.from_numpy(X_pen)

        # Doing reloads from the saved model for every fine tuning.
        restarted_model = RankingLossPoint(input_dim=self.input_dim, ssid=self.ssid)
        restarted_model.fine_tune(X_obs, y_obs, epochs=1000, lr=0.02)
        scores = average_ranks(X_pen, restarted_model)

        idx = np.argmax(scores)
        return idx

def evaluate_DE_point(hpob_hdlr, keys_to_evaluate):
    performance = []
    for key in keys_to_evaluate:
        search_space, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space, dataset)
        method = RankingLossPoint(input_dim=input_dim, ssid=search_space)
        res = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=[key])
        performance += res

    return performance

def non_transfer_PointWise(i, run):
    hpob_hdlr = HPOBHandler(root_dir="../../HPO_B/hpob-data/", mode="v3-test")
    keys = get_all_combinations(hpob_hdlr, 100)
    print("Evaluating", i, "of ", len(keys))
    keys = keys[i:i + 1]  # Only executing the required keys.
    performance = evaluate_DE_point(hpob_hdlr, keys_to_evaluate=keys)
    store_object(performance, "./" + str(run) + "/results/DE_POINT_32x32x10_E1000_l0_02_OPT" + str(i))

if __name__ == '__main__':
    i = int(sys.argv[1])
    run = int(sys.argv[2])

    if non_transfer:
        print("Non Transfer: Evaluating DE with Point-wise loss");
        non_transfer_PointWise(i, run)

    if transfer:
        print("HPO Transfer: Evaluating DE with Point-wise loss (Not yet implemented)");
