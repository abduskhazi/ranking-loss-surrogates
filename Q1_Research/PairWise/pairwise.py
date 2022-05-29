import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

from itertools import product
from torch.nn import BCEWithLogitsLoss

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

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1

def rankNet(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


def average_ranks(X_query, rl_model):
    # Calculating the average rank of all inputs.
    score_list = []
    for nn in rl_model.sc:
        score_list += [nn(X_query).detach().numpy().flatten()]

    # Rank them and return the average rank.
    score_list = np.stack(score_list)
    ranks = scipy.stats.rankdata(score_list, axis=-1)
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

class RankingLossPair(nn.Module):
    def __init__(self, input_dim, file_name=None, ssid=None):
        super(RankingLossPair, self).__init__()
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

            loss = rankNet(prediction, y_obs)

            loss.backward()
            optimizer.step()

            loss_list += [loss.item()]

        # Plotting fine tune loss
        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        legend = ["Fine tune Loss for pairwise Ranking loss"]
        plt.legend(legend)
        plt.title("SSID: " + self.ssid + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.save_folder + self.ssid + "_" + sys.argv[1] + "_fine_tune_loss.png")
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
        restarted_model = RankingLossPair(input_dim=self.input_dim, ssid=self.ssid)
        restarted_model.fine_tune(X_obs, y_obs, epochs=1000, lr=0.02)
        scores = average_ranks(X_pen, restarted_model)

        idx = np.argmax(scores)
        return idx

def evaluate_DE_pair(hpob_hdlr, keys_to_evaluate):
    performance = []
    for key in keys_to_evaluate:
        search_space, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space, dataset)
        method = RankingLossPair(input_dim=input_dim, ssid=search_space)
        res = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=[key])
        performance += res

    return performance

def non_transfer_PairWise(i, run):
    hpob_hdlr = HPOBHandler(root_dir="../../HPO_B/hpob-data/", mode="v3-test")
    keys = get_all_combinations(hpob_hdlr, 100)
    print("Evaluating", i, "of ", len(keys))
    keys = keys[i:i + 1]  # Only executing the required keys.
    performance = evaluate_DE_pair(hpob_hdlr, keys_to_evaluate=keys)
    store_object(performance, "./" + str(run) + "/results/DE_PAIR_32x32x10_E1000_l0_02_OPT" + str(i))

if __name__ == '__main__':
    i = int(sys.argv[1])
    run = int(sys.argv[2])

    if non_transfer:
        print("Non Transfer: Evaluating DE with Pair-Wise loss");
        non_transfer_PairWise(i, run)

    if transfer:
        print("HPO Transfer: Evaluating DE with Pair-Wise loss (Not yet implemented)");
