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

def listMLE(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    # Weighted ranking because it is more important to get the the first ranks right than the rest
    weight = np.log(np.arange(observation_loss.shape[-1]) + 2) # To prevent loss of log(1)
    weight = np.array(weight, dtype=np.float32)
    weight = torch.from_numpy(weight)[None, :]
    observation_loss = observation_loss / weight

    return torch.mean(torch.sum(observation_loss, dim=1))


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

def average_ranks_deep_set(input, rl_model):
    # Calculating the average rank of all inputs.

    score_list = []
    for sl in rl_model.forward_separate_deep_set(input):
        score_list += [sl.detach().numpy().flatten()]

    # Rank them and return the average rank.
    score_list = np.stack(score_list)
    ranks = scipy.stats.rankdata(score_list, axis=-1)
    mean_rank = np.mean(ranks, axis=0)

    return mean_rank


def get_fine_tune_batch(X_obs, y_obs):

    # Taking 20% of the data as the support set.
    support_size = int(0.2 * X_obs.shape[0])
    idx_support = np.random.choice(X_obs.shape[0], size=support_size, replace=False)
    idx_query = np.delete(np.arange(X_obs.shape[0]), idx_support)

    s_ft_X = X_obs[idx_support]
    s_ft_y = y_obs[idx_support]
    q_ft_X = X_obs[idx_query]
    q_ft_y = y_obs[idx_query]

    return s_ft_X, s_ft_y, q_ft_X, q_ft_y

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


class DeepSet(nn.Module):
    def __init__(self, input_dim=1, latent_dim=1, output_dim=1):
        super(DeepSet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.phi = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.rho = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # Encoder: First get the latent embedding of the whole batch
        x = self.phi(x)

        # Pool operation: Aggregate all the outputs to a single output.
        #                 i.e across size of support set
        # Using mean as the validation error instead of sum
        # because the cardinality should be irrelevant
        x = torch.mean(x, dim=-2)

        # Decoder: Decode the latent output to result
        x = self.rho(x)

        return x


class RankingLossList(nn.Module):
    def __init__(self, input_dim, file_name=None, ssid=None):
        super(RankingLossList, self).__init__()
        self.save_folder = "./" + str(sys.argv[2]) + "/results/";
        self.file_name = file_name
        self.ssid = ssid
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        if file_name:
            self.load(file_name)
        else:
            self.input_dim = input_dim
            self.sc, self.ds_embedder = self.create_embedder_scorers_uncertainty(self.input_dim)

    def create_embedder_scorers_uncertainty(self, in_dim):
        ds_embedder = DeepSet(input_dim=in_dim + 1, latent_dim=32, output_dim=16)
        sc_list = []
        for i in range(10):
            sc_list += [Scorer(input_dim=16 + in_dim)]
        # Using Module List to make it easier to store and retrieve the neural networks.
        return nn.ModuleList(sc_list), ds_embedder

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

            loss = listMLE(prediction, y_obs)

            loss.backward()
            optimizer.step()

            loss_list += [loss.item()]

        # Plotting fine tune loss
        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        legend = ["Fine tune Loss for listwise Ranking loss"]
        plt.legend(legend)
        plt.title("SSID: " + self.ssid + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.save_folder + self.ssid + "_" + sys.argv[1] + "_fine_tune_loss.png")
        plt.close()

    def fine_tune(self, X_obs, y_obs, epochs, lr):
        for nn in self.sc:
            self.fine_tune_single(nn, X_obs, y_obs, epochs, lr)

    def forward(self, input):
        s_X, s_y, q_X = input

        # Creating an embedding of X:y for the support data using the embedder
        s_X = torch.cat((s_X, s_y), dim=-1)
        s_X = self.ds_embedder(s_X)

        # Creating an input for the scorer.
        s_X = s_X[..., None, :]
        repeat_tuple = (1,) * (len(s_X.shape)-2) + (q_X.shape[-2], 1)
        s_X = s_X.repeat(repeat_tuple)
        q_X = torch.cat((s_X, q_X), dim=-1)

        predictions = []
        for s in self.sc:
            predictions += [s(q_X)]

        predictions = torch.stack(predictions)
        return torch.mean(predictions, dim=0)

    def forward_separate_deep_set(self, input):
        s_X, s_y, q_X = input

        # Creating an embedding of X:y for the support data using the embedder
        s_X = torch.cat((s_X, s_y), dim=-1)
        s_X = self.ds_embedder(s_X)

        # Creating an input for the scorer.
        s_X = s_X[..., None, :]
        repeat_tuple = (1,) * (len(s_X.shape)-2) + (q_X.shape[-2], 1)
        s_X = s_X.repeat(repeat_tuple)
        q_X = torch.cat((s_X, q_X), dim=-1)

        predictions = []
        for s in self.sc:
            predictions += [s(q_X)]

        return predictions

    def fine_tune_together(self, X_obs, y_obs, epochs, lr):
        epochs = epochs
        loss_list = []
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}, ])
        for i in range(epochs):
            self.train()
            optimizer.zero_grad()

            s_ft_X, s_ft_y, q_ft_X, q_ft_y = get_fine_tune_batch(X_obs, y_obs)

            prediction = self.forward((s_ft_X, s_ft_y, q_ft_X))
            prediction, y_obs_res = self.flatten_for_loss_list(prediction, y_obs)

            # Viewing everything as a 2D tensor.
            y_obs_res = y_obs_res.view(-1, y_obs_res.shape[-1])
            prediction = prediction.view(-1, prediction.shape[-1])

            loss = listMLE(prediction, y_obs_res)

            loss.backward()
            optimizer.step()

            loss_list += [loss.item()]

        # Plotting fine tune loss
        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        legend = ["Fine tune Loss for listwise Ranking loss"]
        plt.legend(legend)
        plt.title("SSID: " + self.ssid + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.save_folder + self.ssid + "_" + sys.argv[1] + "_fine_tune_loss.png")
        plt.close()

    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        X_obs = np.array(X_obs, dtype=np.float32)
        y_obs = np.array(y_obs, dtype=np.float32)
        X_pen = np.array(X_pen, dtype=np.float32)
        X_obs = torch.from_numpy(X_obs)
        y_obs = torch.from_numpy(y_obs)
        X_pen = torch.from_numpy(X_pen)

        # Doing reloads from the saved model for every fine tuning.
        restarted_model = RankingLossList(input_dim=self.input_dim, ssid=self.ssid)
        restarted_model.fine_tune_together(X_obs, y_obs, epochs=1000, lr=0.02)
        scores = average_ranks_deep_set((X_obs, y_obs, X_pen), restarted_model)

        idx = np.argmax(scores)
        return idx

def evaluate_DE_list(hpob_hdlr, keys_to_evaluate):
    performance = []
    for key in keys_to_evaluate:
        search_space, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space, dataset)
        method = RankingLossList(input_dim=input_dim, ssid=search_space)
        res = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=[key])
        performance += res

    return performance

def non_transfer_ListWise(i, run):
    hpob_hdlr = HPOBHandler(root_dir="../../HPO_B/hpob-data/", mode="v3-test")
    keys = get_all_combinations(hpob_hdlr, 100)
    print("Evaluating", i, "of ", len(keys))
    keys = keys[i:i + 1]  # Only executing the required keys.
    performance = evaluate_DE_list(hpob_hdlr, keys_to_evaluate=keys)
    store_object(performance, "./" + str(run) + "/results/DE_LIST_32x32x10_E1000_l0_02_OPT" + str(i))

if __name__ == '__main__':
    i = int(sys.argv[1])
    run = int(sys.argv[2])

    if non_transfer:
        print("Non Transfer: Evaluating DE with List-Wise loss");
        non_transfer_ListWise(i, run)

    if transfer:
        print("HPO Transfer: Evaluating DE with List-Wise loss (Not yet implemented)");
