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
non_transfer = False
transfer = True
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

    def generate_loss(self, prediction, y_true):
        prediction, y_true = flatten_for_loss_list(prediction, y_true)
        # Viewing everything as a 2D tensor.
        y_true = y_true.view(-1, y_true.shape[-1])
        prediction = prediction.view(-1, prediction.shape[-1])
        loss = pointwise_rmse(prediction, y_true, y_true.shape[-1])
        return loss

    def meta_train(self,meta_train_data, meta_val_data, epochs, batch_size, list_size, lr):
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}, ])  # 0.0001 giving good results
        loss_list = []
        val_loss_list = []
        for _ in range(epochs):
            self.train()
            for __ in range(100):
                optimizer.zero_grad()

                train_X, train_y = get_batch_HPBO_single(meta_train_data, 1, list_size)
                prediction = self.forward(train_X)
                loss = self.generate_loss(prediction, train_y)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.eval()

                # Calculating full training loss
                train_X, train_y = get_batch_HPBO(meta_train_data, batch_size, list_size)
                prediction = self.forward(train_X)
                loss = self.generate_loss(prediction, train_y)

                # Calculating validation loss
                val_X, val_y = get_batch_HPBO(meta_val_data, batch_size, list_size)
                pred_val = self.forward(val_X)
                val_loss = self.generate_loss(pred_val, val_y)

            # print("Epoch[", _, "] ==> Loss =", loss.item(), "; Val_loss =", val_loss.item())
            loss_list += [loss.item()]
            val_loss_list += [val_loss.item()]

        return loss_list, val_loss_list

def flatten_for_loss_list(pred, y):
    flatten_from_dim = len(pred.shape) - 2
    pred = torch.flatten(pred, start_dim=flatten_from_dim)
    y = torch.flatten(y, start_dim=flatten_from_dim)
    return pred, y

def get_batch_HPBO(meta_data, batch_size, list_size, random_state=None):
    query_X = []
    query_y = []

    rand_num_gen = np.random.RandomState(seed=random_state)  # As of now unused

    # Sample all tasks and form a high dimensional tensor of size
    #   (tasks, batch_size, list_size, input_dim)
    #   Suggestion : Take a tensor batch_size, list_size, input_dim for one gradient step.
    #   For segregation. https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html
    for data_task_id in meta_data.keys():
        data = meta_data[data_task_id]
        X = data["X"]
        y = data["y"]
        idx_query = rand_num_gen.choice(X.shape[0], size=(batch_size, list_size), replace=True)
        query_X += [torch.from_numpy(X[idx_query])]
        query_y += [torch.from_numpy(y[idx_query][..., 0])]

    return torch.stack(query_X), torch.stack(query_y)

def get_batch_HPBO_single(meta_train_data, batch_size, slate_length):
    query_X = []
    query_y = []
    for i in range(batch_size):
        data = meta_train_data[np.random.choice(list(meta_train_data.keys()))]
        X = data["X"]
        y = data["y"]
        idx = np.random.choice(X.shape[0], size=slate_length, replace=True)
        query_X += [torch.from_numpy(X[idx])]
        query_y += [torch.from_numpy(y[idx].flatten())]
    return torch.stack(query_X), torch.stack(query_y)

class RankingLossSurrogate(nn.Module):
    def __init__(self, input_dim, ssid, load=False):
        super(RankingLossSurrogate, self).__init__()
        self.input_dim = input_dim
        self.ssid = ssid
        self.M = 10

        self.save_folder = "./" + str(sys.argv[2]) + "/results/";
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        if load:
            self.load()
        else:
            self.sc = self.create_embedder_scorers_uncertainty(self.input_dim)


    def create_embedder_scorers_uncertainty(self, in_dim):
        sc_list = []
        for i in range(10):
            sc_list += [Scorer(input_dim=in_dim)]
        # For easing saving and loading from hard disk
        return nn.ModuleList(sc_list)

    def save(self):
        file_name = self.save_folder + self.ssid
        state_dict = self.sc.state_dict()
        torch.save({"input_dim": self.input_dim,
                    "ssid": self.ssid,
                    "M": self.M,
                    "scorer": state_dict,
                    "save_folder": self.save_folder},
                   file_name)

    def load(self):
        file_name = self.save_folder + self.ssid
        state_dict = torch.load(file_name)
        dict = torch.load(file_name)
        self.input_dim = dict["input_dim"]
        self.ssid = dict["ssid"]
        self.M = dict["M"]
        self.save_folder = dict["save_folder"]

        # Creating and initializing the scorer
        self.sc = self.create_embedder_scorers_uncertainty(self.input_dim)
        self.sc.load_state_dict(state_dict["scorer"])

    def train_model_separate(self, meta_train_data, meta_val_data, epochs, batch_size, list_size, lr):
        loss_list = []
        val_loss_list = []
        for nn in self.sc:
            l, vl = nn.meta_train(meta_train_data, meta_val_data, epochs, batch_size, list_size, lr)
            loss_list += [l]
            val_loss_list += [vl]

        loss_list = np.array(loss_list, dtype=np.float32)
        val_loss_list = np.array(val_loss_list, dtype=np.float32)
        loss_list = np.mean(loss_list, axis=0).tolist()
        val_loss_list = np.mean(val_loss_list, axis=0).tolist()

        return loss_list, val_loss_list

    def get_fine_tune_batch(self, X_obs, y_obs):

        idx_support = np.random.choice(X_obs.shape[0], size=support_size, replace=False)
        idx_query = np.delete(np.arange(X_obs.shape[0]), idx_support)

        s_ft_X = X_obs[idx_support]
        s_ft_y = y_obs[idx_support]
        q_ft_X = X_obs[idx_query]
        q_ft_y = y_obs[idx_query]

        return s_ft_X, s_ft_y, q_ft_X, q_ft_y

    def fine_tune_single(self, nn, X_obs, y_obs, epochs, lr):
        epochs = epochs
        loss_list = []
        optimizer = torch.optim.Adam([{'params': nn.parameters(), 'lr': lr},])
        for i in range(epochs):
            nn.train()
            optimizer.zero_grad()

            prediction = nn.forward(X_obs)
            loss = nn.generate_loss(prediction, y_obs)
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
        self.load()
        self.fine_tune(X_obs, y_obs, epochs=1000, lr=0.001)
        scores = average_ranks(X_pen, self)

        idx = np.argmax(scores)
        return idx

def evaluate_transfer_rl(hpob_hdlr, keys_to_evaluate):
    performance = []
    for key in keys_to_evaluate:
        search_space, dataset, _, _ = key
        input_dim = hpob_hdlr.get_input_dim(search_space, dataset)
        method = RankingLossSurrogate(input_dim=input_dim, ssid=search_space, load=True)
        res = evaluate_combinations(hpob_hdlr, method, keys_to_evaluate=[key])
        performance += res

    return performance

def transfer_eval(i, run):
    hpob_hdlr = HPOBHandler(root_dir="../../HPO_B/hpob-data/", mode="v3-test")
    keys = get_all_combinations(hpob_hdlr, 100)
    print("Evaluating", i, "of ", len(keys))
    keys = keys[i:i + 1]  # Only executing the required keys.
    performance = evaluate_transfer_rl(hpob_hdlr, keys_to_evaluate=keys)
    store_object(performance, "./" + str(run) + "/results/DE_POINT_32x32x10_E1000_l0_02_OPT" + str(i))

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
        list_size = 100
        rl_surrogate = RankingLossSurrogate(input_dim=input_dim, ssid=search_space_id)
        loss_list, val_loss_list = \
            rl_surrogate.train_model_separate(meta_train_data, meta_val_data, epochs, batch_size, list_size, 0.001)

        rl_surrogate.save()
        rl_surrogate.load()

        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        plt.plot(np.array(val_loss_list, dtype=np.float32))
        legend = ["Loss",
                  "Validation Loss"
                  ]
        plt.legend(legend)
        plt.title("SSID: " + search_space_id + "; Input dim: " + str(input_dim))
        plt.savefig(rl_surrogate.save_folder + "loss_" + search_space_id + ".png")


if __name__ == '__main__':
    i = int(sys.argv[1])
    run = int(sys.argv[2])

    if non_transfer:
        print("Non Transfer: Evaluating DE with Point-Wise loss");
        non_transfer_PointWise(i, run)

    if transfer:
        print("HPO Transfer: Evaluating DE with Point-Wise loss");
        if sys.argv[3] == "train":
            print("Meta training", i)
            meta_train_on_HPOB(i)
        elif sys.argv[3] == "evaluate":
            print("Evaluating", i)
            transfer_eval(i, run)
        else:
            print("Unknown option specified")
