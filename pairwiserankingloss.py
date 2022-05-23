import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
# Local imports
from HPO_B.hpob_handler import HPOBHandler
from fsbo import convert_meta_data_to_np_dictionary, get_input_dim

import sys

from rankNet import rankNet

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1

def acquisition_EI(X, y, X_query, rl_model):
    # https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec
    # Find the best value of the objective function so far according to data.
    # Is this according to the gaussian fit or according to the actual values.???
    # For now using the best according to the actual values.
    best_y = np.max(y.detach().cpu().numpy())
    # Calculate the predicted mean & variance values of all the required samples
    # mean, variance = rl_model.forward_mean_var((X, y, X_query))
    mean, variance = rl_model.forward_mean_var(X)
    mean = mean.detach().cpu().numpy()
    std_dev = torch.sqrt(variance).detach().cpu().numpy()
    z = (mean - best_y) / (std_dev + 1E-9)
    return (mean - best_y) * norm.cdf(z) + (std_dev + 1E-9) * norm.pdf(z)

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
        # The last layer must not be passed through relu

        # however we pass it through torch.tanh to keep the output in a resonable range
        # I had answered regarding this on stackoverflow
        # https://ai.stackexchange.com/questions/31595/are-the-q-values-of-dqn-bounded-at-a-single-timestep/31648#31648
        # The range of our modelled function should be large enough to correctly map all input domain values.
        return 2 * torch.tanh(0.01 * x)


def get_batch_HPBO(meta_data, batch_size, list_size, random_state=None):
    query_X = []
    query_y = []

    rand_num_gen = np.random.RandomState(seed=random_state)

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
    def __init__(self, input_dim, file_name=None):
        super(RankingLossSurrogate, self).__init__()
        self.save_folder = "./save/pair_rlsurrogates_uncertainty/"
        self.file_name = file_name
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        if file_name:
            self.load(file_name)
        else:
            self.input_dim = input_dim
            self.sc = self.create_embedder_scorers_uncertainty(self.input_dim)

    def create_embedder_scorers_uncertainty(self, in_dim):
        sc_list = []
        for i in range(5):
            sc_list += [Scorer(input_dim=in_dim)]
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

    def forward_mean_var(self, input):
        X = input

        predictions = []
        for s in self.sc:
            predictions += [s(X)]

        predictions = torch.stack(predictions)
        mean = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0, keepdim=True)[0]

        return mean, variance

    def forward(self, input):
        X = input

        # return self.sc(q_X)
        predictions = []
        for s in self.sc:
            predictions += [s(X)]

        predictions = torch.stack(predictions)
        return torch.mean(predictions, dim=0)  #  self.sc(q_X)
        # return torch.sum(predictions, dim=0)  # self.sc(q_X)

    def flatten_for_loss_list(self, pred, y):
        flatten_from_dim = len(pred.shape) - 2
        pred = torch.flatten(pred, start_dim=flatten_from_dim)
        y = torch.flatten(y, start_dim=flatten_from_dim)
        return pred, y

    def train_model(self, meta_train_data, meta_val_data, epochs, batch_size, list_size, search_space_id):
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': 0.001},])  # 0.0001 giving good results
        loss_list = []
        val_loss_list = []
        for _ in range(epochs):
            self.train()

            batch_loss = []
            for __ in range(100):
                optimizer.zero_grad()
                train_X, train_y = get_batch_HPBO_single(meta_train_data, 1, list_size)

                prediction = self.forward(train_X)
                prediction, q_train_y = self.flatten_for_loss_list(prediction, train_y)

                # Viewing everything as a 2D tensor.
                train_y = train_y.view(-1, train_y.shape[-1])
                prediction = prediction.view(-1, prediction.shape[-1])

                loss = rankNet(prediction, train_y)

                loss.backward()
                optimizer.step()
                batch_loss += [loss]

            with torch.no_grad():
                self.eval()
                # Calculating training loss
                train_X, train_y = get_batch_HPBO(meta_train_data, batch_size, list_size)

                prediction = self.forward(train_X)
                prediction, train_y = self.flatten_for_loss_list(prediction, train_y)

                # Viewing everything as a 2D tensor.
                train_y = train_y.view(-1, train_y.shape[-1])
                prediction = prediction.view(-1, prediction.shape[-1])

                loss = rankNet(prediction, train_y)

                # Calculating validation loss
                val_X, val_y = get_batch_HPBO(meta_val_data, batch_size, list_size, int(search_space_id))

                pred_val = self.forward(val_X)
                pred_val, val_y = self.flatten_for_loss_list(pred_val, val_y)

                # Viewing everything as a 2D tensor.
                val_y = val_y.view(-1, val_y.shape[-1])
                pred_val = pred_val.view(-1, pred_val.shape[-1])

                val_loss = rankNet(pred_val, val_y)
                # val_loss = torch.mean(val_loss)

            # if val_loss.item() < min(val_loss_list + [np.inf]):
            #    self.save(search_space_id)

            print("Epoch[", _, "] ==> Loss =", loss.item(), "; Val_loss =", val_loss.item())
            loss_list += [loss.item()]
            val_loss_list += [val_loss.item()]

        self.save(search_space_id)

        return loss_list, val_loss_list

    def get_fine_tune_batch(self, X_obs, y_obs):

        idx_support = np.random.choice(X_obs.shape[0], size=support_size, replace=False)
        idx_query = np.delete(np.arange(X_obs.shape[0]), idx_support)

        s_ft_X = X_obs[idx_support]
        s_ft_y = y_obs[idx_support]
        q_ft_X = X_obs[idx_query]
        q_ft_y = y_obs[idx_query]

        return s_ft_X, s_ft_y, q_ft_X, q_ft_y

    def fine_tune(self, X_obs, y_obs):
        epochs = 1000
        loss_list = []
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': 0.001},])
        scheduler_fn = lambda x, y: torch.optim.lr_scheduler.CosineAnnealingLR(x, y, eta_min=0.0001)
        scheduler = scheduler_fn(optimizer, epochs)
        for i in range(epochs):
            self.train()
            optimizer.zero_grad()

            prediction = self.forward(X_obs)
            prediction, y_obs = self.flatten_for_loss_list(prediction, y_obs)

            # Viewing everything as a 2D tensor.
            y_obs = y_obs.view(-1, y_obs.shape[-1])
            prediction = prediction.view(-1, prediction.shape[-1])

            loss = rankNet(prediction, y_obs)

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_list += [loss.item()/X_obs.shape[0]]

        # Plotting fine tune loss
        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        legend = ["Fine tune Loss: Ranking losses"]
        plt.legend(legend)
        plt.title("SSID: " + self.file_name + "; Input dim: " + str(self.input_dim))
        plt.savefig(self.save_folder + self.file_name + "_fine_tune_loss.png")
        plt.close()

        # self.save(self.file_name + "_ft_early_stop")
        # self.load(self.file_name + "_ft_early_stop")

        return loss_list

    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        X_obs = np.array(X_obs, dtype=np.float32)
        y_obs = np.array(y_obs, dtype=np.float32)
        X_pen = np.array(X_pen, dtype=np.float32)
        X_obs = torch.from_numpy(X_obs)
        y_obs = torch.from_numpy(y_obs)
        X_pen = torch.from_numpy(X_pen)

        # Doing reloads from the saved model for every fine tuning.
        restarted_model = RankingLossSurrogate(input_dim=-1, file_name=self.file_name)
        restarted_model.fine_tune(X_obs, y_obs)
        scores = acquisition_EI(X_obs, y_obs, X_pen, restarted_model)

        idx = np.argmax(scores)
        return idx

def pre_train_HPOB(i):
    hpob_hdlr = HPOBHandler(root_dir="HPO_B/hpob-data/", mode="v3")

    # Pretrain Ranking loss surrogate with all search spaces
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
        rlsurrogate = RankingLossSurrogate(input_dim=input_dim)
        loss_list, val_loss_list = \
            rlsurrogate.train_model(meta_train_data, meta_val_data, epochs, batch_size, list_size, search_space_id)

        rlsurrogate.load(search_space_id)

        plt.figure(np.random.randint(999999999))
        plt.plot(np.array(loss_list, dtype=np.float32))
        plt.plot(np.array(val_loss_list, dtype=np.float32))
        legend = ["Loss",
                  "Validation Loss"
                  ]
        plt.legend(legend)
        plt.title("SSID: " + search_space_id + "; Input dim: " + str(input_dim))
        plt.savefig(rlsurrogate.save_folder + "loss_" + search_space_id + ".png")

if __name__ == '__main__':
    # Unit testing our loss functions
    # test_toy_problem()

    pre_train_HPOB(int(sys.argv[1]))
