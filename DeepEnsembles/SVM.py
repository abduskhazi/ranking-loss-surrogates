# Here, we use Deep ensembles as a surrogate for Hyper-Parameter loss surface as apposed to using a Gaussing process.
# We optimize only continous Hyper Parameters of SVM to begin with

# Three Hyper-Parameters optimized (We have a 3 dimensional input search space):
# 1. C - Log Scale [E-5 to E15]
# 2. gamma - Log Scale [E-15 to E3]
# 4. tol - [E-4 to E-2]

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import math
import torch
import numpy as np
from DeepEnsemble import DeepEnsemble

# Dataset for Binary classification task.
cancer = datasets.load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.15)

# Objective function is the function whose extremizer and extrema needs to be found
# Returns a score (i.e Loss/Performance) for any sampled input space point.
def objective(param_list):
    # Model for fitting the classifier for the given dataset.
    model = svm.SVC()

    # Preparing the parameters for the model.
    # This is required because the scales of the input dimensions are different.
    C = math.exp(param_list[0])
    gamma = math.exp(param_list[1])
    tol = param_list[2]

    # Further dividing the data into training and validation sets to evaluate the HPs.
    X_t, X_v, Y_t, Y_v = train_test_split(X_train, Y_train, test_size=0.20)
    model.set_params(C=C, gamma=gamma, tol=tol)
    model.fit(X_t, Y_t)
    score = model.score(X_v, Y_v)

    return score

# Defining the acquisition function - Upper Confidence Bound
def acquisition_UCB(X_samples, surrogate_model):
    mean, variance = surrogate_model.predict(X_samples)
    mean = mean.detach().numpy()
    std_dev = torch.sqrt(variance).detach().numpy()
    ucb = mean + 2 * std_dev # We are assuming Beta to be 4.
    return ucb

# Defining the optimization function of the acquisition function
def optimize_acquisition(surrogate_model):
    X_samples = np.array([sample_search_space() for i in range(1000)], dtype=np.float32)
    scores = acquisition_UCB(torch.from_numpy(X_samples), surrogate_model)
    ind = np.argmax(scores)
    return X_samples[ind]

# Function to random sample from the search space.
def sample_search_space():
    S = list(np.random.random(size=3))
    S[0] = (15 - (-5))   * S[0] - 5
    S[1] = (3 - (-15))   * S[1] - 15
    S[2] = (1e-2 - 1e-4) * S[2] + 1e-4
    return S

def get_search_space_range():
    ranges = torch.ones(1, 3)
    ranges[0, 0] = (15 - (-5))
    ranges[0, 1] = (3 - (-15))
    ranges[0, 2] = (1e-2 - 1e-4)
    return ranges

print("Getting initial evaluations of input samples")
# First get a few evaluations of the objective function to begin with
# Naming theta_X and theta_Y here so as not to get confused with
theta_X = np.array([sample_search_space() for i in range(5)], dtype=np.float32)
theta_Y = np.array([objective(t_x) for t_x in theta_X], dtype=np.float32)
print("Finished evaluations")

DE = DeepEnsemble(input_dim=3, M=5)

print("Training Begins")
ranges = get_search_space_range()
DE.train(theta_X, theta_Y.reshape(-1, 1), ranges, epochs=1000, batch_size=50)
print("Training finished")

#  Optimization cycle.
print("Running optimisation cycle")
for _ in range(20):
    print("Iteration:", _, end="\r")
    tx_opt = optimize_acquisition(DE)
    ty_opt = np.array([objective(tx_opt)])
    tx_opt = tx_opt.reshape(1, -1)
    theta_X = np.append(theta_X, tx_opt, axis=0)
    theta_Y = np.append(theta_Y, ty_opt, axis=0)
    # Training the neural networks only for a small number of epochs
    # Rationale - Most of the training has been completed, only slight modification needs to
    #             be done due to an additional data point.
    DE.train(theta_X, theta_Y.reshape(-1, 1), epochs=100, batch_size=100)
    # Using a fixed batch size of 100 as defined in the paper

print()
print("After optimization")
print("Maximizer =", theta_X[np.argmax(theta_Y)], ", Maxima =", np.max(theta_Y))

# Results obtained after running one optmisation
# After optimization (100 iterations) ADAM(0.01), Batch Size = 20
# Maximizer = [-1.7665786e+00 -7.2858520e+00  4.9927384e-03] , Maxima = 0.979381443298969
# After optimization (1000 iterations) ADAM(0.01), Batch size adaptive >= 50
# Maximizer = [-1.6634938  -8.352619    0.00887886] , Maxima = 0.979381443298969

# Results obtained by using parameters of the paper
# Maximizer = [ 1.3411719e+01 -1.4029912e+01  2.7184824e-03] , Maxima = 0.9896907216494846