# This file implements SVM Hyper-Parameters without using skopt library functions. Only continous Hyper-Parameters were
# optimized as optimizing discrete parameters needs the development of custom kernels. Custom kernel development is
# out of scope for this task.

# Three Hyper-Parameters optimized (We have a 3 dimensional input search space):
# 1. C - Log Scale [E-5 to E15]
# 2. gamma - Log Scale [E-15 to E3]
# 4. tol - [E-4 to E-2]

from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import svm
from scipy.stats import norm
from matplotlib import pyplot
import numpy as np
import math

cancer = datasets.load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.15)


# Objective function
def objective(param_list):
    # Defining the Support vector classification model.
    model = svm.SVC()

    # Preparing the parameters for the model.
    C = math.exp(param_list[0])
    gamma = math.exp(param_list[1])
    tol = param_list[2]

    # Further dividing the data into training and validation sets to evaluate the HPs.
    X_t, X_v, Y_t, Y_v = train_test_split(X_train, Y_train, test_size=0.20)
    model.set_params(C=C, gamma=gamma, tol=tol)
    model.fit(X_t, Y_t)
    score = model.score(X_v, Y_v)
    
    return score


def gridsample_search_space():
    x = np.array(np.linspace(-5, 15, 100), dtype=np.float32)
    y = np.array(np.linspace(-15, 3, 100), dtype=np.float32)
    z = np.array(np.linspace(1e-4, 1e-2, 100), dtype=np.float32)
    x_1, y_1, z_1 = np.meshgrid(x, y, z)
    return np.stack((x_1, y_1, z_1), axis=-1)

# Function to random sample from the search space.
def randomsample_search_space():
    S = list(np.random.random(size=3))
    S[0] = (15 - (-5))   * S[0] - 5
    S[1] = (3 - (-15))   * S[1] - 15
    S[2] = (1e-2 - 1e-4) * S[2] + 1e-4
    return S


def acquisition(gp_approximator, theta_X, best):
    theta_X = np.array(theta_X)
    mean, std = gp_approximator.predict(theta_X, return_std=True)
    return norm.cdf((mean - best) / (std + 1E-9))


def opt_acquisition(gp_approximator, best):
    theta_X = gridsample_search_space()
    # Reshaping because guassian process regressor does not accept tensors of higher dimensions.
    theta_X = np.reshape(theta_X, (-1, theta_X.shape[-1]))
    theta_Y = acquisition(gp_approximator, theta_X, best)
    i = np.argmax(theta_Y)
    return theta_X[i]


# Obtain the initial random objective evaluations.
theta_X = [randomsample_search_space() for _ in range(5)]
theta_Y = [objective(t_x) for t_x in theta_X]
theta_X = np.array(theta_X)
theta_Y = np.array(theta_Y)

# Create and fit the probabilistic model (GP process)
gp_approximator = GaussianProcessRegressor()
gp_approximator.fit(theta_X, theta_Y)

#   Add incumbent data for plotting
incumbent = []
# Run the optmization loop
for _ in range(100):
    print("Iteration:", _, end='\r')
    # Get the next best sample to evaluate
    t_x = opt_acquisition(gp_approximator, np.max(theta_Y))

    # Evaluate the objective and store the results
    t_y = objective(t_x)
    t_x = np.array(t_x).reshape(1, -1)
    t_y = np.array([t_y])
    theta_X = np.append(theta_X, t_x, axis=0)
    theta_Y = np.append(theta_Y, t_y, axis=0)

    # Refit the probabilistic model.
    gp_approximator.fit(theta_X, theta_Y)

    incumbent += [np.max(theta_Y)]

# Report the maximizer.
i = np.argmax(theta_Y)
maximizer = theta_X[i]
print("Maximizer = ", maximizer)

# Retrain and report the best results
# Preparing the model and its parameters
C, gamma, tol = math.exp(maximizer[0]), math.exp(maximizer[1]), maximizer[2]
model = svm.SVC(C=C, gamma=gamma, tol=tol)

model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)

print("Accuracy with the best HPO on the test data = ", score)

# Plotting the incumbent graph
pyplot.plot(np.array(range(1, len(incumbent)+1)), np.array(incumbent))
pyplot.show()

# Obtained the following result when I ran the above code
# Maximizer =  [ 7.47422034e-01 -1.05591509e+01  8.60451524e-03]
# Accuracy with the best HPO on the test data =  0.9767441860465116
