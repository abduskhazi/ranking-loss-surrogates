import numpy as np
import torch
from DeepEnsembles.DeepEnsemble import DeepEnsemble
from scipy.stats import norm

def acquisition_EI(Y, X_samples, surrogate_model):
    # https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec
    # Find the best value of the objective function so far according to data.
    # Is this according to the gaussian fit or according to the actual values.???
    # For now using the best according to the actual values.
    best_y = np.max(Y)
    # Calculate the predicted mean & variance values of all the required samples
    mean, variance = surrogate_model.predict(X_samples)
    mean = mean.detach().numpy()
    std_dev = torch.sqrt(variance).detach().numpy()
    z = (mean - best_y) / (std_dev + 1E-9)
    return (mean - best_y) * norm.cdf(z) + (std_dev + 1E-9) * norm.pdf(z)

class DE_search:

    def __init__(self, input_dim):
        print("Using Deep Ensembles as method...")
        self.DE = DeepEnsemble(M=5, input_dim=input_dim, divided_nn=False, parallel_training=False)

    # Assuming that we are dealing with only discrete case ==> X_pen is None.
    # First fitting the model based on observations.
    # Predicting then the pending configuration for evaluation.
    # Returning the index that gives us the best results.
    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        X_obs = np.array(X_obs, dtype=np.float32)
        y_obs = np.array(y_obs, dtype=np.float32)
        X_pen = np.array(X_pen, dtype=np.float32)
        self.DE.train(X_obs, y_obs, epochs=100, lr=0.1, adverserial_training=False)
        X_pen = torch.from_numpy(X_pen)
        scores = acquisition_EI(y_obs, X_pen, self.DE)
        idx = np.argmax(scores)
        return idx


