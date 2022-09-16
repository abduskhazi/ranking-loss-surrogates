import scipy
import torch
import numpy as np
from scipy.stats import norm

def get_acuisition_func(name):
    if name.lower() == "avg":
        return average_rank_deep_set
    if name.lower() == "ucb":
        return UCB_rank_deep_set
    if name.lower() == "ei":
        return EI_rank_deep_set

def average_rank_deep_set(input, incumbent, DRE):
    score_list = []
    for s in DRE.forward_separate_deep_set(input):
        score_list += [s.detach().numpy().flatten()]

    # Rank the score list and return the mean rank as the acquisition score.
    score_list = np.stack(score_list)
    ranks = scipy.stats.rankdata(score_list, axis=-1)
    mean_rank = np.mean(ranks, axis=0)
    return mean_rank

def UCB_rank_deep_set(input, incumbent, DRE):
    score_list = []
    for sl in DRE.forward_separate_deep_set(input):
        score_list += [sl.detach().numpy().flatten()]

    # Rank the score list and return the UCB acquisition score.
    score_list = np.stack(score_list)
    ranks = scipy.stats.rankdata(score_list, axis=-1)
    mean_rank = np.mean(ranks, axis=0)
    var_rank = np.var(ranks, axis=0)
    return mean_rank + 0.1 * np.sqrt(var_rank)

def EI_rank_deep_set(input, incumbent, DRE):
    # Append incumbent to the query input (to calculate its rank w.r.t the current query input).
    input = (input[0], input[1], torch.cat((input[2], incumbent[None, :]), axis=0))

    score_list = []
    for s in DRE.forward_separate_deep_set(input):
        score_list += [s.detach().numpy().flatten()]

    # Rank the score list and calculate the mean and standard deviation of the ranks.
    score_list = np.stack(score_list)
    ranks = scipy.stats.rankdata(score_list, axis=-1)
    mean_rank = np.mean(ranks, axis=0)
    std_rank = np.sqrt(np.var(ranks, axis=0))

    # Obtain and remove the incumbent from the results.
    best_y = mean_rank[-1]
    mean_rank = mean_rank[:-1]
    std_rank = std_rank[:-1]

    # Return the EI acquisition score.
    z = (mean_rank - best_y) / (std_rank + 1E-9)
    return (mean_rank - best_y) * norm.cdf(z) + (std_rank + 1E-9) * norm.pdf(z)
