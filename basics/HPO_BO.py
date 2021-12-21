from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer
from sklearn.model_selection import cross_val_score
from skopt.utils import use_named_args
from skopt import gp_minimize
import numpy as np

X, Y = make_blobs(n_samples=500, centers=3, n_features=2)
model = KNeighborsClassifier()

search_space = [Integer(1,50,name='n_neighbors'), Integer(1,2,name='p')]

# Defining the objective function for our HPO
@use_named_args(search_space)
def evaluate_HP_configuration(n_neighbors, p):
    model.set_params(n_neighbors=n_neighbors, p=p)
    scores = cross_val_score(model, X, Y, cv=5, n_jobs=-1, scoring='accuracy')
    mean = np.mean(scores)
    return 1.0 - mean

res = gp_minimize(evaluate_HP_configuration, search_space)

print("Best Accuracy =", 1.0 - res.fun)
print("Best HP config =", res.x)
