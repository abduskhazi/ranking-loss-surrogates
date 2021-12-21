from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn import metrics
from skopt.utils import use_named_args
from skopt import gp_minimize
import skopt.space as spc
import numpy as np

# Obtain data and split it
cancer = datasets.load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.15)

clf = svm.SVC()
clf.fit(X_train, Y_train)
Y_hat = clf.predict(X_test)

print("Accuracy score =", metrics.accuracy_score(Y_test, Y_hat))
print("Precision score =", metrics.precision_score(Y_test, Y_hat))
print("Recall score =", metrics.recall_score(Y_test, Y_hat))

# Doing the hyperparameter optmization using GPs.
# Search Space
#    C -> 0.01 to 10 (float)
#    kernel -> Categorical {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
#              But we will use an RBF kernel by default.
#    degree -> Integer 1 to 5.
#    gamma -> Categorail {‘scale’, ‘auto’}
search_space = [
            spc.Real(0.001, 10, name='C'),
            spc.Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid'], name='kernel')#,
            #spc.Integer(1, 3, name='degree'),
            #spc.Real(1.0e-2, 1.0, name='gamma')
        ]

# Defining the objective function for our HPO
@use_named_args(search_space)
def objective(**params):
    clf.set_params(**params)
    scores = cross_val_score(clf, X_train, Y_train, cv=5, n_jobs=-1, scoring='accuracy')
    mean = np.mean(scores)
    return 1.0 - mean

res = gp_minimize(objective, search_space, verbose=True)
print("Best Accuracy =", 1.0 - res.fun)
print("Best HP config =", res.x)
