from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# Obtain data and split it
cancer = datasets.load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.15)
# Further splitting it into training and validation for hyperparameter optimization
X_train, X_validate, Y_train, Y_Validate = train_test_split(X_train, Y_train, test_size=0.15)

clf = svm.SVC()
clf.fit(X_train, Y_train)
Y_hat = clf.predict(X_test)

print("Accuracy score =", metrics.accuracy_score(Y_test, Y_hat))
print("Precision score =", metrics.precision_score(Y_test, Y_hat))
print("Recall score =", metrics.recall_score(Y_test, Y_hat))



#print(cancer.data.shape)
#print(cancer.data[0:5])
#print(cancer.target)
#print("Features: ", cancer.feature_names)
#print("Labels: ", cancer.target_names)
