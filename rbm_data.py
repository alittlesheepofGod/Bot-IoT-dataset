# import the necessary packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import numpy as np
from keras.datasets import mnist

# dataset
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

x_test = np.arange(3).reshape(-1, 1)
y_test = np.array([0, 0, 0])

# model 

# rbm = BernoulliRBM()
# logistic = LogisticRegression()
# classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])

# params = {
# 		"rbm__learning_rate": [0.1, 0.01, 0.001],
# 		"rbm__n_iter": [20, 40, 80],
# 		"rbm__n_components": [50, 100, 200],
# 		"logistic__C": [1.0, 10.0, 100.0]}
# # perform a grid search over the parameter
# gs = GridSearchCV(classifier, params, n_jobs = -1, verbose = 1)
# gs.fit(x, y)
# # print diagnostic information to the user and grab the
# # best model

# print("best score: %0.3f" % (gs.best_score_))
# print("RBM + LOGISTIC REGRESSION PARAMETERS")
# bestParams = gs.best_estimator_.get_params()
# # loop over the parameters and print each of them out
# # so they can be manually set
# for p in sorted(params.keys()):
# 	print("\t %s: %f" % (p, bestParams[p]))


rbm = BernoulliRBM(n_components = 50, n_iter = 20,
learning_rate = 0.01,  verbose = True)
logistic = LogisticRegression(C = 1.0)
# train the classifier and show an evaluation report
classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
classifier.fit(x, y)
print("RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET")
print(classification_report(y_test, classifier.predict(x_test)))
# nudge the dataset and then re-evaluate
print("RBM + LOGISTIC REGRESSION ON NUDGED DATASET")
print(classification_report(y_test, classifier.predict(x_test)))