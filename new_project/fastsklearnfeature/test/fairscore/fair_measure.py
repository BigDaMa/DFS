from sklearn.metrics import make_scorer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd

def true_positive_rate_score(y_true, y_pred, sensitive_data, labels =[False, True]):
	ids = [[],[]]

	flat_y_true = y_true.values.flatten()

	true_positive_rate = np.zeros(2)

	sensitive_values = list(np.unique(sensitive_data))

	if len(sensitive_values) != 2:
		return 0.0

	for i in range(2):
		ids[i] = np.where(sensitive_data[y_true.index.values] == sensitive_values[i])[0]

		tp = -1
		if len(ids[i]) == 0:
			true_positive_rate[i] = 0.0
			continue
		elif len(np.unique(flat_y_true[ids[i]])) == 1:
			if np.unique(flat_y_true[ids[i]])[0] == labels[0]:
				tp = 0
			else:
				tp = len(ids[i])
		else:
			_, _, _, tp = confusion_matrix(flat_y_true[ids[i]], y_pred[ids[i]], labels=labels).ravel()

		true_positive_rate[i] = tp / float(len(ids[i]))

	return np.abs(true_positive_rate[0] - true_positive_rate[1])




'''
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target == 1

sensitive = np.array(iris.target == 0)

parameters = {'C': [1, 0.1]}

fair_scorer = make_scorer(true_positive_rate_score, sensitive_data=sensitive)

logreg = LogisticRegression()
gridsearch = GridSearchCV(logreg, scoring=fair_scorer, cv=10, param_grid=parameters)
gridsearch.fit(X, pd.DataFrame(Y))

print(gridsearch.best_index_)
'''

