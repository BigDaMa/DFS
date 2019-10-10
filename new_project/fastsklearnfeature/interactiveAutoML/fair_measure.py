import numpy as np
from sklearn.metrics import confusion_matrix

# Equality of Opportunity
#https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb
def true_positive_rate_score(y_true, y_pred, sensitive_data, labels =[False, True]):
	ids = [[],[]]

	flat_y_true = y_true.values.flatten()
	true_positive_rate = np.zeros(2)
	sensitive_values = list(np.unique(sensitive_data))

	for i in range(len(sensitive_values)):
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