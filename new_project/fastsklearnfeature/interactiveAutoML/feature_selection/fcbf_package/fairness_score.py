import numpy as np
import copy

def fairness_score(X, y=None, estimator=None, sensitive_ids=None):
	new_X_train = copy.deepcopy(X)
	new_y_train = copy.deepcopy(new_X_train[:, sensitive_ids[0]])
	new_X_train[:, sensitive_ids] = 0

	ranking_model = estimator
	ranking_model.fit(new_X_train, new_y_train)
	fairness_ranking = ranking_model.feature_importances_

	fairness_ranking[sensitive_ids] = np.max(fairness_ranking)

	fairness_ranking *= -1

	return fairness_ranking