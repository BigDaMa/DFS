import numpy as np
from sklearn.feature_selection.from_model import _get_feature_importances

def model_score(X, y=None, estimator=None):
	estimator.fit(X,y)
	scores = _get_feature_importances(estimator)
	return scores