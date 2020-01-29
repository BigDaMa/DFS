import numpy as np
from sklearn.feature_selection import chi2

def chi2_score_wo(X, y=None):
	scores,_ = chi2(X, y)
	return scores