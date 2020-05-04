import numpy as np
from sklearn.feature_selection import f_classif

def my_permutation_importance(X, y=None):
	scores,_ = f_classif(X, y)
	return scores