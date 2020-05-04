import numpy as np
from sklearn.feature_selection import f_classif

def f_anova_wo(X, y=None):
	scores,_ = f_classif(X, y)
	return scores