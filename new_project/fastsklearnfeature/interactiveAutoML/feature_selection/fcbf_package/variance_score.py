import numpy as np

def variance(X, y=None):
	variances_ = np.var(X, axis=0)
	return variances_