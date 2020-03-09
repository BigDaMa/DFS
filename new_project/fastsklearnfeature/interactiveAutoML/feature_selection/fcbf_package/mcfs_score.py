import copy
from skfeature.function.sparse_learning_based.MCFS import mcfs

def my_mcfs(X, y):
	result =  mcfs(copy.deepcopy(X), X.shape[1])
	new_result = result.max(1)
	return new_result