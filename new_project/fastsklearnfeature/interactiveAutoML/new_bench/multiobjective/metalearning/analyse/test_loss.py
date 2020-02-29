import numpy as np

my_manually_defined_array = np.array([[1,2,3], [4,5,6]])
y_pred = np.array([[0,0,1], [0,1,0]])

def my_loss (y_pred, y_true):
	return np.max(y_pred*my_manually_defined_array, axis=1)

print(my_loss(y_pred, None))