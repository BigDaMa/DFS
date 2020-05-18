import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad, hessian


def custom_loss(y_true, y_pred):
	return np.sum(np.pow(y_pred, 3) * y_true) * -1


grad_custom_loss = grad(custom_loss)
hessian_custom_loss = grad(custom_loss)