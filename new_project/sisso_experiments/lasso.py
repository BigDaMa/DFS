import numpy as np
import sklearn.linear_model as lm

eps = 1e-08
X = np.load('/home/felix/phd/Lasso_debug/X_error.npy')
target = np.load('/home/felix/phd/Lasso_debug/target_error.npy')

print(X.shape)
print(target.shape)

reg = lm.LassoLarsCV(eps=eps)
reg.fit(X, target)
print(reg.predict(X))