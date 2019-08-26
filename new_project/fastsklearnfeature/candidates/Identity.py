import numpy as np
def identity(X):
    try:
        return X.reshape(-1, 1).astype('float64')
    except:
        return X.reshape(-1, 1)