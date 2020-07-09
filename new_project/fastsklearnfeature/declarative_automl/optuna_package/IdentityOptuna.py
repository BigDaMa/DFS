from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.MyIdentity import IdentityTransformation

class IdentityOptuna(IdentityTransformation):

    def __init__(self):
        pass

    def init_hyperparameters(self, trial, X, y):
        pass

