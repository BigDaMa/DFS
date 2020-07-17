from sklearn.preprocessing import OneHotEncoder
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class OneHotEncoderOptuna(OneHotEncoder):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('OneHotEncoder_')

        self.sparse = False
        self.handle_unknown='ignore'
