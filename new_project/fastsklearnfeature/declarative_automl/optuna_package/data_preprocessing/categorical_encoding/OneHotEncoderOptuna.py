from sklearn.preprocessing import OneHotEncoder
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class OneHotEncoderOptuna(OneHotEncoder):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('OneHotEncoder_')

        self.sparse = False
        self.handle_unknown = 'ignore'

        self.sparse = trial.suggest_categorical(self.name + 'sparse', [True, False])

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('OneHotEncoder_')
        space_gen.generate_cat(self.name + 'sparse', [True, False], True, depending_node=depending_node)
