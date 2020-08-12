from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class FastICAOptuna(FastICA):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('FastICA_')

        self.n_components = trial.suggest_int(self.name + "n_components", 10, 2000, log=False)
        self.algorithm = trial.suggest_categorical(self.name + 'algorithm', ['parallel', 'deflation'])
        self.whiten = trial.suggest_categorical(self.name + 'whiten', [False, True])
        self.fun = trial.suggest_categorical(self.name + 'fun', ['logcosh', 'exp', 'cube'])

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('FastICA_')

        space_gen.generate_number(self.name + "n_components", 100, depending_node=depending_node)
        space_gen.generate_cat(self.name + 'algorithm', ['parallel', 'deflation'], 'parallel', depending_node=depending_node)
        space_gen.generate_cat(self.name + 'whiten', [False, True], False, depending_node=depending_node)
        space_gen.generate_cat(self.name + 'fun', ['logcosh', 'exp', 'cube'], 'logcosh', depending_node=depending_node)



