from sklearn.kernel_approximation import Nystroem
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class NystroemOptuna(Nystroem):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('Nystroem_')

        self.kernel = trial.suggest_categorical(self.name + 'kernel', ['poly', 'rbf', 'sigmoid', 'cosine', 'chi2'])
        self.n_components = trial.suggest_int(self.name + "n_components", 50, 10000, log=True)
        self.gamma = trial.suggest_loguniform(self.name + "gamma", 3.0517578125e-05, 8)
        self.degree = trial.suggest_int(self.name + 'degree', 2, 5, log=False)
        self.coef0 = trial.suggest_uniform(self.name + "coef0", -1, 1)

    def generate_hyperparameters(self, space_gen):
        self.name = id_name('Nystroem_')

        space_gen.generate_cat(self.name + 'kernel', ['poly', 'rbf', 'sigmoid', 'cosine', 'chi2'], 'rbf')
        space_gen.generate_number(self.name + "n_components", 100)
        space_gen.generate_number(self.name + "gamma", 0.1)
        space_gen.generate_number(self.name + 'degree', 3)
        space_gen.generate_number(self.name + "coef0", 0)

