from sklearn.random_projection import SparseRandomProjection
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class SparseRandomProjectionOptuna(SparseRandomProjection):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SparseRandomProjection_')

        self.eps = trial.suggest_loguniform(self.name + "eps", 1e-7, 1.0)

        if trial.suggest_categorical(self.name + 'density_auto', [True, False]):
            self.density = 'auto'
        else:
            self.density = trial.suggest_uniform(self.name + 'density_def', 1e-6, 1)

        self.dense_output = trial.suggest_categorical(self.name + 'dense_output', [True, False])

        self.n_components = 'auto'


    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SparseRandomProjection_')

        space_gen.generate_number(self.name + "eps", 0.1, depending_node=depending_node)

        category_density = space_gen.generate_cat(self.name + 'density_auto', [True, False], True,
                                                 depending_node=depending_node)

        space_gen.generate_number(self.name + 'density_def', 'auto', depending_node=category_density[1])

        space_gen.generate_cat(self.name + 'dense_output', [True, False], False, depending_node=depending_node)




