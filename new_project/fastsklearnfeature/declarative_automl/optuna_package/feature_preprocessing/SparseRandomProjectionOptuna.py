from sklearn.random_projection import SparseRandomProjection
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class SparseRandomProjectionOptuna(SparseRandomProjection):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SparseRandomProjection_')

        self.n_components_fraction = trial.suggest_uniform(self.name + 'n_components_fraction', 0.0, 1.0)

        if trial.suggest_categorical(self.name + 'density_auto', [True, False]):
            self.density = 'auto'
        else:
            self.density = trial.suggest_uniform(self.name + 'density_def', 1e-6, 1)

        self.dense_output = trial.suggest_categorical(self.name + 'dense_output', [True, False])

        self.sparse = not self.dense_output

    def fit(self, X, y=None):
        self.n_components = max(1, int(self.n_components_fraction * X.shape[1]))
        return super().fit(X=X, y=y)


    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SparseRandomProjection_')

        space_gen.generate_number(self.name + 'n_components_fraction', 0.1, depending_node=depending_node)

        category_density = space_gen.generate_cat(self.name + 'density_auto', [True, False], True,
                                                 depending_node=depending_node)

        space_gen.generate_number(self.name + 'density_def', 'auto', depending_node=category_density[1])

        space_gen.generate_cat(self.name + 'dense_output', [True, False], False, depending_node=depending_node)




