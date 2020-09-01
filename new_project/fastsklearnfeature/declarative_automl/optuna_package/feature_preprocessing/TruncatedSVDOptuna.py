from sklearn.decomposition import TruncatedSVD
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class TruncatedSVDOptuna(TruncatedSVD):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('TruncatedSVD_')

        self.n_components_fraction = trial.suggest_uniform(self.name + 'n_components_fraction', 0.0, 1.0)

        self.sparse = False

    def fit(self, X, y=None):
        self.n_components = min(max(1, int(self.n_components_fraction * X.shape[1])), X.shape[1]-1)
        #print('ncom: ' + str(self.n_components))
        return super().fit(X=X, y=y)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('TruncatedSVD_')

        space_gen.generate_number(self.name + 'n_components_fraction', 0.5, depending_node=depending_node)



