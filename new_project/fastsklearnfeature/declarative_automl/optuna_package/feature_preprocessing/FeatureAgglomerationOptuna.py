from sklearn.cluster import FeatureAgglomeration
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import categorical
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class FeatureAgglomerationOptuna(FeatureAgglomeration):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('FeatureAgglomeration_')

        self.n_components_fraction = trial.suggest_uniform(self.name + 'n_components_fraction', 0.0, 1.0)

        self.linkage = trial.suggest_categorical(self.name + "linkage", ["ward", "complete", "average"])

        if self.linkage == 'ward':
            self.affinity = "euclidean"
        else:
            self.affinity = trial.suggest_categorical(self.name + "affinity", ["euclidean", "manhattan", "cosine"])

        self.pooling_func = trial.suggest_categorical(self.name + "pooling_func", [np.mean, np.median, np.max])

        self.sparse = False

    def fit(self, X, y=None, **params):
        self.n_clusters = max(1, int(self.n_components_fraction * X.shape[1]))
        #print('ncom: ' + str(self.n_clusters))
        return super().fit(X=X, y=y, **params)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('FeatureAgglomeration_')

        space_gen.generate_number(self.name + 'n_components_fraction', 0.5, depending_node=depending_node)
        space_gen.generate_cat(self.name + "linkage", ["ward", "complete", "average"], "ward", depending_node=depending_node)
        space_gen.generate_cat(self.name + "affinity", ["euclidean", "manhattan", "cosine"], "euclidean", depending_node=depending_node)
        space_gen.generate_cat(self.name + "pooling_func", [np.mean, np.median, np.max], np.mean, depending_node=depending_node)


