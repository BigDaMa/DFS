from sklearn.ensemble import RandomTreesEmbedding
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class RandomTreesEmbeddingOptuna(RandomTreesEmbedding):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('RandomTreesEmbedding_')

        self.n_estimators = trial.suggest_int(self.name + "n_estimators", 10, 100)
        self.max_depth = trial.suggest_int(self.name + "max_depth", 2, 10)
        self.min_samples_split = trial.suggest_int(self.name + "min_samples_split", 2, 20)
        self.min_samples_leaf = trial.suggest_int(self.name + "min_samples_leaf", 1, 20)
        self.min_weight_fraction_leaf = 0.0
        self.max_leaf_nodes = None
        self.bootstrap = trial.suggest_categorical(self.name + 'bootstrap', [True, False])

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('RandomTreesEmbedding_')

        space_gen.generate_number(self.name + "n_estimators", 10, depending_node=depending_node)
        space_gen.generate_number(self.name + "max_depth", 5, depending_node=depending_node)
        space_gen.generate_number(self.name + "min_samples_split", 2, depending_node=depending_node)
        space_gen.generate_number(self.name + "min_samples_leaf", 1, depending_node=depending_node)
        space_gen.generate_cat(self.name + "bootstrap", [True, False], False, depending_node=depending_node)


