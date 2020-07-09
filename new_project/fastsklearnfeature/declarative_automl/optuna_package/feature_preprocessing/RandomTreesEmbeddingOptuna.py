from sklearn.ensemble import RandomTreesEmbedding

class RandomTreesEmbeddingOptuna(RandomTreesEmbedding):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'RandomTreesEmbedding_'

        self.n_estimators = trial.suggest_int("n_estimators", 10, 100)
        self.max_depth = trial.suggest_int("max_depth", 2, 10)
        self.min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        self.min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        self.min_weight_fraction_leaf = 0.0
        self.max_leaf_nodes = None
        self.bootstrap = trial.suggest_categorical('bootstrap', [True, False])

