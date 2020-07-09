from sklearn.decomposition import PCA

class KernelPCAOptuna(PCA):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'PCA_'

        self.keep_variance = trial.suggest_uniform(self.name + "keep_variance", 0.5, 0.9999)
        self.whiten = trial.suggest_categorical(self.name + "whiten", [False, True])

