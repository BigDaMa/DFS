from sklearn.kernel_approximation import RBFSampler

class RBFSamplerOptuna(RBFSampler):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'RBFSampler_'

        self.gamma = trial.suggest_loguniform(self.name + "gamma", 3.0517578125e-05, 8)
        self.n_components = trial.suggest_int(self.name + "n_components", 50, 10000, log=True)

