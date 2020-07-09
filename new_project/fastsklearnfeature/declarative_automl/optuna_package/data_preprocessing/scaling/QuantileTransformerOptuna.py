from sklearn.preprocessing import QuantileTransformer

class QuantileTransformerOptuna(QuantileTransformer):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'QuantileTransformer_'

        self.n_quantiles = trial.suggest_int(self.name + 'n_quantiles', 10, 2000)
        self.output_distribution = trial.suggest_categorical(self.name + 'output_distribution', ['uniform', 'normal'])
