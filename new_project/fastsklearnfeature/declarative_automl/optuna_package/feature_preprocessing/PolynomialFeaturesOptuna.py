from sklearn.preprocessing import PolynomialFeatures

class PolynomialFeaturesOptuna(PolynomialFeatures):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'PolynomialFeatures_'

        self.degree = trial.suggest_int(self.name + "degree", 2, 3)
        self.interaction_only = trial.suggest_categorical(self.name + "interaction_only", [False, True])
        self.include_bias = trial.suggest_categorical(self.name + "include_bias", [True, False])


