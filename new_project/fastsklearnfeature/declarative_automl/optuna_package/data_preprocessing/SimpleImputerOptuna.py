from sklearn.impute import SimpleImputer

class SimpleImputerOptuna(SimpleImputer):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'SimpleImputer_'

        self.strategy = trial.suggest_categorical(self.name + "strategy", ["mean", "median", "most_frequent"])
