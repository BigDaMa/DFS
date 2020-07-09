from sklearn.linear_model import PassiveAggressiveClassifier

class PassiveAggressiveOptuna(PassiveAggressiveClassifier):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'PassiveAggressive_'
        self.C = trial.suggest_loguniform(self.name + "C", 1e-5, 10)
        self.fit_intercept = True
        self.loss = trial.suggest_categorical(self.name + "loss", ["hinge", "squared_hinge"])
        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.average = trial.suggest_categorical(self.name + "average", [False, True])
