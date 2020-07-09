from sklearn.naive_bayes import BernoulliNB


class BernoulliNBOptuna(BernoulliNB):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'BernoulliNB_'
        #self.classes_ = np.unique(y.astype(int))
        self.name = 'BernoulliNB_'
        self.alpha = trial.suggest_loguniform(self.name + "alpha", 1e-2, 100)
        self.fit_prior = trial.suggest_categorical(self.name + "fit_prior", [True, False])
