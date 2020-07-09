from sklearn.naive_bayes import MultinomialNB


class MultinomialNBOptuna(MultinomialNB):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'MultinomialNB_'
        #self.classes_ = np.unique(y.astype(int))

        self.alpha = trial.suggest_loguniform(self.name + "alpha", 1e-2, 100)
        self.fit_prior = trial.suggest_categorical(self.name + "fit_prior", [True, False])

