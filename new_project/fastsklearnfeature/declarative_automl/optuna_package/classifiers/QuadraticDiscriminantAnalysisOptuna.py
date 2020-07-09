from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class QuadraticDiscriminantAnalysisOptuna(QuadraticDiscriminantAnalysis):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'QuadraticDiscriminantAnalysis_'
        #self.classes_ = np.unique(y.astype(int))

        self.reg_param = trial.suggest_uniform(self.name + 'reg_param', 0.0, 1.0)
