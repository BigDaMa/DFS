from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class QuadraticDiscriminantAnalysisOptuna(QuadraticDiscriminantAnalysis):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('QuadraticDiscriminantAnalysis_')
        #self.classes_ = np.unique(y.astype(int))

        self.reg_param = trial.suggest_uniform(self.name + 'reg_param', 0.0, 1.0)

    def generate_hyperparameters(self, space_gen):
        self.name = id_name('QuadraticDiscriminantAnalysis_')
        space_gen.generate_number(self.name + 'reg_param', 0.0)
