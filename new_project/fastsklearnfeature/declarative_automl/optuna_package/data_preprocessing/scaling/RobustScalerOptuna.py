from sklearn.preprocessing import RobustScaler
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class RobustScalerOptuna(RobustScaler):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('RobustScaler_')

        self.q_min = trial.suggest_uniform(self.name + 'q_min', 0.001, 0.3)
        self.q_max = trial.suggest_uniform(self.name + 'q_max', 0.7, 0.999)
