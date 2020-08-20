from sklearn.preprocessing import RobustScaler
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class RobustScalerOptuna(RobustScaler):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('RobustScaler_')

        self.q_min = trial.suggest_uniform(self.name + 'q_min', 0.001, 0.3)
        self.q_max = trial.suggest_uniform(self.name + 'q_max', 0.7, 0.999)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('RobustScaler_')

        space_gen.generate_number(self.name + 'q_min', 0.25, depending_node=depending_node)
        space_gen.generate_number(self.name + 'q_max', 0.75, depending_node=depending_node)
