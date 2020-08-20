from sklearn.naive_bayes import BernoulliNB
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class BernoulliNBOptuna(BernoulliNB):

    def init_hyperparameters(self, trial, X, y):
        #self.classes_ = np.unique(y.astype(int))
        self.name = id_name('BernoulliNB_')
        self.alpha = trial.suggest_loguniform(self.name + "alpha", 1e-2, 100)
        self.fit_prior = trial.suggest_categorical(self.name + "fit_prior", [True, False])

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('BernoulliNB_')
        space_gen.generate_number(self.name + "alpha", 1, depending_node=depending_node)
        space_gen.generate_cat(self.name + "fit_prior", [True, False], True, depending_node=depending_node)
