from sklearn.linear_model import SGDClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class SGDClassifierOptuna(SGDClassifier):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SGDClassifier_')
        self.loss = trial.suggest_categorical(self.name + "loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"])
        self.penalty = trial.suggest_categorical(self.name + "penalty", ["l1", "l2", "elasticnet"])
        self.alpha = trial.suggest_loguniform(self.name + "alpha", 1e-7, 1e-1)
        self.l1_ratio = trial.suggest_loguniform(self.name + "l1_ratio", 1e-9, 1)
        self.fit_intercept = True
        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.epsilon = trial.suggest_loguniform(self.name + "epsilon", 1e-5, 1e-1)
        self.learning_rate = trial.suggest_categorical(self.name + "learning_rate", ["optimal", "invscaling", "constant"])

        if self.learning_rate == "constant" or self.learning_rate == "invscaling" or self.learning_rate == "adaptive":
            self.eta0 = trial.suggest_loguniform(self.name + "eta0", 1e-7, 1e-1)

        if self.learning_rate == "invscaling":
            self.power_t = trial.suggest_uniform(self.name + "power_t", 1e-5, 1)
        self.average = trial.suggest_categorical(self.name + "average", [False, True])

        self.max_iter = trial.suggest_int(self.name + "max_iter", 10, 1024, log=False)

        #todo: add conditional parameters

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SGDClassifier_')

        space_gen.generate_cat(self.name + "loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"], "log", depending_node=depending_node)
        space_gen.generate_cat(self.name + "penalty", ["l1", "l2", "elasticnet"], "l2", depending_node=depending_node)
        space_gen.generate_number(self.name + "alpha", 0.0001, depending_node=depending_node)
        space_gen.generate_number(self.name + "l1_ratio", 0.15, depending_node=depending_node)
        space_gen.generate_number(self.name + "tol", 1e-4, depending_node=depending_node)
        space_gen.generate_number(self.name + "epsilon", 1e-4, depending_node=depending_node)
        category_learn = space_gen.generate_cat(self.name + "learning_rate", ["optimal", "invscaling", "constant"], "invscaling", depending_node=depending_node)

        space_gen.generate_number(self.name + "eta0", 0.01, depending_node=depending_node) #fix if we change to graph
        space_gen.generate_number(self.name + "power_t", 0.5, depending_node=category_learn[1])
        space_gen.generate_cat(self.name + "average", [False, True], False, depending_node=depending_node)

        space_gen.generate_number(self.name + "max_iter", 1000, depending_node=depending_node)