from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class FastICAOptuna(FastICA):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('FastICA_')

        self.algorithm = trial.suggest_categorical(self.name + 'algorithm', ['parallel', 'deflation'])
        self.whiten = trial.suggest_categorical(self.name + 'whiten', [False, True])

        self.n_components_fraction = None
        if self.whiten == True:
            self.n_components_fraction = trial.suggest_uniform(self.name + 'n_components_fraction', 0.0, 1.0)

        self.fun = trial.suggest_categorical(self.name + 'fun', ['logcosh', 'exp', 'cube'])

        self.sparse = False

    def fit(self, X, y=None):
        if type(self.n_components_fraction) != type(None):
            self.n_components = max(1, int(self.n_components_fraction * X.shape[1]))
            #print('ncom: ' + str(self.n_components))
        return super().fit(X=X, y=y)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('FastICA_')

        category_whiten = space_gen.generate_cat(self.name + 'whiten', [False, True], False,
                                                 depending_node=depending_node)

        space_gen.generate_number(self.name + 'n_components_fraction', 0.5, depending_node=category_whiten[1])
        space_gen.generate_cat(self.name + 'algorithm', ['parallel', 'deflation'], 'parallel', depending_node=depending_node)

        space_gen.generate_cat(self.name + 'fun', ['logcosh', 'exp', 'cube'], 'logcosh', depending_node=depending_node)



