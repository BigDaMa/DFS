from sklearn.preprocessing import Normalizer

class NormalizerOptuna(Normalizer):
    def init_hyperparameters(self, trial, X, y):
        pass

    def generate_hyperparameters(self, space_gen, depending_node=None):
        pass