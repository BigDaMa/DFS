from sklearn.naive_bayes import GaussianNB


class GaussianNBOptuna(GaussianNB):

    def init_hyperparameters(self, trial, X, y):
        #self.classes_ = np.unique(y.astype(int))
        pass

    def generate_hyperparameters(self, space_gen, depending_node=None):
        pass
