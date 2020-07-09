from sklearn.preprocessing import OneHotEncoder

class OneHotEncoderOptuna(OneHotEncoder):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'OneHotEncoder_'

        self.sparse = False
        self.handle_unknown='ignore'
