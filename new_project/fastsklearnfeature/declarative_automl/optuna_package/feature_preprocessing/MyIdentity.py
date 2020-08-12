from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformation(BaseEstimator, TransformerMixin):
    def transform(self, X):
        return X

    def fit(self, X, y=None):
        return self

    def generate_hyperparameters(self, space_gen, depending_node=None):
        pass
