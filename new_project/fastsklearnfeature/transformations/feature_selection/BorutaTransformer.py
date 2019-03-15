from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


class BorutaTransformer(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features,  output_dimensions):
        Transformation.__init__(self, 'Boruta',
                 number_parent_features, output_dimensions=output_dimensions,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)

        #classifier = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=10000)
        classifier = RandomForestClassifier(class_weight='balanced', max_depth=5)
        self.model = BorutaPy(classifier, n_estimators='auto', random_state=1)

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def transform(self, data):
        return self.model.transform(data)


