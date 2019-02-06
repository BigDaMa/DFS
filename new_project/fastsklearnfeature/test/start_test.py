from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from fastsklearnfeature.transformations.HigherOrderCommutativeTransformation import HigherOrderCommutativeTransformation
from fastsklearnfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastsklearnfeature.transformations.binary.NonCommutativeBinaryTransformation import NonCommutativeBinaryTransformation
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion




X, y = make_moons(noise=0.3, random_state=0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)


clf = LogisticRegression()
'''
pipeline1 = Pipeline([
('union1', ColumnTransformer(
        [
            # Pulling features from the post's subject line (first column)
            ('discretize0',  PandasDiscretizerTransformation(10), 0),
            ('0+1',  NonCommutativeBinaryTransformation(np.divide), [0,1])
        ]
    ))
    , ('logistic', clf)])
'''


# one feature one pipeline with \tmp
# https://www.kaggle.com/metadist/work-like-a-pro-with-pipelines-and-feature-unions

pipeline1 = Pipeline([
('union1', ColumnTransformer(
        [
            # Pulling features from the post's subject line (first column)
            ('discretize0',  PandasDiscretizerTransformation(10), 0),
            ('0+1',  NonCommutativeBinaryTransformation(np.divide), [0,1])
        ]
    ))])
pipeline2 = Pipeline([
('union2', ColumnTransformer(
        [
            # Pulling features from the post's subject line (first column)
            ('discretize0',  PandasDiscretizerTransformation(10), 0)
        ]
    ))])

combined_features = FeatureUnion([("p1", pipeline1), ("p2", pipeline2)])

pipeline = Pipeline([("features", combined_features), ('logistic', clf)])


scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(scores)