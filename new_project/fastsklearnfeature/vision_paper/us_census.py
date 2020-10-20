import pandas as pd
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn import preprocessing

import sklearn.metrics
import autosklearn.regression
import autosklearn.classification
import autosklearn.pipeline.components.regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.vision_paper.NewOneHot import NewOneHot
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def df2data(df, train_index=None, test_index=None, target_attribute='Income', target_encoder=None):
    y = df[target_attribute].values
    if type(target_encoder) == type(None):
        target_encoder = preprocessing.LabelEncoder()
    y = target_encoder.fit_transform(y)
    df = df.drop(columns=[target_attribute])

    categorical_features = np.where(df.dtypes == object)[0]
    cat_pip = Pipeline(steps=[('impute', SimpleImputer(strategy='constant', fill_value='unknown_value')), ('one', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    pipeline = ColumnTransformer(
        transformers=[
            ('cat', cat_pip, categorical_features)
        ], remainder='passthrough'
    )
    X = df.values

    if type(train_index) == type(None):
        group_kfold = StratifiedKFold(n_splits=3)
        for train_index, test_index in group_kfold.split(X, y, groups=y):
            break

    X_train = pipeline.fit_transform(X[train_index])
    X_train = X_train.astype(float)
    X_test = pipeline.transform(X[test_index])
    X_test = X_test.astype(float)

    y_train = y[train_index]
    y_test = y[test_index]

    return X_train, y_train, X_test, y_test, train_index, test_index, target_encoder

df_clean = pd.read_csv('/home/felix/phd2/flights_vision/data/USCensus/raw/Holoclean_mv_clean.csv')
df_dirty = pd.read_csv('/home/felix/phd2/flights_vision/data/USCensus/raw/raw.csv')

X_train_clean, y_train_clean, X_test_clean, y_test_clean, train_index, test_index, target_encoder = df2data(df_clean)
X_train_dirty, y_train_dirty, X_test_dirty, y_test_dirty, _, _, _ = df2data(df_dirty,
                                                                               train_index=train_index,
                                                                               test_index=test_index,
                                                                               target_attribute='Income',
                                                                               target_encoder=target_encoder)


scorer = autosklearn.metrics.make_scorer(
        'f1_score',
        sklearn.metrics.f1_score
    )

cls_clean = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60*60,
                                                  resampling_strategy='cv',
                                                  resampling_strategy_arguments={'folds': 10},
                                                             metric=scorer)
cls_clean.fit(X_train_clean.copy(), y_train_clean.copy())
cls_clean.refit(X_train_clean.copy(), y_train_clean.copy())
predictions = cls_clean.predict(X_test_clean.copy())
print('clean: ' + str(f1_score(y_test_clean, predictions)))


cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60*60,
                                                       resampling_strategy='cv',
                                                       resampling_strategy_arguments={'folds': 10},
                                                       metric=scorer)
cls.fit(X_train_dirty.copy(), y_train_dirty.copy())
cls.refit(X_train_dirty.copy(), y_train_dirty.copy())
predictions = cls.predict(X_test_dirty.copy())
print('dirty: ' + str(f1_score(y_test_clean, predictions)))
