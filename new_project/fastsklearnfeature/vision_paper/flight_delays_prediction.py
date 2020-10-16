import pandas as pd
from dateutil.parser import parse
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import GroupKFold
import copy

from fastsklearnfeature.vision_paper.NewOneHot import NewOneHot


def toTimestamp(x):
    try:
        return parse(x).timestamp()
    except:
        return np.NAN


def df_to_ML_task(df, train_index=None, test_index=None):
    df = df.drop(columns=['tuple_id'])

    df['sched_arr_time'] = df['sched_arr_time'].apply(lambda x: toTimestamp(x))
    df['sched_dep_time'] = df['sched_dep_time'].apply(lambda x: toTimestamp(x))
    df['act_arr_time'] = df['act_arr_time'].apply(lambda x: toTimestamp(x))
    df['act_dep_time'] = df['act_dep_time'].apply(lambda x: toTimestamp(x))

    df[['Airline', 'Flight Number', 'dep_airport', 'arr_airport']] = df.flight.str.split("-", expand=True, )
    df = df.drop(columns=['flight'])

    y = (df['act_arr_time'].values - df['sched_arr_time'].values) / 60.0 #minutes

    y = y > 5.0

    #for i in range(len(df['sched_arr_time'].values)):
    #    print('scheduled:' + str(df['sched_arr_time'].values[i]) + ' actual: ' + str(df['act_arr_time'].values[i]) + ' diff: ' + str(y[i]/60.0))

    df = df.drop(columns=['act_arr_time'])

    categorical_features = np.where(df.dtypes == object)[0]
    pipeline = ColumnTransformer(
        transformers=[
            ('cat', NewOneHot(), categorical_features)
        ], remainder='passthrough'
    )



    X = df.values

    if type(train_index) == type(None):
        group_kfold = GroupKFold(n_splits=3)
        for train_index, test_index in group_kfold.split(X, y, df['Flight Number']):
            break

    X_train = pipeline.fit_transform(X[train_index])
    X_train = X_train.astype(float)
    X_test = pipeline.transform(X[test_index])
    X_test = X_test.astype(float)
    group_train = df['Flight Number'].values[train_index]
    group_test = df['Flight Number'].values[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    return X_train, y_train, group_train, X_test, y_test, group_test, train_index, test_index



df_clean = pd.read_csv('/home/felix/phd2/flights_vision/clean.csv')
df_dirty = pd.read_csv('/home/felix/phd2/flights_vision/dirty.csv')


X_train_clean, y_train_clean, group_train_clean, X_test_clean, y_test_clean, group_test_clean, train_index, test_index = df_to_ML_task(df_clean)
X_train_dirty, y_train_dirty, _, X_test_dirty, y_test_dirty, _, _, _ = df_to_ML_task(df_dirty, train_index, test_index)

#import matplotlib.pyplot as plt
#plt.hist(y_train_clean, bins=100)
#plt.show()

#y_train_clean

print(type(X_train_clean))
print(type(X_train_dirty))

#remove instances with missing labels
X_train_dirty_new = []
y_train_dirty_new = []
group_train_new = []
for i in range(len(X_train_dirty)):
    if not np.isnan(y_train_dirty[i]):
        X_train_dirty_new.append(X_train_dirty[i])
        y_train_dirty_new.append(y_train_dirty[i])
        group_train_new.append(group_train_clean[i])
X_train_dirty_new = np.array(X_train_dirty_new)

print(y_train_dirty_new)

scorer = autosklearn.metrics.make_scorer(
        'f1_score',
        sklearn.metrics.f1_score
    )

cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=1*60,
                                                  resampling_strategy=GroupKFold,
                                                  resampling_strategy_arguments={'n_splits': 10, 'groups': np.array(group_train_new)},
                                                       metric=scorer)
cls.fit(X_train_dirty_new.copy(), y_train_dirty_new.copy())
cls.refit(X_train_dirty_new.copy(), y_train_dirty_new.copy())
predictions = cls.predict(X_test_dirty.copy())
#print('dirty: ' + str(r2_score(y_test_clean, predictions)))
print('dirty: ' + str(f1_score(y_test_clean, predictions)))

cls_clean = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=1*60,
                                                  resampling_strategy=GroupKFold,
                                                  resampling_strategy_arguments={'n_splits': 10, 'groups': group_train_clean}, metric=scorer)
cls_clean.fit(X_train_clean.copy(), y_train_clean.copy())
cls_clean.refit(X_train_clean.copy(), y_train_clean.copy())
predictions = cls_clean.predict(X_test_clean.copy())
#print('clean: ' + str(r2_score(y_test_clean, predictions)))
print('clean: ' + str(f1_score(y_test_clean, predictions)))