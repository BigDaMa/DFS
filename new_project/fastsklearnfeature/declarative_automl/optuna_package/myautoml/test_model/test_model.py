from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
import numpy as np
import autosklearn.regression
import autosklearn.classification
import autosklearn.pipeline.components.regression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

X_meta = pickle.load(open('/home/felix/phd2/picture_progress/uniform_sampling/test_model/felix_X_compare.p', "rb"))
y_meta = np.array(pickle.load(open('/home/felix/phd2/picture_progress/uniform_sampling/test_model/felix_y_compare.p', "rb")))
group_meta = np.array(pickle.load(open('/home/felix/phd2/picture_progress/uniform_sampling/test_model/felix_group_compare.p', "rb")))

print(X_meta.shape)

#import matplotlib.pyplot as plt
#plt.hist(y_meta, bins=100)
#plt.show()

gkf = GroupKFold(n_splits=2)
train_ids, test_ids = list(gkf.split(X_meta, y_meta, groups=group_meta))[0]
print(train_ids)

X_train = X_meta[train_ids]
y_train = y_meta[train_ids]
group_train = group_meta[train_ids]

X_test = X_meta[test_ids]
y_test = y_meta[test_ids]
group_test = group_meta[test_ids]

gkf = GroupKFold(n_splits=8)
cross_val = GridSearchCV(RandomForestRegressor(), param_grid={'n_estimators': [1000]}, cv=gkf, refit=True, scoring='r2', n_jobs=8)
cross_val.fit(X_train, y_train, groups=group_train)
model = cross_val.best_estimator_

predictions = model.predict(X_test)
print(r2_score(y_test, predictions))

plt.scatter(y_test, np.square(y_test - predictions))
plt.xlabel('Actual values')
plt.ylabel('Residuals')
plt.show()

plt.scatter(y_test, y_test - predictions)
plt.xlabel('Actual values')
plt.ylabel('Abs Residuals')
plt.show()

plt.scatter(y_test, predictions)
plt.xlabel('Actual values')
plt.ylabel('Predictions')
plt.show()


'''
from autosklearn.metrics import r2

cls = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=60*60*3,
                                                  resampling_strategy=GroupKFold,
                                                  resampling_strategy_arguments={'n_splits': 20, 'groups': np.array(group_train)},
                                                       metric=r2)
cls.fit(X_train.copy(), y_train.copy())
cls.refit(X_train.copy(), y_train.copy())

print(r2_score(y_test, cls.predict(X_test)))
'''
