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

X_meta = pickle.load(open('/home/felix/phd2/picture_progress/uniform_sampling/test_model/felix_X_compare.p', "rb"))
y_meta = np.array(pickle.load(open('/home/felix/phd2/picture_progress/uniform_sampling/test_model/felix_y_compare.p', "rb")))
group_meta = np.array(pickle.load(open('/home/felix/phd2/picture_progress/uniform_sampling/test_model/felix_group_compare.p', "rb")))

print(X_meta.shape)



gkf = GroupKFold(n_splits=20)
cross_val = GridSearchCV(RandomForestRegressor(), param_grid={'n_estimators': [1000]}, cv=gkf, refit=True, scoring='r2', n_jobs=8)
cross_val.fit(X_meta, y_meta, groups=group_meta)
model = cross_val.best_estimator_

print(cross_val.best_score_)

