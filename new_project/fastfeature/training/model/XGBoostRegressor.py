from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

from xgboost import plot_tree

class XGBoostRegressor:
    def __init__(self):
        pass

    def hyperparameter_optimization(self, train, train_target, feature_names, folds=10):
        cv_params = {'min_child_weight': [1, 3, 5],
                     'subsample': [0.7, 0.8, 0.9],
                     'max_depth': [3, 5, 7],
                     'n_estimators': [10, 100, 1000]
                     }

        ind_params = {  # 'min_child_weight': 1, # we could optimize this: 'min_child_weight': [1, 3, 5]
            'learning_rate': 0.1,  # we could optimize this: 'learning_rate': [0.1, 0.01]
            'colsample_bytree': 0.8,
            #'silent': 1,
            'seed': 0,
            'objective': 'reg:linear',
            'n_jobs': 4
        }

        optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params),
                                     cv_params,
                                     scoring='neg_mean_squared_error', cv=folds, n_jobs=1)

        optimized_GBM.fit(train, train_target)

        print(optimized_GBM.cv_results_['mean_test_score'])

        # print "best scores: " + str(optimized_GBM.grid_scores_)

        our_params = ind_params.copy()
        our_params.update(optimized_GBM.best_params_)

        xgdmat = xgb.DMatrix(train, train_target, feature_names=feature_names)
        model = xgb.train(our_params, xgdmat, verbose_eval=False)

        print(model.get_score(importance_type='gain'))

