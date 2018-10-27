import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

class XGBoostClassifier:
    def __init__(self, number_classes, score):
        self.number_classes = number_classes
        self.score = score

        if number_classes > 2:
            self.objective = 'multi:softprob'
        else:
            self.objective = 'binary:logistic'


    def fit(self, train, test, best_params=None):
        if type(best_params) == type(None):
            best_params = {}
            best_params['objective'] = self.objective
            best_params['nthread'] = 4
        model = xgb.XGBClassifier(**best_params)
        model.fit(train, test)
        return model


    def run_cross_validation(self, train, train_target, folds):
        cv_params = {'min_child_weight': [1, 3, 5],
                     'subsample': [0.7, 0.8, 0.9, 1],
                     'max_depth': [3, 5, 7],
                     'colsample_bytree': [0.8, 1],
                     'n_estimators': [100, 1000],
                     }

        ind_params = {  # 'min_child_weight': 1, # we could optimize this: 'min_child_weight': [1, 3, 5]
            'learning_rate': 0.1,  # we could optimize this: 'learning_rate': [0.1, 0.01]
            'silent': 1,
            'seed': 0,
            'objective': self.objective,
            'n_jobs': 4,
            'nthread': 4
        }

        scorer = self.score.get_scorer()
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring=scorer, cv=folds, n_jobs=1, verbose=0)

        optimized_GBM.fit(train, train_target)

        # print "best scores: " + str(optimized_GBM.grid_scores_)

        our_params = ind_params.copy()
        our_params.update(optimized_GBM.best_params_)

        return our_params
