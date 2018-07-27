from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np

class XGBoost_Model():
    def __init__(self):
        self.cv_params = {'min_child_weight': [1, 3, 5],
                     'subsample': [0.7, 0.8, 0.9],
                     'max_depth': [3, 5, 7]}

        self.ind_params = {  # 'min_child_weight': 1, # we could optimize this: 'min_child_weight': [1, 3, 5]
            'learning_rate': 0.1,  # we could optimize this: 'learning_rate': [0.1, 0.01]
            # 'max_depth': 3, # we could optimize this: 'max_depth': [3, 5, 7]
            # 'n_estimators': 1000, # we choose default 100
            'colsample_bytree': 0.8,
            'silent': 0, #1
            'seed': 0,
            'objective': 'multi:softprob',
            'n_jobs': '4'
        }

        self.model = None
        self.classes = []

    def get_classes(self):
        return self.model.classes_


    def optimize_hyperparameters(self, x, y, folds, jobs=4):
        clf = GridSearchCV(xgb.XGBClassifier(**self.ind_params), self.cv_params, cv=folds, n_jobs=jobs, verbose=4)
        clf.fit(x, y)

        our_params = self.ind_params.copy()
        our_params.update(clf.best_params_)

        self.best_params = our_params


    def train(self, x, y, params=None):
        if len(self.classes) == 0:
            self.classes = np.unique(y)

        parameters = params
        if params == None:
            parameters = self.best_params
        parameters.update({'num_class': len(self.classes)})

        self.model = xgb.XGBClassifier(**parameters)
        self.model.fit(x, y)


    def partial_train(self, x, y, params=None):
        parameters = params
        if params == None:
            parameters = self.best_params

        if len(self.classes) == 0:
            self.classes = np.unique(y)
            parameters.update({'num_class': len(self.classes)})
            self.model = xgb.XGBClassifier(**parameters)

            self.model_current = self.model.fit(x, y)


        else:
            parameters.update({'num_class': len(self.classes)})
            parameters.update({'process_type': 'update',
                           'updater': 'refresh',
                           'refresh_leaf': True})

            self.model_current = self.model.fit(x, y, xgb_model=self.model_current)


    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)[0]