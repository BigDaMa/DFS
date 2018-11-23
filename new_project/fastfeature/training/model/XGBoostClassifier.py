from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np

class XGBoostClassifier:
    def __init__(self):
        pass

    def hyperparameter_optimization(self, train, train_target, feature_names, folds=10):
        cv_params = {'min_child_weight': [1, 3, 5],
                     'subsample': [0.7, 0.8, 0.9],
                     'max_depth': [3, 5, 7],
                     'n_estimators': [1, 10, 100, 1000]
                     }

        ind_params = {  # 'min_child_weight': 1, # we could optimize this: 'min_child_weight': [1, 3, 5]
            'learning_rate': 0.1,  # we could optimize this: 'learning_rate': [0.1, 0.01]
            'colsample_bytree': 0.8,
            'silent': 1,
            'seed': 0,
            'objective': 'binary:logistic',
            'n_jobs': 4
        }

        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring='f1', cv=folds, n_jobs=1)

        optimized_GBM.fit(train, train_target)

        print("all_results: " + str(optimized_GBM.cv_results_['mean_test_score']))
        print("best config: " + str(optimized_GBM.best_params_))
        print("best score: " + str(optimized_GBM.best_score_))

        our_params = ind_params.copy()
        our_params.update(optimized_GBM.best_params_)

        print("train shape: " + str(train.shape))
        print("featurenames shape: " + str(len(feature_names)))

        print("label fraction: " + str(np.sum(train_target)/ float(len(train_target))))


        xgdmat = xgb.DMatrix(train, train_target, feature_names=feature_names)
        self.model = xgb.train(our_params, xgdmat, verbose_eval=False)

        print(self.model.get_score(importance_type='gain'))


    def get_k_least_certain_tuples(self, full_matrix, k=10):
        probability_prediction = self.model.predict(full_matrix)
        certainty = np.absolute(probability_prediction - 0.5)

        print("average Certainty (max=0.5): " + str(np.mean(certainty)))

        sorted_ids = np.argsort(certainty)
        return sorted_ids[0:k]

