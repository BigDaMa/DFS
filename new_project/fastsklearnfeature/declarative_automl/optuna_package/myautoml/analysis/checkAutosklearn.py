import sklearn.metrics
from sklearn.metrics import f1_score
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
import autosklearn.classification
from autosklearn.experimental.askl2 import AutoSklearn2Classifier

test_holdout_dataset_ids = [1134, 1495, 41147, 316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]

#test_holdout_dataset_ids = [316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]



memory_budget = 8.0
privacy = None

results_dict = {}

scorer = autosklearn.metrics.make_scorer(
        'f1_score',
        sklearn.metrics.f1_score
    )

for test_holdout_dataset_id in test_holdout_dataset_ids:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data(test_holdout_dataset_id, randomstate=42)

    dynamic_approach = []
    static_approach = []

    for minutes_to_search in [10, 30, 60]:#range(1, 6):

        current_dynamic = []
        current_static = []

        search_time_frozen = minutes_to_search * 60

        for repeat in range(5):

            try:
                feat_type = []
                for c_i in range(len(categorical_indicator_hold)):
                    if categorical_indicator_hold[c_i]:
                        feat_type.append('Categorical')
                    else:
                        feat_type.append('Numerical')

                autosklearn_model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=search_time_frozen,
                                                                             ml_memory_limit=memory_budget*1000,
                                                                             metric=scorer,
                                                                             n_jobs=1,
                                                                             #ensemble_size=1,
                                                                             #initial_configurations_via_metalearning=0
                                                                             )
                autosklearn_model.fit(X_train_hold.copy(), y_train_hold.copy(), feat_type=feat_type)
                autosklearn_model.refit(X_train_hold.copy(), y_train_hold.copy())

                predictions = autosklearn_model.predict(X_test_hold.copy())
                result = f1_score(y_test_hold, predictions)

            except:
                result = 0

            print("test result: " + str(result))
            current_dynamic.append(result)

            print('dynamic: ' + str(current_dynamic))

        dynamic_approach.append(current_dynamic)
        static_approach.append(current_static)

        print('dynamic: ' + str(dynamic_approach))
        print('static: ' + str(static_approach))

        results_dict[test_holdout_dataset_id] = {}
        results_dict[test_holdout_dataset_id]['dynamic'] = dynamic_approach

        pickle.dump(results_dict, open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_auto_sklearn_more_time.p', 'wb+'))
