import pickle
from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.MyAutoMLProcess import MyAutoML
import optuna
import time
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_GenerationTree import SpaceGenerator
import copy
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import plot_most_important_features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import optimize_accuracy_under_constraints2
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import utils_run_AutoML
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names
from anytree import RenderTree

my_scorer=make_scorer(f1_score)


test_holdout_dataset_ids = [1134, 1495, 41147, 316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]

privacy = None

model_compare = pickle.load(open('/home/felix/phd2/picture_progress/al_only/my_great_model_compare.p', "rb"))
model_success = pickle.load(open('/home/felix/phd2/picture_progress/al_only/my_great_model_success.p', "rb"))
model_weights = pickle.load(open('/home/felix/phd2/picture_progress/weights/my_great_model_weights.p', "rb"))

my_list_constraints = ['global_search_time_constraint',
                       'global_evaluation_time_constraint',
                       'global_memory_constraint',
                       'global_cv',
                       'global_number_cv',
                       'privacy',
                       'hold_out_fraction',
                       'sample_fraction',
                       'training_time_constraint',
                       'inference_time_constraint',
                       'pipeline_size_constraint']

_, feature_names = get_feature_names(my_list_constraints)

results_dict = {}

for test_holdout_dataset_id in test_holdout_dataset_ids:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data(test_holdout_dataset_id, randomstate=42)
    metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

    #plot_most_important_features(model, feature_names, k=len(feature_names))

    dynamic_approach = []
    static_approach = []

    minutes_to_search = 5
    memory_budget = 8

    for memory_size in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:

        current_dynamic = []
        current_static = []

        search_time_frozen = minutes_to_search * 60

        for repeat in range(5):

            study_prune = optuna.create_study(direction='maximize')
            study_prune.optimize(lambda trial: optimize_accuracy_under_constraints2(trial=trial,
                                                                                   metafeature_values_hold=metafeature_values_hold,
                                                                                   search_time=search_time_frozen,
                                                                                   model_compare=model_compare,
                                                                                   model_success=model_success,
                                                                                   memory_limit=memory_size,
                                                                                   comparison_weight=0.0,
                                                                                   tune_space=True
                                                                                   ), n_trials=500, n_jobs=4)

            space = study_prune.best_trial.user_attrs['space']



            for pre, _, node in RenderTree(space.parameter_tree):
                if node.status == True:
                    print("%s%s" % (pre, node.name))


            try:
                result, search = utils_run_AutoML(study_prune.best_trial,
                                                             X_train=X_train_hold,
                                                             X_test=X_test_hold,
                                                             y_train=y_train_hold,
                                                             y_test=y_test_hold,
                                                             categorical_indicator=categorical_indicator_hold,
                                                             my_scorer=my_scorer,
                                                             search_time=search_time_frozen,
                                                             memory_limit=memory_size
                                             )

                from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import show_progress
                #show_progress(search, X_test_hold, y_test_hold, my_scorer)

                print("test result: " + str(result))
                current_dynamic.append(result)
            except:
                current_dynamic.append(0.0)

            print('dynamic: ' + str(current_dynamic))
            print('static: ' + str(current_static))


            gen_new = SpaceGenerator()
            space_new = gen_new.generate_params()
            for pre, _, node in RenderTree(space_new.parameter_tree):
                if node.status == True:
                    print("%s%s" % (pre, node.name))

            try:
                search = MyAutoML(n_jobs=1,
                                  time_search_budget=search_time_frozen,
                                  space=space_new,
                                  evaluation_budget=int(0.1 * search_time_frozen),
                                  main_memory_budget_gb=memory_size,
                                  hold_out_fraction=0.33
                                  )

                best_result = search.fit(X_train_hold, y_train_hold, categorical_indicator=categorical_indicator_hold, scorer=my_scorer)


                test_score = my_scorer(search.get_best_pipeline(), X_test_hold, y_test_hold)
            except:
                test_score = 0.0
            current_static.append(test_score)

            print("result: " + str(best_result) + " test: " + str(test_score))

            print('dynamic: ' + str(current_dynamic))
            print('static: ' + str(current_static))

        dynamic_approach.append(current_dynamic)
        static_approach.append(current_static)

        print('dynamic: ' + str(dynamic_approach))
        print('static: ' + str(static_approach))

        results_dict[test_holdout_dataset_id] = {}
        results_dict[test_holdout_dataset_id]['dynamic'] = dynamic_approach
        results_dict[test_holdout_dataset_id]['static'] = static_approach

        pickle.dump(results_dict, open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_memory_size_constraint.p', 'wb+'))

