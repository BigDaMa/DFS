import pandas as pd
import numpy as np
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from sklearn.metrics import make_scorer
from itertools import product
import copy
from immutabledict import immutabledict
import time
from hyperopt import STATUS_OK

def run_grid_search(pipeline, X_train, y_train, X_validation, y_validation, accuracy_scorer, sensitive_ids, min_fairness, min_accuracy, min_robustness, max_number_features, model_hyperparameters, start_time, avoid_robustness=False):
    fair_validation = None
    if type(sensitive_ids) != type(None):
        fair_validation = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_validation[:, sensitive_ids[0]])

    search_configs = [{}]
    if type(model_hyperparameters) != type(None):

        new_model_hyperparameters = {}
        for k, v in model_hyperparameters.items():
            new_model_hyperparameters['clf__' + k] = v
        print(new_model_hyperparameters)

        search_configs = [dict(zip(new_model_hyperparameters, v)) for v in product(*new_model_hyperparameters.values())]

    grid_results = {}

    for configuration in search_configs:
        new_pipeline = copy.deepcopy(pipeline)
        new_pipeline.set_params(**configuration)
        new_pipeline.fit(X_train, pd.DataFrame(y_train))

        validation_number_features = float(np.sum(new_pipeline.named_steps['selection']._get_support_mask())) / float(X_train.shape[1])
        validation_acc = accuracy_scorer(new_pipeline, X_validation, pd.DataFrame(y_validation))

        print("accuracy: " + str(validation_acc))

        validation_fair = 0.0
        if type(sensitive_ids) != type(None):
            validation_fair = 1.0 - fair_validation(new_pipeline, X_validation, pd.DataFrame(y_validation))

        validation_robust = 0.0
        if not avoid_robustness:
            validation_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_validation, y_test=y_validation,
                                                    model=new_pipeline.named_steps['clf'],
                                                    feature_selector=new_pipeline.named_steps['selection'],
                                                    scorer=accuracy_scorer)

        loss = 0.0
        if min_fairness > 0.0 and validation_fair < min_fairness:
            loss += (min_fairness - validation_fair) ** 2
        if min_accuracy > 0.0 and validation_acc < min_accuracy:
            loss += (min_accuracy - validation_acc) ** 2
        if min_robustness > 0.0 and validation_robust < min_robustness:
            loss += (min_robustness - validation_robust) ** 2
        if max_number_features < 1.0 and validation_number_features > max_number_features:
            loss += (validation_number_features - max_number_features) ** 2

        grid_results[immutabledict(configuration)] = {'loss': loss,
                                    'cv_fair': validation_fair,
                                    'cv_acc': validation_acc,
                                    'cv_robust': validation_robust,
                                    'cv_number_features': validation_number_features}

    # get minimum loss hyperparameter configuration
    min_loss = np.inf
    best_parameter_configuration = {}
    for k, v in grid_results.items():
        if min_loss > v['loss']:
            min_loss = v['loss']
            best_parameter_configuration = k

    print(best_parameter_configuration)

    best_pipeline = copy.deepcopy(pipeline)
    best_pipeline.set_params(**best_parameter_configuration)
    results = grid_results[best_parameter_configuration]
    results['model'] = best_pipeline
    results['time'] = time.time() - start_time
    results['status'] = STATUS_OK

    return results

