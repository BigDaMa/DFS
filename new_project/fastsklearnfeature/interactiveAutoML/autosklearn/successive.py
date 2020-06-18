import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
import openml
#from tpot import TPOTClassifier
import time

def get_smac_object_callback(budget_type):
    def get_smac_object(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.intensification.successive_halving import SuccessiveHalving
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario

        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
            smac_run_id=seed if not scenario_dict['shared-model'] else '*',
        )
        scenario = Scenario(scenario_dict)
        if len(metalearning_configurations) > 0:
            default_config = scenario.cs.get_default_configuration()
            initial_configurations = [default_config] + metalearning_configurations
        else:
            initial_configurations = None
        rh2EPM = RunHistory2EPM4LogCost

        ta_kwargs['budget_type'] = budget_type

        return SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=seed,
            intensifier=SuccessiveHalving,
            intensifier_kwargs={
                'initial_budget': 10.0,
                'max_budget': 100,
                'eta': 2,
                'min_chall': 1},
            )
    return get_smac_object

auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)


task = openml.tasks.get_task(31)  # download the OpenML task
X, y = task.get_X_and_y()  # get the data

X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

'''
t = ConstructionTransformer(c_max=2, scoring=auc, n_jobs=4, model=LogisticRegression(), cv=3)
t.fit(X_train, y_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
'''

print(X_train)
print(X_train.shape)

roc_auc_scorer = autosklearn.metrics.make_scorer(
        name="roc_auc",
        score_func=roc_auc_score,
        optimum=1,
        greater_is_better=True,
        needs_threshold=True
    )


start = time.time()
#automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60*5, n_jobs=4, ensemble_size=1, resampling_strategy='cv', resampling_strategy_arguments={'folds': 5})



automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60*5,
        per_run_time_limit=5,
        tmp_folder='/tmp/autosklearn_sh_example_tmp',
        output_folder='/tmp/autosklearn_sh_example_out',
        disable_evaluator_output=False,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67},
        include_estimators=['extra_trees', 'gradient_boosting', 'random_forest', 'sgd',
                            'passive_aggressive'],
        include_preprocessors=['no_preprocessing'],
        get_smac_object_callback=get_smac_object_callback('iterations'),
    )


automl.fit(X_train, y_train, metric=roc_auc_scorer)
y_hat = automl.predict(X_test)

print(automl.sprint_statistics())

print("AUC score: ", auc(automl, X_test, y_test))
#print(automl.cv_results_)
print(time.time() - start)

print(automl.show_models())


'''
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=4)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
'''