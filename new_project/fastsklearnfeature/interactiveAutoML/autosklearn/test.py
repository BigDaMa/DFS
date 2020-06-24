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
import numpy as np
from fastsklearnfeature.interactiveAutoML.autosklearn.SFFS import parallel
from fastsklearnfeature.interactiveAutoML.autosklearn.SFFS_construction import parallel_construct
from fastsklearnfeature.interactiveAutoML.autosklearn.SFFS_extend import parallel_extend
from fastsklearnfeature.candidate_generation.feature_space.division import get_onehot_and_imputation
from fastsklearnfeature.interactiveAutoML.autosklearn.SFFS_back import parallel_back
from fastsklearnfeature.interactiveAutoML.autosklearn.SFFS_RFE import parallel_rfe
from fastsklearnfeature.interactiveAutoML.autosklearn.SFFS_size import parallel_size
from fastsklearnfeature.interactiveAutoML.autosklearn.SFFS_run import parallel_run

import warnings

warnings.filterwarnings("ignore")


auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)


log_it = {}

#for data_id in [31, 179, 1464, 37, 50, 334, 3, 1480, 15]: #3 fails
for data_id in [31]: #3 fails

    log_it[data_id] = {}

    dataset = openml.datasets.get_dataset(data_id)
    #dataset = openml.datasets.get_dataset(31)
    #dataset = openml.datasets.get_dataset(179)
    #dataset = openml.datasets.get_dataset(1464)
    #dataset = openml.datasets.get_dataset(37)
    #dataset = openml.datasets.get_dataset(50)
    #dataset = openml.datasets.get_dataset(334)
    #dataset = openml.datasets.get_dataset(3)
    #dataset = openml.datasets.get_dataset(1462) #does not work
    #dataset = openml.datasets.get_dataset(1480)
    #dataset = openml.datasets.get_dataset(4135) # needs more data
    #dataset = openml.datasets.get_dataset(15)
    #dataset = openml.datasets.get_dataset(40536) # does not work
    #print(dataset.features)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    if len(X) > 1000:
        X = X[0:1000]
        y = y[0:1000]

    X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)



    start_construction = time.time()

    t = ConstructionTransformer(c_max=3, epsilon=-np.inf, scoring=auc, n_jobs=4, model=LogisticRegression(), cv=2, feature_names=attribute_names, feature_is_categorical=categorical_indicator,parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'], 'class_weight': ['balanced'], 'max_iter': [10], 'multi_class':['auto']})
    #t = ConstructionTransformer(c_max=3, epsilon=0, scoring=auc, n_jobs=4, model=LogisticRegression(), cv=10, feature_names=attribute_names, feature_is_categorical=categorical_indicator)

    #t = ConstructionTransformer(c_max=3, epsilon=0, scoring=auc, n_jobs=4, model=LogisticRegression(), cv=5, feature_names=attribute_names, feature_is_categorical=categorical_indicator)

    #t = ConstructionTransformer(c_max=3, epsilon=0, scoring=auc, n_jobs=4, model=LogisticRegression(), cv=5)
    #t.fit(X_train[0:500], y_train[0:500])
    t.fit(X_train, y_train)

    log_it[data_id]['construction_time'] = time.time() - start_construction


    #print(t.numeric_features)

    #X_train, X_test = parallel(X_train, y_train,X_test=X_test, y_test=y_test, floating=True, max_number_features=20, feature_generator=t, folds=10, number_cvs=1)

    #X_train, X_test = parallel_extend(X_train, y_train, X_test=X_test, y_test=y_test, floating=True, max_number_features=50, feature_generator=t, folds=5, number_cvs=1)

    #X_train, X_test = parallel_construct(X_train, y_train,X_test=X_test, y_test=y_test, floating=True, construction_floating=True, max_number_features=10, feature_generator=t, folds=5, number_cvs=1)

    #X_train, X_test = parallel_back(X_train, y_train, X_test=X_test, y_test=y_test, floating=True, max_number_features=50, feature_generator=t, folds=5, number_cvs=1)

    X_train, X_test = parallel_rfe(X_train, y_train, X_test=X_test, y_test=y_test, floating=True,max_number_features=2000, feature_generator=t, folds=5, number_cvs=1)

    #X_train, X_test = parallel_size(X_train, y_train, X_test=X_test, y_test=y_test, floating=True, max_number_features=2000, feature_generator=t, folds=5, number_cvs=1)

    #log_it[data_id]['hpo_time'], log_it[data_id]['test_auc'] = parallel_run(X_train, y_train, X_test=X_test, y_test=y_test, feature_generator=t, folds=5, number_cvs=1)

    print(log_it)

    '''

    roc_auc_scorer = autosklearn.metrics.make_scorer(
            name="roc_auc",
            score_func=roc_auc_score,
            optimum=1,
            greater_is_better=True,
            needs_threshold=True
        )


    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60*60,
                                                              n_jobs=4,
                                                              resampling_strategy='cv',
                                                              resampling_strategy_arguments={'folds': 10},
                                                              ensemble_size=1
                                                              )

    feat_type = ['Categorical' if ci else 'Numerical' for ci in categorical_indicator]
    automl.fit(X_train, y_train, metric=roc_auc_scorer, feat_type=feat_type)
    automl.refit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("data: "  + str(data_id) + " AUC score: " + str(auc(automl, X_test, y_test)) + '\n')
    #print(automl.show_models())




    #here implement a highly optimized - parallelized SFS until N/2 features (or twice then original or just as original) then choose optimal features
    #first try simple SFFS to check whether we get really nice accuracy
    #if we get really nice accuracy, we can start to think how to optimize
    #optimistic execution
    
    '''


    '''
    
    from autofeat import AutoFeatClassifier
    afreg = AutoFeatClassifier(verbose=1, feateng_steps=3)
    # fit autofeat on less data, otherwise ridge reg model with xval will overfit on new features
    X_train = afreg.fit_transform(X_train, y_train)
    X_test = afreg.transform(X_test)
    print("autofeat new features:", len(afreg.new_feat_cols_))
    '''

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
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60*10, n_jobs=4, ensemble_size=1,
                                                              include_estimators= ['liblinear_svc'], #only simple classifiers
                                                              include_preprocessors= ['select_percentile_classification'],
                                                              resampling_strategy='cv', resampling_strategy_arguments={'folds': 10},
                                                              initial_configurations_via_metalearning=0) # only feature selection as preprocessing
    
    
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60*20, n_jobs=4, ensemble_size=1,
                                                              include_estimators= ['sgd', 'k_nearest_neighbors', 'passive_aggressive', 'decision_tree', 'liblinear_svc', 'libsvm_svc'], #only simple classifiers
                                                              #include_preprocessors= ['select_percentile_classification', 'polynomial'],
                                                              include_preprocessors= ['select_percentile_classification'],
                                                              resampling_strategy='cv', resampling_strategy_arguments={'folds': 10},
                                                              initial_configurations_via_metalearning=0) # only feature selection as preprocessing
    
    
    
    feat_type = ['Categorical' if ci else 'Numerical'
                     for ci in categorical_indicator]
    automl.fit(X_train, y_train, metric=roc_auc_scorer, feat_type=feat_type)
    
    automl.fit(X_train, y_train, metric=roc_auc_scorer)
    automl.refit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("AUC score: ", auc(automl, X_test, y_test))
    #print(automl.cv_results_)
    print(time.time() - start)
    
    print(automl.show_models())
    
    
    
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=4)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
'''