import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.interactiveAutoML.feature_selection.ForwardSequentialSelection import ForwardSequentialSelection
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List, Dict, Set
from fastsklearnfeature.interactiveAutoML.CreditWrapper import run_pipeline
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
import matplotlib.pyplot as plt
from fastsklearnfeature.interactiveAutoML.Runner import Runner
import copy
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time

from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.metrics import loss_sensitivity
from art.metrics import empirical_robustness
from sklearn.pipeline import FeatureUnion


from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from art.classifiers import XGBoostClassifier, LightGBMClassifier, SklearnClassifier
from art.attacks import HopSkipJump


from fastsklearnfeature.interactiveAutoML.feature_selection.RunAllKBestSelection import RunAllKBestSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fairness_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import robustness_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection
from sklearn.model_selection import train_test_split

from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import unit_test_score

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test

from fastsklearnfeature.configuration.Config import Config

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif


from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score

from sklearn.ensemble import RandomForestRegressor

import diffprivlib.models as models
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.weighted_ranking import weighted_ranking

#static constraints: fairness, number of features (absolute and relative), robustness, privacy, accuracy

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_data

X_train, X_test, y_train, y_test, names, sensitive_ids = get_data(data_path='/heart/dataset_53_heart-statlog.csv',
																  continuous_columns = [0,3,4,7,9,10,11],
																  sensitive_attribute = "sex",
																  limit=250)




#run on tiny sample
X_train_tiny, _, y_train_tiny, _ = train_test_split(X_train, y_train, train_size=100, random_state=42)

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
fair_train_tiny = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train_tiny[:, sensitive_ids[0]])

time_limit = 60 * 10

meta_classifier = RandomForestClassifier(n_estimators=100)

cv_splitter = StratifiedKFold(5, random_state=42)


def objective(hps):
	cv_k = 1.0
	cv_privacy = hps['privacy']
	model = LogisticRegression()
	if type(cv_privacy) == type(None):
		cv_privacy = X_train_tiny.shape[0]
	else:
		model = models.LogisticRegression(epsilon=cv_privacy)

	robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train_tiny, y=y_train_tiny, model=model,
								feature_selector=None, scorer=auc_scorer)


	cv = GridSearchCV(model, param_grid={'C': [1.0]}, scoring={'AUC': auc_scorer, 'Fairness': fair_train_tiny, 'Robustness': robust_scorer}, refit=False, cv=cv_splitter)
	cv.fit(X_train_tiny, pd.DataFrame(y_train_tiny))
	cv_acc = cv.cv_results_['mean_test_AUC'][0]
	cv_fair = 1.0 - cv.cv_results_['mean_test_Fairness'][0]
	cv_robust = 1.0 - cv.cv_results_['mean_test_Robustness'][0]

	#construct feature vector
	feature_list = []
	#user-specified constraints
	feature_list.append(hps['accuracy'])
	feature_list.append(hps['fairness'])
	feature_list.append(hps['k'])
	feature_list.append(hps['robustness'])
	feature_list.append(cv_privacy)
	#differences to sample performance
	feature_list.append(cv_acc - hps['accuracy'])
	feature_list.append(cv_fair - hps['fairness'])
	feature_list.append(cv_k - hps['k'])
	feature_list.append(cv_robust - hps['robustness'])
	#privacy constraint is always satisfied => difference always zero => constant => unnecessary

	#metadata features
	#feature_list.append(X_train.shape[0])#number rows
	#feature_list.append(X_train.shape[1])#number columns

	features = np.array(feature_list)

	#predict the best model and calculate uncertainty

	loss = 0
	try:
		proba_predictions = meta_classifier.predict_proba([features])[0]
		proba_predictions = np.sort(proba_predictions)
		uncertainty = 1 - (proba_predictions[-1] - proba_predictions[-2])
		loss = -1 * uncertainty  # we want to maximize uncertainty
	except:
		pass

	return {'loss': loss, 'status': STATUS_OK, 'features': features}



space = {
		 'k': hp.choice('k_choice',
						[
							(1.0),
							(hp.uniform('k_specified', 0, 1))
						]),
		 'accuracy': hp.choice('accuracy_choice',
						[
							(0.0),
							(hp.uniform('accuracy_specified', 0, 1))
						]),
         'fairness': hp.choice('fairness_choice',
						[
							(0.0),
							(hp.uniform('fairness_specified', 0, 1))
						]),
		 'privacy': hp.choice('privacy_choice',
						[
							(None),
							(hp.lognormal('privacy_specified', 0, 1))
						]),
		 'robustness': hp.choice('robustness_choice',
						[
							(0.0),
							(hp.uniform('robustness_specified', 0, 1))
						]),
		}

trials = Trials()
i = 1
while True:
	fmin(objective, space=space, algo=tpe.suggest, max_evals=i, trials=trials)
	i += 1

	#break, once convergence tolerance is reached and generate new dataset
	if trials.trials[-1]['result']['loss'] == 0 or i >=100:
		most_uncertain_f = trials.trials[-1]['misc']['vals']
		#print(most_uncertain_f)

		min_accuracy = 0.0
		if most_uncertain_f['accuracy_choice'][0]:
			min_accuracy = most_uncertain_f['accuracy_specified'][0]
		min_fairness = 0.0
		if most_uncertain_f['fairness_choice'][0]:
			min_fairness = most_uncertain_f['fairness_specified'][0]
		min_robustness = 0.0
		if most_uncertain_f['robustness_choice'][0]:
			min_robustness = most_uncertain_f['robustness_specified'][0]
		max_number_features = X_train.shape[1]
		if most_uncertain_f['k_choice'][0]:
			max_number_features = int(most_uncertain_f['k_specified'][0] * X_train.shape[1])


		# Execute each search strategy with a given time limit (in parallel)
		# maybe run multiple times to smooth stochasticity

		model = LogisticRegression()
		if most_uncertain_f['privacy_choice'][0]:
			model = models.LogisticRegression(epsilon=most_uncertain_f['privacy_specified'][0])

		#define rankings
		rankings = [variance, chi2_score_wo] #simple rankings
		rankings.append(partial(model_score, estimator=ExtraTreesClassifier(n_estimators=1000))) #accuracy ranking
		rankings.append(partial(robustness_score, model=model, scorer=auc_scorer)) #robustness ranking
		rankings.append(partial(fairness_score, estimator=ExtraTreesClassifier(n_estimators=1000), sensitive_ids=sensitive_ids)) #fairness ranking

		runtime, success = weighted_ranking(X_train, X_test, y_train, y_test, names, sensitive_ids,
						 ranking_functions=rankings,
						 clf=model,
						 min_accuracy=min_accuracy,
						 min_fairness=min_fairness,
						 min_robustness=min_robustness,
						 max_number_features=max_number_features,
						 max_search_time=time_limit,
						 cv_splitter=cv_splitter)

		print("Runtime: " + str(runtime))
		print("Success: " + str(success))



		# then add runs to training data
		break


#train other prediction tasks:
#will it satisfy the constraints
# what is the runtime
# how well does it with respect to all constraints
# what is the expected k that the best selection has