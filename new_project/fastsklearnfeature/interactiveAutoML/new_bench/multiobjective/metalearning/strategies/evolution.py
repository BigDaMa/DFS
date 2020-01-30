import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List, Dict, Set
from fastsklearnfeature.interactiveAutoML.CreditWrapper import run_pipeline
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
import matplotlib.pyplot as plt
from fastsklearnfeature.interactiveAutoML.Runner import Runner
import copy

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
from pymoo.factory import get_problem, get_termination
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
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection
from sklearn.model_selection import train_test_split

from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.configuration.Config import Config
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.my_global_variable as my_global_variable

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_data
from pymoo.model.termination import Termination

import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.cheating_global as cheating_global
import random


def evolution(X_train, X_test, y_train, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness = 0.0, min_robustness = 0.0, max_number_features = None, max_search_time=np.inf, cv_splitter = None):

	hash = str(random.getrandbits(128)) + str(time.time())
	cheating_global.successfull_result[hash] = {}

	start_time = time.time()

	auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
	fair_train = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train[:, sensitive_ids[0]])
	fair_test = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_test[:, sensitive_ids[0]])


	def f_clf1(mask):
		model = Pipeline([
			('selection', MaskSelection(mask)),
			('clf', LogisticRegression())
		])
		return model

	# define an objective function
	def objective(features):
		if 'time' in cheating_global.successfull_result[hash]:
			return [0.0, 0.0, 0.0, 0.0]

		model = f_clf1(features)

		robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train, y=y_train, model=clf,
									feature_selector=model.named_steps['selection'], scorer=auc_scorer)

		cv = GridSearchCV(model, param_grid={'clf__C': [1.0]}, cv=cv_splitter,
						  scoring={'AUC': auc_scorer, 'Fairness': fair_train, 'Robustness': robust_scorer},
						  refit=False)
		cv.fit(X_train, pd.DataFrame(y_train))
		cv_acc = cv.cv_results_['mean_test_AUC'][0]
		cv_fair = 1.0 - cv.cv_results_['mean_test_Fairness'][0]
		cv_robust = 1.0 - cv.cv_results_['mean_test_Robustness'][0]

		cv_number_features = float(np.sum(model.named_steps['selection']._get_support_mask())) / float(
			len(model.named_steps['selection']._get_support_mask()))

		cv_simplicity = 1.0 - cv_number_features


		#check constraints for test set
		if cv_fair >= min_fairness and cv_acc >= min_accuracy and cv_robust >= min_robustness and cv_number_features <= max_number_features:
			model.fit(X_train, pd.DataFrame(y_train))

			test_acc = 0.0
			if min_accuracy > 0.0:
				test_acc = auc_scorer(model, X_test, pd.DataFrame(y_test))
			test_fair = 0.0
			if min_fairness > 0.0:
				test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
			test_robust = 0.0
			if min_robustness > 0.0:
				test_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_test, y_test=y_test, model=model.named_steps['clf'], feature_selector=model.named_steps['selection'], scorer=auc_scorer)

			if test_fair >= min_fairness and test_acc >= min_accuracy and test_robust >= min_robustness:
				print('fair: ' + str(min(cv_fair, test_fair)) + ' acc: ' + str(min(cv_acc, test_acc)) + ' robust: ' + str(min(test_robust, cv_robust)) + ' k: ' + str(cv_number_features))
				cheating_global.successfull_result[hash]['time'] = time.time() - start_time

		return [cv_acc, cv_fair, cv_robust, cv_simplicity]
	
	class MyProblem(Problem):
	
		def __init__(self):
			number_objectives = 0
			if min_accuracy > 0.0:
				number_objectives +=1
			if min_fairness > 0.0:
				number_objectives +=1
			if min_robustness > 0.0:
				number_objectives +=1
			if max_number_features < 1.0:
				number_objectives +=1

			super().__init__(n_var=X_train.shape[1],
								 n_obj=number_objectives,
								 n_constr=0,
								 xl=0, xu=1, type_var=anp.bool)
	
		def _evaluate(self, x, out, *args, **kwargs):
			accuracy_batch = []
			fairness_batch = []
			robustness_batch = []
			simplicity_batch = []
	
			for i in range(len(x)):
				results = objective(x[i])
				accuracy_batch.append(results[0]*-1)#accuracy
				fairness_batch.append(results[1]*-1)#fairness
				robustness_batch.append(results[2]*-1)#robustness
				simplicity_batch.append(results[3]*-1)  # simplicity

			##objectives
			objectives = []
			if min_accuracy > 0.0:
				objectives.append(accuracy_batch)
			if min_fairness > 0.0:
				objectives.append(fairness_batch)
			if min_robustness > 0.0:
				objectives.append(robustness_batch)
			if max_number_features < 1.0:
				objectives.append(simplicity_batch)
	
			out["F"] = anp.column_stack(objectives)
	

	problem = MyProblem()
	
	
	population_size = 100
	cross_over_rate = 0.9
	algorithm = NSGA2(pop_size=population_size,
					  sampling=get_sampling("bin_random"),
					  crossover=get_crossover('bin_one_point'),#get_crossover("bin_hux"),#get_crossover("bin_two_point"),
					  mutation=BinaryBitflipMutation(1.0 / X_train.shape[1]),
					  elimate_duplicates=True,
					  #n_offsprings= cross_over_rate * population_size
					  )
		
	
	class MyTermination(Termination):
	
		def __init__(self, start_time=None, time_limit=None) -> None:
			super().__init__()
			self.start_time = start_time
			self.time_limit = time_limit
	
		def _do_continue(self, algorithm):
			if 'time' in cheating_global.successfull_result[hash] or (time.time() - self.start_time) > self.time_limit:
				return False
	
			return True

	minimize(problem=problem, algorithm=algorithm, termination=MyTermination(max_search_time), disp=False)

	runtime = time.time() - start_time
	success = 'time' in cheating_global.successfull_result[hash]
	del cheating_global.successfull_result[hash]
	return runtime, success