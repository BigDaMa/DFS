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



n_estimators = 5

X_train, X_test, y_train, y_test, names, sensitive_ids = get_data(data_path='/heart/dataset_53_heart-statlog.csv',
																  continuous_columns = [0,3,4,7,9,10,11],
																  sensitive_attribute = "sex",
																  limit=250)





start_time = time.time()

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
fair_train = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train[:, sensitive_ids[0]])
fair_test = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_test[:, sensitive_ids[0]])


history = []

min_accuracy = 1.00
min_fairness = 1.00
min_robustness = 0.0
privacy_epsilon = None
max_number_features = X_train.shape[1]


min_avg_model_accuracy = 0.0 #does not really make sense


# define an objective function
def objective(features):
	model = Pipeline([
		('selection', MaskSelection(features)),
		('clf', LogisticRegression())
	])

	robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train, y=y_train,
								feature_selector=model.named_steps['selection'])
	robust_scorer_test = make_scorer(robust_score_test, greater_is_better=True, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
								feature_selector=model.named_steps['selection'])

	ncv = 5

	cv_acc = np.mean(
		cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(ncv, random_state=42), scoring=auc_scorer))
	cv_fair = 1.0 - np.mean(cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(ncv, random_state=42), scoring=fair_train))
	cv_robust = 1.0 - np.mean(cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(ncv, random_state=42), scoring=robust_scorer))
	#cv_robust = 1.0

	print('cv acc: ' + str(cv_acc) + ' cv fair: ' + str(cv_fair) + ' cv robust: ' + str(cv_robust))

	model.fit(X_train, pd.DataFrame(y_train))
	test_acc = auc_scorer(model, X_test, pd.DataFrame(y_test))
	test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
	test_robust = 1.0 - robust_scorer_test(model, X_test, pd.DataFrame(y_test))

	simplicity = -1 * np.sum(features)


	my_global_variable.satisfied_constraints.append([min(test_acc, cv_acc), min(test_fair, cv_fair)])
	my_global_variable.times.append(time.time() - start_time)
	my_global_variable.iterations.append(my_global_variable.current_iteration)


	my_global_variable.current_iteration += 1


	return [cv_acc, cv_fair, cv_robust, simplicity]

class MyProblem(Problem):

	def __init__(self):
		super().__init__(n_var=X_train.shape[1],
                         n_obj=2,
                         n_constr=0,
						 xl=0, xu=1, type_var=anp.bool)

	def _evaluate(self, x, out, *args, **kwargs):
		f1_all = []
		f2_all = []
		f3_all = []
		f4_all = []

		print(x)

		for i in range(len(x)):
			results = objective(x[i])
			f1_all.append(results[0]*-1)#accuracy
			f2_all.append(results[1]*-1)#fairness
			f3_all.append(results[2]*-1)#robustness
			f4_all.append(results[3] * -1)  # simplicity

			#g1_all.append(c1)


		out["F"] = anp.column_stack([f1_all, f2_all])



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
'''
algorithm = NSGA2(pop_size=population_size,
				  sampling=get_sampling("bin_random"),
				  crossover=get_crossover("bin_hux"),#get_crossover("bin_two_point"),
				  mutation=get_mutation("bin_bitflip"),
				  elimate_duplicates=True)
'''

from pymoo.model.termination import Termination
class MyTermination(Termination):

	def __init__(self, start_time=None, time_limit=None) -> None:
		super().__init__()
		self.start_time = start_time
		self.time_limit = time_limit

	def _do_continue(self, algorithm):
		if (time.time() - self.start_time) > self.time_limit:
			return False

		return True


class MyTermination2(Termination):

	def __init__(self, max_iterations=None) -> None:
		super().__init__()
		self.max_iterations = max_iterations

	def _do_continue(self, algorithm):
		print('tes: ' + str(my_global_variable.current_iteration) + ' _ ' + str(self.max_iterations))
		if my_global_variable.current_iteration > self.max_iterations:
			return False

		return True

'''
res = minimize(problem=problem,
               algorithm=algorithm, termination=MyTermination(start_time, 60 * 1),
               disp=False)
'''

print('hello evolution :)')
res = minimize(problem=problem,
               algorithm=algorithm, termination=MyTermination2(1000),
               disp=False)

def is_pareto_efficient(costs, return_mask = True):
	is_efficient = np.arange(costs.shape[0])
	n_points = costs.shape[0]
	next_point_index = 0  # Next index in the is_efficient array to search for
	while next_point_index < len(costs):
		nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
		nondominated_point_mask[next_point_index] = True
		is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
		costs = costs[nondominated_point_mask]
		next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
	if return_mask:
		is_efficient_mask = np.zeros(n_points, dtype=bool)
		is_efficient_mask[is_efficient] = True
		return is_efficient_mask
	else:
		return is_efficient



pareto_ids = is_pareto_efficient(np.array(my_global_variable.satisfied_constraints)*-1.0, False)
print(np.array(my_global_variable.satisfied_constraints)[pareto_ids].tolist())

print("times= " + str(np.array(my_global_variable.times)[pareto_ids].tolist()))
print("iterations= " + str(np.array(my_global_variable.iterations)[pareto_ids].tolist()))